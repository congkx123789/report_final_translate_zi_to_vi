#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chuẩn hóa đối chiếu BLEU và chất lượng dịch thuật tại Checkpoint 200,000 steps.
So sánh 100% công bằng (Apple-to-Apple) giữa:
  1. Evolving-Head Hybrid Attention-SSM (checkpoints/ssm_model_step_200000.pt)
  2. Standard Transformer Baseline (checkpoints_transformer/transformer_step_200000.pt)

Tiêu chuẩn đánh giá:
  - Cùng tập dữ liệu test đa miền (50 câu với bản dịch tham chiếu chuẩn tiếng Việt)
  - Cùng bộ Tokenizer (VocabTokenizer, vocab_size=25,905)
  - Cùng định dạng Prompt (<src> zh_text </src> + [EOS, BOS])
  - Cùng giải thuật sinh chuỗi: Greedy Decoding (argmax) + Trigram Repetition Blocking + Suppress UNK
  - Đo đạc chuẩn các chỉ số: SacreBLEU, NLTK BLEU-1/2/3/4, chrF++, ROUGE-1/2/L, Tốc độ sinh từ (tok/s), VRAM tiêu thụ.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import json
import torch
import sacrebleu
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config import config
from vocab_tokenizer import VocabTokenizer
from clean_input import build_prompt, clean_input
from models.micro_head_ssm import EvolvingHeadSSMModel, load_checkpoint_compatible
from models.transformer_decoder import StandardTransformerDecoder

def generate_tokens(model, prompt_ids, device, max_len=128, suppress_unk=True):
    """
    Hàm sinh chuỗi giải mã chuẩn Greedy Decoding với Trigram blocking giống nhau tuyệt đối.
    """
    full_seq = list(prompt_ids)
    gen_ids = []
    
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for step in range(max_len):
            inp = torch.tensor([full_seq], dtype=torch.long, device=device)
            logits, _ = model(inp)
            next_logits = logits[0, -1]
            del logits
            
            if suppress_unk:
                next_logits[config.unk_idx] = float("-inf")
            next_logits[config.pad_idx] = float("-inf")
            
            # Trigram repetition blocking
            if len(gen_ids) >= 3:
                last_bigram = tuple(gen_ids[-2:])
                for i in range(len(gen_ids) - 2):
                    if (gen_ids[i], gen_ids[i+1]) == last_bigram:
                        banned_tok = gen_ids[i+2]
                        next_logits[banned_tok] = float("-inf")
                        
            next_tok = torch.argmax(next_logits, dim=-1).item()
            if next_tok == config.eos_idx or next_tok == config.pad_idx:
                break
                
            gen_ids.append(next_tok)
            full_seq.append(next_tok)
            
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    tok_count = len(gen_ids)
    speed = tok_count / elapsed if elapsed > 0 else 0.0
    return gen_ids, elapsed, tok_count, speed

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("🚀 BẮT ĐẦU CHƯƠNG TRÌNH ĐO CHUẨN BLEU TẠI CHECKPOINT 200,000 STEPS (APPLE-TO-APPLE)")
    print(f"   Thiết bị thực thi: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 100)
    
    tokenizer = VocabTokenizer()
    vocab_size = tokenizer.vocab_size
    
    # 1. Tải mô hình Evolving-Head Hybrid SSM tại Step 200,000
    print("\n[1/4] Khởi tạo mô hình Evolving-Head Hybrid SSM (Đề xuất)...")
    ssm_model = EvolvingHeadSSMModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_micro_heads=config.nhead,
        num_layers=config.num_encoder_layers,
        window_sizes=(5, 20, None),
        dim_feedforward=config.dim_feedforward,
        deep_ssm_expand_factor=2.0,
        dropout=0.0,
        max_seq_len=config.max_seq_len,
        pad_idx=config.pad_idx,
    ).to(device)
    
    ckpt_ssm_path = "checkpoints/ssm_model_step_200000.pt"
    print(f"      Tải trọng số: {ckpt_ssm_path}")
    ckpt_ssm = torch.load(ckpt_ssm_path, map_location=device, weights_only=False)
    load_checkpoint_compatible(ssm_model, ckpt_ssm.get("model_state_dict", ckpt_ssm))
    ssm_model.eval()
    print("      ✓ Evolving-Head Hybrid SSM Step 200k sẵn sàng!")
    
    # 2. Tải mô hình Standard Transformer tại Step 200,000
    print("\n[2/4] Khởi tạo mô hình Standard Transformer Baseline...")
    tf_model = StandardTransformerDecoder(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_layers=config.num_encoder_layers,
        num_heads=config.nhead,
        dim_feedforward=config.dim_feedforward,
        max_seq_len=config.max_seq_len,
        dropout=0.0,
        tie_weights=False,
    ).to(device)
    
    ckpt_tf_path = "checkpoints_transformer/transformer_step_200000.pt"
    print(f"      Tải trọng số: {ckpt_tf_path}")
    ckpt_tf = torch.load(ckpt_tf_path, map_location=device, weights_only=False)
    tf_model.load_state_dict(ckpt_tf.get("model_state_dict", ckpt_tf))
    tf_model.eval()
    print("      ✓ Standard Transformer Step 200k sẵn sàng!")
    
    # 3. Tải tập kiểm thử chuẩn 50 câu đa miền
    print("\n[3/4] Tải tập kiểm thử chuẩn benchmark_test_dataset.json...")
    with open("benchmark_test_dataset.json", "r", encoding="utf-8") as f:
        test_dataset = json.load(f)
    print(f"      Số lượng mẫu kiểm thử: {len(test_dataset)} câu song ngữ Trung - Việt.")
    
    # 4. Thiết lập bộ đo lường
    smoother = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    results = []
    
    print("\n[4/4] Bắt đầu suy luận song song và chấm điểm từng câu...")
    print("-" * 100)
    
    domain_data = {}
    
    for idx, item in enumerate(test_dataset, 1):
        domain = item["domain"]
        zh = item["zh"]
        ref = item["ref"]
        
        if domain not in domain_data:
            domain_data[domain] = {
                "refs": [],
                "ssm_hyps": [],
                "tf_hyps": [],
                "ssm_speeds": [],
                "tf_speeds": [],
                "ssm_times": [],
                "tf_times": [],
                "ssm_vrams": [],
                "tf_vrams": [],
            }
            
        # Chuẩn hóa prompt chuẩn
        prompt_str = build_prompt(zh)
        src_ids = tokenizer.encode_zh(prompt_str, add_bos_eos=False)
        prompt_ids = src_ids + [config.eos_idx, config.bos_idx]
        
        # --- A. Suy luận SSM ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ssm_ids, ssm_time, ssm_toks, ssm_speed = generate_tokens(ssm_model, prompt_ids, device)
        ssm_vram = torch.cuda.max_memory_allocated() / (1024**2)
        ssm_trans = tokenizer.decode_vi(ssm_ids).strip()
        
        # --- B. Suy luận Transformer ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tf_ids, tf_time, tf_toks, tf_speed = generate_tokens(tf_model, prompt_ids, device)
        tf_vram = torch.cuda.max_memory_allocated() / (1024**2)
        tf_trans = tokenizer.decode_vi(tf_ids).strip()
        
        # --- Tính điểm cho từng câu ---
        # 1. SacreBLEU câu đơn
        ssm_sacre = sacrebleu.sentence_bleu(ssm_trans, [ref]).score
        tf_sacre = sacrebleu.sentence_bleu(tf_trans, [ref]).score
        
        # 2. chrF++
        ssm_chrf = sacrebleu.sentence_chrf(ssm_trans, [ref], word_order=2).score
        tf_chrf = sacrebleu.sentence_chrf(tf_trans, [ref], word_order=2).score
        
        # 3. NLTK BLEU-4
        ref_tokens = ref.lower().split()
        ssm_tokens = ssm_trans.lower().split()
        tf_tokens = tf_trans.lower().split()
        
        ssm_nltk_b4 = sentence_bleu([ref_tokens], ssm_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother) * 100
        tf_nltk_b4 = sentence_bleu([ref_tokens], tf_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother) * 100
        
        # 4. ROUGE-L
        ssm_r = rouge.score(ref, ssm_trans)['rougeL'].fmeasure * 100
        tf_r = rouge.score(ref, tf_trans)['rougeL'].fmeasure * 100
        
        # Ghi nhận vào domain data
        domain_data[domain]["refs"].append(ref)
        domain_data[domain]["ssm_hyps"].append(ssm_trans)
        domain_data[domain]["tf_hyps"].append(tf_trans)
        domain_data[domain]["ssm_speeds"].append(ssm_speed)
        domain_data[domain]["tf_speeds"].append(tf_speed)
        domain_data[domain]["ssm_times"].append(ssm_time * 1000)
        domain_data[domain]["tf_times"].append(tf_time * 1000)
        domain_data[domain]["ssm_vrams"].append(ssm_vram)
        domain_data[domain]["tf_vrams"].append(tf_vram)
        
        res_entry = {
            "id": idx,
            "domain": domain,
            "zh": zh,
            "ref": ref,
            "ssm_trans": ssm_trans,
            "tf_trans": tf_trans,
            "ssm_sacrebleu": ssm_sacre,
            "tf_sacrebleu": tf_sacre,
            "ssm_chrf": ssm_chrf,
            "tf_chrf": tf_chrf,
            "ssm_nltk_b4": ssm_nltk_b4,
            "tf_nltk_b4": tf_nltk_b4,
            "ssm_rouge_l": ssm_r,
            "tf_rouge_l": tf_r,
            "ssm_speed": ssm_speed,
            "tf_speed": tf_speed,
            "ssm_time_ms": ssm_time * 1000,
            "tf_time_ms": tf_time * 1000,
            "ssm_vram_mb": ssm_vram,
            "tf_vram_mb": tf_vram,
        }
        results.append(res_entry)
        
        # In log trực quan
        print(f"[{idx:02d}/50] ({domain})")
        print(f"🇨🇳 ZH : {zh}")
        print(f"🎯 REF: {ref}")
        print(f"🟢 SSM: {ssm_trans} | BLEU: {ssm_sacre:5.2f} | chrF++: {ssm_chrf:5.2f} | {ssm_speed:5.1f} tok/s | {ssm_vram:5.1f}MB")
        print(f"🔴 TF : {tf_trans} | BLEU: {tf_sacre:5.2f} | chrF++: {tf_chrf:5.2f} | {tf_speed:5.1f} tok/s | {tf_vram:5.1f}MB")
        print("-" * 100)
        sys.stdout.flush()

    # =========================================================================
    # TỔNG HỢP KẾT QUẢ ĐO TOÀN TẬP VÀ THEO TỪNG MIỀN
    # =========================================================================
    all_refs = [r["ref"] for r in results]
    all_ssm_hyps = [r["ssm_trans"] for r in results]
    all_tf_hyps = [r["tf_trans"] for r in results]
    
    # Corpus SacreBLEU toàn tập
    total_ssm_sacre = sacrebleu.corpus_bleu(all_ssm_hyps, [all_refs]).score
    total_tf_sacre = sacrebleu.corpus_bleu(all_tf_hyps, [all_refs]).score
    
    # Corpus chrF++ toàn tập
    total_ssm_chrf = sacrebleu.corpus_chrf(all_ssm_hyps, [all_refs], word_order=2).score
    total_tf_chrf = sacrebleu.corpus_chrf(all_tf_hyps, [all_refs], word_order=2).score
    
    # NLTK Corpus BLEU-1,2,3,4
    list_of_refs = [[r.lower().split()] for r in all_refs]
    ssm_tokenized_hyps = [h.lower().split() for h in all_ssm_hyps]
    tf_tokenized_hyps = [h.lower().split() for h in all_tf_hyps]
    
    total_ssm_b1 = corpus_bleu(list_of_refs, ssm_tokenized_hyps, weights=(1, 0, 0, 0), smoothing_function=smoother) * 100
    total_tf_b1 = corpus_bleu(list_of_refs, tf_tokenized_hyps, weights=(1, 0, 0, 0), smoothing_function=smoother) * 100
    
    total_ssm_b2 = corpus_bleu(list_of_refs, ssm_tokenized_hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother) * 100
    total_tf_b2 = corpus_bleu(list_of_refs, tf_tokenized_hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother) * 100
    
    total_ssm_b3 = corpus_bleu(list_of_refs, ssm_tokenized_hyps, weights=(0.333, 0.333, 0.334, 0), smoothing_function=smoother) * 100
    total_tf_b3 = corpus_bleu(list_of_refs, tf_tokenized_hyps, weights=(0.333, 0.333, 0.334, 0), smoothing_function=smoother) * 100
    
    total_ssm_b4 = corpus_bleu(list_of_refs, ssm_tokenized_hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother) * 100
    total_tf_b4 = corpus_bleu(list_of_refs, tf_tokenized_hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother) * 100
    
    # Trung bình ROUGE-L
    avg_ssm_rouge = sum(r["ssm_rouge_l"] for r in results) / len(results)
    avg_tf_rouge = sum(r["tf_rouge_l"] for r in results) / len(results)
    
    # Tốc độ & VRAM
    avg_ssm_speed = sum(r["ssm_speed"] for r in results) / len(results)
    avg_tf_speed = sum(r["tf_speed"] for r in results) / len(results)
    
    avg_ssm_time = sum(r["ssm_time_ms"] for r in results) / len(results)
    avg_tf_time = sum(r["tf_time_ms"] for r in results) / len(results)
    
    avg_ssm_vram = sum(r["ssm_vram_mb"] for r in results) / len(results)
    avg_tf_vram = sum(r["tf_vram_mb"] for r in results) / len(results)

    # Thống kê theo từng Domain
    domain_summary = {}
    for d_name, d_dict in domain_data.items():
        d_sacre_ssm = sacrebleu.corpus_bleu(d_dict["ssm_hyps"], [d_dict["refs"]]).score
        d_sacre_tf = sacrebleu.corpus_bleu(d_dict["tf_hyps"], [d_dict["refs"]]).score
        d_chrf_ssm = sacrebleu.corpus_chrf(d_dict["ssm_hyps"], [d_dict["refs"]], word_order=2).score
        d_chrf_tf = sacrebleu.corpus_chrf(d_dict["tf_hyps"], [d_dict["refs"]], word_order=2).score
        
        domain_summary[d_name] = {
            "count": len(d_dict["refs"]),
            "ssm_sacrebleu": d_sacre_ssm,
            "tf_sacrebleu": d_sacre_tf,
            "diff_bleu": d_sacre_ssm - d_sacre_tf,
            "ssm_chrf": d_chrf_ssm,
            "tf_chrf": d_chrf_tf,
            "ssm_speed": sum(d_dict["ssm_speeds"]) / len(d_dict["ssm_speeds"]),
            "tf_speed": sum(d_dict["tf_speeds"]) / len(d_dict["tf_speeds"]),
            "speedup": (sum(d_dict["ssm_speeds"]) / len(d_dict["ssm_speeds"])) / (sum(d_dict["tf_speeds"]) / len(d_dict["tf_speeds"])),
            "ssm_vram": sum(d_dict["ssm_vrams"]) / len(d_dict["ssm_vrams"]),
            "tf_vram": sum(d_dict["tf_vrams"]) / len(d_dict["tf_vrams"]),
            "vram_saving": (1.0 - (sum(d_dict["ssm_vrams"]) / len(d_dict["ssm_vrams"])) / (sum(d_dict["tf_vrams"]) / len(d_dict["tf_vrams"]))) * 100,
        }

    # In Bảng Báo Cáo
    print("\n" + "=" * 100)
    print("📊 BÁO CÁO TỔNG KẾT ĐỐI CHIẾU BLEU TẠI CHECKPOINT 200,000 STEPS")
    print("=" * 100)
    print(f"{'Chỉ số đánh giá':<30} | {'Transformer (Baseline)':<22} | {'Hybrid SSM (Đề xuất)':<22} | {'Chênh lệch':<15}")
    print("-" * 100)
    print(f"{'Corpus SacreBLEU (Toàn tập)':<30} | {total_tf_sacre:<22.2f} | {total_ssm_sacre:<22.2f} | {total_ssm_sacre - total_tf_sacre:+6.2f} BLEU")
    print(f"{'Corpus chrF++ (Toàn tập)':<30} | {total_tf_chrf:<22.2f} | {total_ssm_chrf:<22.2f} | {total_ssm_chrf - total_tf_chrf:+6.2f} pts")
    print(f"{'NLTK BLEU-1':<30} | {total_tf_b1:<22.2f} | {total_ssm_b1:<22.2f} | {total_ssm_b1 - total_tf_b1:+6.2f} pts")
    print(f"{'NLTK BLEU-2':<30} | {total_tf_b2:<22.2f} | {total_ssm_b2:<22.2f} | {total_ssm_b2 - total_tf_b2:+6.2f} pts")
    print(f"{'NLTK BLEU-3':<30} | {total_tf_b3:<22.2f} | {total_ssm_b3:<22.2f} | {total_ssm_b3 - total_tf_b3:+6.2f} pts")
    print(f"{'NLTK BLEU-4':<30} | {total_tf_b4:<22.2f} | {total_ssm_b4:<22.2f} | {total_ssm_b4 - total_tf_b4:+6.2f} pts")
    print(f"{'ROUGE-L Score':<30} | {avg_tf_rouge:<22.2f} | {avg_ssm_rouge:<22.2f} | {avg_ssm_rouge - avg_tf_rouge:+6.2f} pts")
    print(f"{'Tốc độ sinh (tokens/s)':<30} | {avg_tf_speed:<22.1f} | {avg_ssm_speed:<22.1f} | {avg_ssm_speed/avg_tf_speed:5.2f}x Nhanh hơn")
    print(f"{'Độ trễ trung bình (ms/câu)':<30} | {avg_tf_time:<22.1f} | {avg_ssm_time:<22.1f} | {avg_tf_time - avg_ssm_time:+6.1f} ms")
    print(f"{'VRAM tiêu thụ đỉnh (MB)':<30} | {avg_tf_vram:<22.1f} | {avg_ssm_vram:<22.1f} | {-(1.0 - avg_ssm_vram/avg_tf_vram)*100:6.1f}% Tiết kiệm")
    print("-" * 100)
    
    print("\n📋 CHI TIẾT THEO TỪNG MIỀN DỮ LIỆU:")
    print(f"{'Miền dữ liệu':<32} | {'TF SacreBLEU':<14} | {'SSM SacreBLEU':<14} | {'BLEU Diff':<12} | {'Tăng tốc':<10}")
    print("-" * 90)
    for d_name, d_stat in domain_summary.items():
        print(f"{d_name:<32} | {d_stat['tf_sacrebleu']:<14.2f} | {d_stat['ssm_sacrebleu']:<14.2f} | {d_stat['diff_bleu']:+8.2f}     | {d_stat['speedup']:4.2f}x")
    print("=" * 100)
    
    # Lưu file JSON và file Markdown
    final_output = {
        "metadata": {
            "step": 200000,
            "ssm_checkpoint": ckpt_ssm_path,
            "transformer_checkpoint": ckpt_tf_path,
            "num_samples": len(results),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "eval_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "overall_metrics": {
            "sacrebleu": {"transformer": total_tf_sacre, "ssm": total_ssm_sacre, "diff": total_ssm_sacre - total_tf_sacre},
            "chrf_plus": {"transformer": total_tf_chrf, "ssm": total_ssm_chrf, "diff": total_ssm_chrf - total_tf_chrf},
            "nltk_bleu_1": {"transformer": total_tf_b1, "ssm": total_ssm_b1, "diff": total_ssm_b1 - total_tf_b1},
            "nltk_bleu_2": {"transformer": total_tf_b2, "ssm": total_ssm_b2, "diff": total_ssm_b2 - total_tf_b2},
            "nltk_bleu_3": {"transformer": total_tf_b3, "ssm": total_ssm_b3, "diff": total_ssm_b3 - total_tf_b3},
            "nltk_bleu_4": {"transformer": total_tf_b4, "ssm": total_ssm_b4, "diff": total_ssm_b4 - total_tf_b4},
            "rouge_l": {"transformer": avg_tf_rouge, "ssm": avg_ssm_rouge, "diff": avg_ssm_rouge - avg_tf_rouge},
            "tokens_per_sec": {"transformer": avg_tf_speed, "ssm": avg_ssm_speed, "speedup": avg_ssm_speed / avg_tf_speed},
            "latency_ms": {"transformer": avg_tf_time, "ssm": avg_ssm_time, "reduction": avg_tf_time - avg_ssm_time},
            "peak_vram_mb": {"transformer": avg_tf_vram, "ssm": avg_ssm_vram, "savings_percent": (1.0 - avg_ssm_vram / avg_tf_vram) * 100}
        },
        "domain_summary": domain_summary,
        "sample_details": results
    }
    
    json_path = "benchmark_bleu_200k_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu kết quả chi tiết dạng JSON: {json_path}")
    
    # Tạo báo cáo Markdown
    md_path = "benchmark_bleu_200k_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🏆 BÁO CÁO ĐỐI CHIẾU CHẤT LƯỢNG DỊCH THUẬT VÀ BLEU SCORE TẠI CHECKPOINT 200,000 STEPS\n\n")
        f.write(f"- **Mô hình A (Đề xuất):** `EvolvingHeadSSMModel` (`checkpoints/ssm_model_step_200000.pt`, 71.97M params)\n")
        f.write(f"- **Mô hình B (Baseline):** `StandardTransformerDecoder` (`checkpoints_transformer/transformer_step_200000.pt`, 68.2M params)\n")
        f.write(f"- **Phần cứng:** {torch.cuda.get_device_name(0)}\n")
        f.write(f"- **Số lượng câu test:** {len(results)} câu song ngữ đa miền\n")
        f.write(f"- **Thời gian chạy test:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Bảng Tổng Hợp Chỉ Số Đánh Giá Toàn Tập\n\n")
        f.write("| Chỉ số đánh giá | Transformer (Baseline) | Evolving-Head Hybrid SSM (Đề xuất) | Chênh lệch / Ưu thế |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Corpus SacreBLEU** | **{total_tf_sacre:.2f}** | **{total_ssm_sacre:.2f}** | **{total_ssm_sacre - total_tf_sacre:+.2f} BLEU** |\n")
        f.write(f"| **Corpus chrF++** | {total_tf_chrf:.2f} | {total_ssm_chrf:.2f} | {total_ssm_chrf - total_tf_chrf:+.2f} pts |\n")
        f.write(f"| **NLTK BLEU-1** | {total_tf_b1:.2f} | {total_ssm_b1:.2f} | {total_ssm_b1 - total_tf_b1:+.2f} pts |\n")
        f.write(f"| **NLTK BLEU-2** | {total_tf_b2:.2f} | {total_ssm_b2:.2f} | {total_ssm_b2 - total_tf_b2:+.2f} pts |\n")
        f.write(f"| **NLTK BLEU-3** | {total_tf_b3:.2f} | {total_ssm_b3:.2f} | {total_ssm_b3 - total_tf_b3:+.2f} pts |\n")
        f.write(f"| **NLTK BLEU-4** | {total_tf_b4:.2f} | {total_ssm_b4:.2f} | {total_ssm_b4 - total_tf_b4:+.2f} pts |\n")
        f.write(f"| **ROUGE-L Score** | {avg_tf_rouge:.2f} | {avg_ssm_rouge:.2f} | {avg_ssm_rouge - avg_tf_rouge:+.2f} pts |\n")
        f.write(f"| **Tốc độ sinh (tokens/s)** | {avg_tf_speed:.1f} tok/s | {avg_ssm_speed:.1f} tok/s | **{avg_ssm_speed/avg_tf_speed:.2f}x Nhanh hơn** |\n")
        f.write(f"| **Độ trễ (Latency)** | {avg_tf_time:.1f} ms | {avg_ssm_time:.1f} ms | {avg_tf_time - avg_ssm_time:+.1f} ms nhanh hơn |\n")
        f.write(f"| **VRAM tiêu thụ** | {avg_tf_vram:.1f} MB | {avg_ssm_vram:.1f} MB | **{(1.0 - avg_ssm_vram/avg_tf_vram)*100:.1f}% Tiết kiệm** |\n\n")
        
        f.write("## 2. Bảng Phân Tích Theo Từng Miền Dữ Liệu\n\n")
        f.write("| Miền dữ liệu | Số câu | Transformer BLEU | Hybrid SSM BLEU | Chênh lệch BLEU | Tốc độ SSM | Tăng tốc độ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for d_name, d_stat in domain_summary.items():
            f.write(f"| **{d_name}** | {d_stat['count']} | {d_stat['tf_sacrebleu']:.2f} | {d_stat['ssm_sacrebleu']:.2f} | **{d_stat['diff_bleu']:+.2f}** | {d_stat['ssm_speed']:.1f} tok/s | **{d_stat['speedup']:.2f}x** |\n")
            
        f.write("\n## 3. Trích Xuất 10 Cặp Câu Song Ngữ Điển Hình\n\n")
        for r in results[:10]:
            f.write(f"### Câu #{r['id']} ({r['domain']})\n")
            f.write(f"- 🇨🇳 **Nguyên tác Tiếng Trung:** `{r['zh']}`\n")
            f.write(f"- 🎯 **Bản dịch chuẩn (Ground Truth):** {r['ref']}\n")
            f.write(f"- 🟢 **Evolving-Head Hybrid SSM:** {r['ssm_trans']} *(BLEU: {r['ssm_sacrebleu']:.1f} | {r['ssm_speed']:.1f} tok/s)*\n")
            f.write(f"- 🔴 **Standard Transformer:** {r['tf_trans']} *(BLEU: {r['tf_sacrebleu']:.1f} | {r['tf_speed']:.1f} tok/s)*\n\n")
            
    print(f"✓ Đã xuất báo cáo Markdown hoàn chỉnh: {md_path}")
    print("=" * 100)

if __name__ == "__main__":
    main()
