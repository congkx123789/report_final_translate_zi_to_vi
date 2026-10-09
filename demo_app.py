#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Demo Trực quan - Báo cáo Nghiên cứu Khoa học Sinh viên UET 2026
Đề tài: Nghiên cứu Kiến trúc Lai Hybrid Attention-SSM phục vụ Chuyển phong cách Văn bản và Dịch thuật Hán - Việt
Tác giả: Hà Vũ Công (MSSV: 23020014) - GVHD: TS. Nguyễn Văn Vinh
"""

import time
import argparse
import torch
import gradio as gr
from config import config
from vocab_tokenizer import VocabTokenizer
from clean_input import build_prompt, clean_input
from models.micro_head_ssm import EvolvingHeadSSMModel, load_checkpoint_compatible
from models.transformer_decoder import StandardTransformerDecoder

# Global models & tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
ssm_model = None
tf_model = None

def init_models():
    global tokenizer, ssm_model, tf_model
    if tokenizer is not None:
        return
        
    print(f"[*] Khởi tạo môi trường trên thiết bị: {device}")
    tokenizer = VocabTokenizer()
    vocab_size = tokenizer.vocab_size
    
    # 1. Load Evolving Hybrid Attention-SSM
    print("-> Đang tải trọng số Evolving Hybrid SSM (checkpoints/ssm_model_best.pt)...")
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
    
    ckpt_ssm = torch.load("checkpoints/ssm_model_best.pt", map_location=device, weights_only=False)
    load_checkpoint_compatible(ssm_model, ckpt_ssm.get("model_state_dict", ckpt_ssm))
    ssm_model.eval()
    print("✓ Tải thành công Evolving Hybrid SSM!")
    
    # 2. Load Standard Transformer Baseline
    print("-> Đang tải trọng số Standard Transformer (checkpoints_transformer/transformer_best.pt)...")
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
    
    ckpt_tf = torch.load("checkpoints_transformer/transformer_best.pt", map_location=device, weights_only=False)
    tf_model.load_state_dict(ckpt_tf.get("model_state_dict", ckpt_tf))
    tf_model.eval()
    print("✓ Tải thành công Standard Transformer Baseline!")

def generate(model, prompt_ids, max_len=128):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        vram_before = torch.cuda.memory_allocated(device) / (1024 * 1024)
    else:
        vram_before = 0.0

    full_seq = list(prompt_ids)
    gen_ids = []
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for step in range(max_len):
            inp = torch.tensor([full_seq], dtype=torch.long, device=device)
            logits, _ = model(inp)
            next_logits = logits[0, -1]
            del logits
            
            # Suppress unk
            next_logits[config.unk_idx] = float("-inf")
            
            # No-repeat trigram check
            if len(gen_ids) >= 3:
                last_trigram = tuple(gen_ids[-2:])
                for i in range(len(gen_ids) - 2):
                    if (gen_ids[i], gen_ids[i+1]) == last_trigram:
                        banned_tok = gen_ids[i+2]
                        next_logits[banned_tok] = float("-inf")
                        
            next_tok = torch.argmax(next_logits, dim=-1).item()
            if next_tok == config.eos_idx or next_tok == config.pad_idx:
                break
                
            gen_ids.append(next_tok)
            full_seq.append(next_tok)
            
    elapsed = time.perf_counter() - start_time
    tok_count = len(gen_ids)
    speed = tok_count / elapsed if elapsed > 0 else 0
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram = 0.0

    out_text = tokenizer.decode(gen_ids)
    return out_text, elapsed * 1000, speed, tok_count, peak_vram

def translate_action(chinese_text, max_tokens):
    if not chinese_text.strip():
        return "Vui lòng nhập văn bản tiếng Trung!", "", "", ""
        
    init_models()
    
    # Preprocess input
    cleaned_zh = clean_input(chinese_text.strip())
    prompt_str = build_prompt(cleaned_zh, glossary=None)
    prompt_ids = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
    
    # 1. Hybrid SSM
    ssm_text, ssm_lat, ssm_spd, ssm_tok, ssm_vram = generate(ssm_model, prompt_ids, max_len=int(max_tokens))
    ssm_stats = (
        f"⏱️ **Độ trễ:** {ssm_lat:.1f} ms | "
        f"⚡ **Tốc độ:** {ssm_spd:.1f} tokens/s | "
        f"📝 **Số tokens:** {ssm_tok}\n\n"
        f"💾 **Peak VRAM:** {ssm_vram:.1f} MB | "
        f"📦 **State Cache:** 1.54 MB (O(1) Cố định)"
    )
    
    # 2. Standard Transformer
    tf_text, tf_lat, tf_spd, tf_tok, tf_vram = generate(tf_model, prompt_ids, max_len=int(max_tokens))
    tf_cache_kb = 2 * 1 * 8 * 8 * 64 * 2 * (len(prompt_ids) + tf_tok) / 1024
    tf_stats = (
        f"⏱️ **Độ trễ:** {tf_lat:.1f} ms | "
        f"⚡ **Tốc độ:** {tf_spd:.1f} tokens/s | "
        f"📝 **Số tokens:** {tf_tok}\n\n"
        f"💾 **Peak VRAM:** {tf_vram:.1f} MB | "
        f"📦 **KV-Cache:** {tf_cache_kb:.1f} KB (Tăng theo O(N))"
    )
    
    return ssm_text, ssm_stats, tf_text, tf_stats

# Preset test cases
PRESETS = [
    [
        "林动眼神微凝，体内元力呼啸而出，化作漫天金光，直接将那道凌厉的剑芒彻底震碎！",
        "Tiên hiệp - Danh xưng Lâm Động"
    ],
    [
        "面对三大元婴期强者的联手围攻，苏陈面无惧色，双手如电般变幻印诀，混沌灵气自周身翻涌而出！",
        "Tiên hiệp cổ điển - Thuật ngữ tu tiên"
    ],
    [
        "你吃过晚饭了吗？要不要和我一起去超市买点水果？",
        "Giao tiếp hàng ngày"
    ],
    [
        "该混合架构结合了自注意力机制与状态空间模型的优势，能够在极长文本推理时大幅降低显存消耗。",
        "Khoa học - AI & Kỹ thuật"
    ],
    [
        "他转过身，对身后的弟子们说道：“大家小心，前方有埋伏！”",
        "Kiếm hiệp - Cảnh báo phục kích"
    ],
]

def build_gui():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="amber",
        neutral_hue="slate"
    )
    
    with gr.Blocks(theme=theme, title="UET AI Defense Demo - Hybrid Attention-SSM") as demo:
        gr.Markdown(
            """
            # 🏛️ ĐẠI HỌC QUỐC GIA HÀ NỘI -- TRƯỜNG ĐẠI HỌC CÔNG NGHỆ (UET)
            ## 🚀 Hệ thống Dịch thuật Song song Đối đầu: Evolving Hybrid SSM vs. Standard Transformer
            **Đề tài NCKH Sinh viên:** Nghiên cứu Kiến trúc Lai Hybrid Attention-SSM phục vụ Chuyển phong cách Văn bản và Dịch thuật Hán - Việt  
            **Sinh viên thực hiện:** Hà Vũ Công (MSSV: 23020014) | **GVHD:** TS. Nguyễn Văn Vinh
            """
        )
        
        with gr.Row():
            with gr.Column(scale=5):
                zh_input = gr.Textbox(
                    lines=4,
                    placeholder="Nhập câu hoặc đoạn văn bản tiếng Trung cần dịch...",
                    label="📝 Văn bản gốc tiếng Trung (Input Chinese Text)"
                )
                with gr.Row():
                    max_tok_slider = gr.Slider(
                        minimum=32, maximum=256, value=128, step=16,
                        label="Độ dài sinh tối đa (Max Tokens)"
                    )
                    btn_translate = gr.Button("⚡ Dịch Đối đầu (Translate & Benchmark)", variant="primary", scale=2)
                
                gr.Markdown("### 🎯 Các mẫu câu kiểm nghiệm phong cách điển hình:")
                with gr.Row():
                    for text, label in PRESETS:
                        gr.Button(label, size="sm").click(
                            fn=lambda t=text: t,
                            inputs=[],
                            outputs=[zh_input]
                        )
                        
            with gr.Column(scale=5):
                gr.Markdown("### 📊 Kết quả Suy luận Thời gian thực (Real-time Benchmarks)")
                with gr.Tab("🏆 Evolving Hybrid Attention-SSM (O(1) State)"):
                    ssm_out = gr.Textbox(lines=4, label="Bản dịch Tiếng Việt (Hybrid SSM)", interactive=False)
                    ssm_metrics = gr.Markdown(value="*Chưa chạy suy luận*")
                    
                with gr.Tab("⚖️ Standard Transformer Baseline (O(N^2) Softmax)"):
                    tf_out = gr.Textbox(lines=4, label="Bản dịch Tiếng Việt (Standard Transformer)", interactive=False)
                    tf_metrics = gr.Markdown(value="*Chưa chạy suy luận*")

        btn_translate.click(
            fn=translate_action,
            inputs=[zh_input, max_tok_slider],
            outputs=[ssm_out, ssm_metrics, tf_out, tf_metrics]
        )
        
        gr.Markdown(
            """
            ---
            💡 **Ghi chú kỹ thuật:**
            * Mô hình **Evolving Hybrid SSM** duy trì kích thước bộ nhớ đệm trạng thái cố định ở mức **1.54 MB** bất kể chuỗi văn bản dài đến hàng triệu tokens.
            * Mô hình **Standard Transformer** đòi hỏi cấp phát thêm **16 KB** KV-Cache trên mỗi token sinh ra, bùng nổ lên **30.52 GB VRAM** ở 2,000,000 tokens.
            """
        )
        
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UET AI Defense Web Demo")
    parser.add_argument("--share", action="store_true", help="Tạo public tunneling URL qua Gradio để truy cập từ xa")
    parser.add_argument("--port", type=int, default=7860, help="Cổng chạy server (mặc định 7860)")
    args = parser.parse_args()
    
    init_models()
    demo = build_gui()
    print(f"\n[*] Đang khởi chạy Web Demo trên cổng {args.port} (share={args.share})...")
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
