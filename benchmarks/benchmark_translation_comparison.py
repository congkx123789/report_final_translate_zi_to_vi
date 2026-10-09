import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import torch
import torch.nn.functional as F
from config import config
from vocab_tokenizer import VocabTokenizer
from clean_input import build_prompt, clean_input
from models.micro_head_ssm import EvolvingHeadSSMModel, load_checkpoint_compatible
from models.transformer_decoder import StandardTransformerDecoder

def generate_translation(model, prompt_ids, device, max_len=128, suppress_unk=True):
    full_seq = list(prompt_ids)
    gen_ids = []
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for step in range(max_len):
            inp = torch.tensor([full_seq], dtype=torch.long, device=device)
            logits, _ = model(inp)
            next_logits = logits[0, -1]
            del logits
            
            if suppress_unk:
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
    return gen_ids, elapsed, tok_count, speed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Models on {device}...")
    tokenizer = VocabTokenizer()
    vocab_size = tokenizer.vocab_size
    
    # 1. Load SSM Model
    print("-> Initializing Evolving Hybrid Attention-SSM...")
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
    print("✓ Evolving SSM Loaded!")
    
    # 2. Load Transformer Model
    print("-> Initializing Standard Transformer Baseline...")
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
    print("✓ Standard Transformer Loaded!")
    
    # Test suite covering diverse domains
    test_cases = [
        {
            "category": "1. Giao tiếp thường nhật (Daily Life)",
            "zh": "你吃过晚饭了吗？要不要和我一起去超市买点水果？",
            "glossary": None,
        },
        {
            "category": "2. Tiên hiệp / Huyền huyễn (Webnovel / Cultivation)",
            "zh": "林动眼神微凝，体内元力呼啸而出，化作漫天金光，直接将那道凌厉的剑芒彻底震碎！",
            "glossary": None,
        },
        {
            "category": "3. Công nghệ / Trí tuệ nhân tạo (AI & Tech)",
            "zh": "该混合架构结合了自注意力机制与状态空间模型的优势，能够在极长文本推理时大幅降低显存消耗。",
            "glossary": None,
        },
        {
            "category": "4. Thành ngữ & Văn phong cổ (Idiom & Literature)",
            "zh": "千里之行，始于足下。面对前方的艰难险阻，我们绝不能半途而废。",
            "glossary": None,
        },
        {
            "category": "5. Câu ghép logic & Cảm xúc (Complex Compound Sentence)",
            "zh": "虽然这次实验失败了多次，但每一次错误都让我们更接近最终的真相，因此没有人打算放弃。",
            "glossary": None,
        },
        {
            "category": "6. Dịch có định hướng thuật ngữ (Glossary Constrained)",
            "zh": "主角施展了九天玄雷诀，击败了黑煞宗的长老。",
            "glossary": {"九天玄雷诀": "Cửu Thiên Huyền Lôi Quyết", "黑煞宗": "Hắc Sát Tông"},
        }
    ]
    
    print("\n" + "="*90)
    print("⚡ BẮT ĐẦU ĐỐI SÁNH DỊCH THUẬT GIỮA HYBRID SSM VÀ TRANSFORMER")
    print("="*90)
    
    results = []
    for idx, tc in enumerate(test_cases, 1):
        zh_text = tc["zh"]
        glossary = tc["glossary"]
        cat = tc["category"]
        
        # Build prompt
        if glossary:
            prompt_str = build_prompt(zh_text, glossary)
        else:
            prompt_str = build_prompt(zh_text)
            
        src_ids = tokenizer.encode_zh(prompt_str, add_bos_eos=False)
        prompt_ids = src_ids + [config.eos_idx, config.bos_idx]
        
        # 1. Infer SSM
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ssm_ids, ssm_time, ssm_toks, ssm_speed = generate_translation(ssm_model, prompt_ids, device)
        ssm_vram = torch.cuda.max_memory_allocated() / (1024**2)
        ssm_vi = tokenizer.decode_vi(ssm_ids)
        
        # 2. Infer Transformer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tf_ids, tf_time, tf_toks, tf_speed = generate_translation(tf_model, prompt_ids, device)
        tf_vram = torch.cuda.max_memory_allocated() / (1024**2)
        tf_vi = tokenizer.decode_vi(tf_ids)
        
        results.append({
            "category": cat,
            "zh": zh_text,
            "ssm_vi": ssm_vi,
            "tf_vi": tf_vi,
            "ssm_speed": ssm_speed,
            "tf_speed": tf_speed,
            "ssm_vram": ssm_vram,
            "tf_vram": tf_vram,
        })
        
        print(f"\n[{cat}]")
        print(f"🇨🇳 Tiếng Trung: {zh_text}")
        if glossary:
            print(f"📖 Glossary: {glossary}")
        print(f"🔵 Evolving SSM : {ssm_vi}")
        print(f"   (Tốc độ: {ssm_speed:.1f} tok/s | Thời gian: {ssm_time*1000:.1f}ms)")
        print(f"🔴 Transformer  : {tf_vi}")
        print(f"   (Tốc độ: {tf_speed:.1f} tok/s | Thời gian: {tf_time*1000:.1f}ms)")
        print("-" * 90)

if __name__ == "__main__":
    main()
