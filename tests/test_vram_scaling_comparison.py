import time
import torch
import torch.nn as nn
from config import config
from models.transformer_decoder import StandardTransformerDecoder
from models.micro_head_ssm import EvolvingHeadSSMModel
from models.chunked_loss import ChunkedLinearCrossEntropy

def test_scaling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print(f"🔬 ĐỐI SÁNH TRỰC TIẾP: VRAM ĐỈNH & THỜI GIAN THEO ĐỘ DÀI CHUỖI (SEQUENCE LENGTH)")
    print(f"Thiết bị: {torch.cuda.get_device_name(0)} (Blackwell SM120, BF16)")
    print(f"Cấu hình: Batch Size = 4 | Chunked Cross-Entropy (Chunk=512) | Vocab = 40,960")
    print("=" * 85)

    seq_lengths = [128, 256, 512, 1024, 1536, 2048]
    criterion = ChunkedLinearCrossEntropy(chunk_size=512, ignore_index=config.pad_idx)

    results = []

    for seq_len in seq_lengths:
        print(f"\n>>> Đang đo độ dài chuỗi N = {seq_len} tokens...")
        
        # --- 1. Đo Standard Transformer Decoder ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model_tf = StandardTransformerDecoder(
            vocab_size=40960, d_model=512, num_layers=8, num_heads=8, dim_feedforward=2048, max_seq_len=2048, tie_weights=False
        ).to(device)
        model_tf.train()
        
        x = torch.randint(4, 40960, (4, seq_len), device=device)
        target = torch.randint(4, 40960, (4, seq_len), device=device)
        
        # Warmup
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h, _ = model_tf(x, return_hidden=True)
            l = criterion(h, model_tf.generator.weight, target)
        l.backward()
        model_tf.zero_grad()
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        for _ in range(5):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h, _ = model_tf(x, return_hidden=True)
                l = criterion(h, model_tf.generator.weight, target)
            l.backward()
            model_tf.zero_grad()
            
        torch.cuda.synchronize()
        t_tf = (time.perf_counter() - t0) / 5.0 * 1000.0
        vram_tf = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        del model_tf, x, target, h, l
        torch.cuda.empty_cache()

        # --- 2. Đo Evolving-Head Hybrid Attention-SSM ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model_ssm = EvolvingHeadSSMModel(
            vocab_size=40960, d_model=512, num_micro_heads=8, num_layers=8, dim_feedforward=2048, max_seq_len=2048, pad_idx=config.pad_idx
        ).to(device)
        model_ssm.train()
        
        x = torch.randint(4, 40960, (4, seq_len), device=device)
        target = torch.randint(4, 40960, (4, seq_len), device=device)
        
        # Warmup
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h, _ = model_ssm(x, return_hidden=True)
            l = criterion(h, model_ssm.generator.weight, target)
        l.backward()
        model_ssm.zero_grad()
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        for _ in range(5):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h, _ = model_ssm(x, return_hidden=True)
                l = criterion(h, model_ssm.generator.weight, target)
            l.backward()
            model_ssm.zero_grad()
            
        torch.cuda.synchronize()
        t_ssm = (time.perf_counter() - t0) / 5.0 * 1000.0
        vram_ssm = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        del model_ssm, x, target, h, l
        torch.cuda.empty_cache()

        results.append({
            "seq_len": seq_len,
            "tf_vram": vram_tf,
            "ssm_vram": vram_ssm,
            "tf_time": t_tf,
            "ssm_time": t_ssm,
        })
        print(f"   [N={seq_len:4d}] Transformer: VRAM={vram_tf:6.1f}MB, Time={t_tf:5.1f}ms | Hybrid SSM: VRAM={vram_ssm:6.1f}MB, Time={t_ssm:5.1f}ms")

    # In Bảng tổng hợp Markdown
    print("\n" + "=" * 90)
    print("📊 BẢNG TỔNG HỢP SO SÁNH VRAM ĐỈNH & THỜI GIAN BƯỚC (STEP TIME) THEO ĐỘ DÀI CHUỖI N")
    print("=" * 90)
    print(f"| {'Độ dài N (tokens)':<17} | {'Transformer VRAM':<16} | {'Hybrid SSM VRAM':<15} | {'Transformer Time':<16} | {'Hybrid SSM Time':<15} | {'Tăng trưởng VRAM (TF vs SSM)':<26} |")
    print(f"|:{'-'*17}-|-:{'-'*15}:|-:{'-'*14}:|-:{'-'*15}:|-:{'-'*14}:|-:{'-'*26}:|")
    
    base_tf_vram = results[0]["tf_vram"]
    base_ssm_vram = results[0]["ssm_vram"]
    
    for r in results:
        tf_v_growth = (r["tf_vram"] - base_tf_vram)
        ssm_v_growth = (r["ssm_vram"] - base_ssm_vram)
        growth_str = f"+{tf_v_growth:.0f}MB (TF) vs +{ssm_v_growth:.0f}MB (SSM)"
        print(f"| {r['seq_len']:<17} | {r['tf_vram']:>13.1f} MB | {r['ssm_vram']:>12.1f} MB | {r['tf_time']:>13.1f} ms | {r['ssm_time']:>12.1f} ms | {growth_str:<26} |")
    print("=" * 90)

if __name__ == "__main__":
    test_scaling()
