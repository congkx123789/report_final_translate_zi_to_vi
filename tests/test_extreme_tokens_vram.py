import time
import torch
import torch.nn as nn
from config import config
from models.transformer_decoder import StandardTransformerDecoder
from models.micro_head_ssm import EvolvingHeadSSMModel
from models.chunked_loss import ChunkedLinearCrossEntropy

def test_extreme_scaling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print(f"🚀 THỬ NGHIỆM TẢI CỰC HẠN (EXTREME LONG SEQUENCE): ĐO VRAM ĐỈNH & THỜI GIAN")
    print(f"Thiết bị: {torch.cuda.get_device_name(0)} (Blackwell SM120, BF16)")
    print(f"Cấu hình: Batch Size = 2 | Chunked Cross-Entropy (Chunk=512) | Vocab = 40,960")
    print(f"Mục tiêu: Đẩy độ dài chuỗi từ 1,024 lên tới 8,192 tokens để kiểm tra điểm tới hạn OOM")
    print("=" * 95)

    seq_lengths = [1024, 2048, 3072, 4096, 6144, 8192]
    criterion = ChunkedLinearCrossEntropy(chunk_size=512, ignore_index=config.pad_idx)

    results = []
    max_test_len = max(seq_lengths)

    for seq_len in seq_lengths:
        print(f"\n=======================================================")
        print(f"⚡ ĐANG ĐO ĐỘ DÀI CHUỖI CỰC ĐẠI N = {seq_len:,} TOKENS...")
        print(f"=======================================================")
        
        x = torch.randint(4, 40960, (2, seq_len), device=device)
        target = torch.randint(4, 40960, (2, seq_len), device=device)

        # --- 1. Đo Standard Transformer Decoder ---
        vram_tf = None
        t_tf = None
        tf_status = "OK"
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model_tf = StandardTransformerDecoder(
                vocab_size=40960, d_model=512, num_layers=8, num_heads=8, dim_feedforward=2048, 
                max_seq_len=max_test_len, tie_weights=False
            ).to(device)
            model_tf.train()

            # Warmup
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h, _ = model_tf(x, return_hidden=True)
                l = criterion(h, model_tf.generator.weight, target)
            l.backward()
            model_tf.zero_grad()
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h, _ = model_tf(x, return_hidden=True)
                    l = criterion(h, model_tf.generator.weight, target)
                l.backward()
                model_tf.zero_grad()
            torch.cuda.synchronize()
            t_tf = (time.perf_counter() - t0) / 3.0 * 1000.0
            vram_tf = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"   ✓ [Transformer] VRAM: {vram_tf:7.1f} MB | Time: {t_tf:6.1f} ms")
            del model_tf, h, l
        except torch.cuda.OutOfMemoryError:
            tf_status = "💥 OOM (Out Of Memory)"
            print(f"   ✗ [Transformer] 💥 TRÀN VRAM (OOM) ở N = {seq_len:,} tokens!")
        finally:
            torch.cuda.empty_cache()

        # --- 2. Đo Hybrid SSM (Bật Checkpoint) ---
        vram_ssm = None
        t_ssm = None
        ssm_status = "OK"
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model_ssm = EvolvingHeadSSMModel(
                vocab_size=40960, d_model=512, num_micro_heads=8, num_layers=8, dim_feedforward=2048, 
                max_seq_len=max_test_len, pad_idx=config.pad_idx
            ).to(device)
            model_ssm.train()

            # Warmup
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h, _ = model_ssm(x, return_hidden=True, use_checkpointing=True)
                l = criterion(h, model_ssm.generator.weight, target)
            l.backward()
            model_ssm.zero_grad()
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h, _ = model_ssm(x, return_hidden=True, use_checkpointing=True)
                    l = criterion(h, model_ssm.generator.weight, target)
                l.backward()
                model_ssm.zero_grad()
            torch.cuda.synchronize()
            t_ssm = (time.perf_counter() - t0) / 3.0 * 1000.0
            vram_ssm = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"   ✓ [Hybrid SSM]  VRAM: {vram_ssm:7.1f} MB | Time: {t_ssm:6.1f} ms")
            del model_ssm, h, l
        except torch.cuda.OutOfMemoryError:
            ssm_status = "💥 OOM (Out Of Memory)"
            print(f"   ✗ [Hybrid SSM]  💥 TRÀN VRAM (OOM) ở N = {seq_len:,} tokens!")
        finally:
            del x, target
            torch.cuda.empty_cache()

        results.append({
            "seq_len": seq_len,
            "vram_tf": vram_tf,
            "vram_ssm": vram_ssm,
            "t_tf": t_tf,
            "t_ssm": t_ssm,
            "tf_status": tf_status,
            "ssm_status": ssm_status,
        })

    # In Bảng tổng hợp Markdown
    print("\n" + "=" * 100)
    print("📊 BẢNG TỔNG HỢP: ĐỐI SÁNH TẢI CỰC ĐẠI ĐẾN 8,192 TOKENS (NVIDIA RTX 5060 Ti)")
    print("=" * 100)
    print(f"| {'Độ dài N (tokens)':<17} | {'Transformer VRAM':<18} | {'Hybrid SSM VRAM':<18} | {'Transformer Time':<18} | {'Hybrid SSM Time':<18} |")
    print(f"|:{'-'*17}-|-:{'-'*16}:|-:{'-'*16}:|-:{'-'*16}:|-:{'-'*16}:|")
    
    for r in results:
        v_tf_str = f"{r['vram_tf']:.1f} MB" if r['vram_tf'] is not None else r['tf_status']
        v_ssm_str = f"{r['vram_ssm']:.1f} MB" if r['vram_ssm'] is not None else r['ssm_status']
        t_tf_str = f"{r['t_tf']:.1f} ms" if r['t_tf'] is not None else "--"
        t_ssm_str = f"{r['t_ssm']:.1f} ms" if r['t_ssm'] is not None else "--"
        print(f"| {r['seq_len']:<17,d} | {v_tf_str:>16} | {v_ssm_str:>16} | {t_tf_str:>16} | {t_ssm_str:>16} |")
    print("=" * 100)

if __name__ == "__main__":
    test_extreme_scaling()
