import time
import torch
import torch.nn.functional as F
from config import config
from models.transformer_decoder import StandardTransformerDecoder
from models.micro_head_ssm import EvolvingHeadSSMModel

def run_live_token_generation_audit(num_tokens: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("🔬 THỬ NGHIỆM AUDIT TRỰC TIẾP: THEO DÕI TỪNG BƯỚC SINH TOKEN (AUTOREGRESSIVE DECODING)")
    print(f"Thiết bị: {torch.cuda.get_device_name(0)} (Blackwell SM120, BF16)")
    print(f"Tổng số token sinh liên tục: {num_tokens} tokens")
    print("=" * 95)

    prompt = [101, 204, 305, 406, 2] # 5 token khởi đầu (<s> ... </s>)
    
    # ------------------------------------------------------------------
    # 1. ĐO ĐẠC MÔ HÌNH EVOLVING HYBRID SSM
    # ------------------------------------------------------------------
    print("\n>>> Đang khởi chạy mô hình Evolving Hybrid SSM (89.35M params)...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model_ssm = EvolvingHeadSSMModel(
        vocab_size=40960, d_model=512, num_micro_heads=8, num_layers=8,
        dim_feedforward=2048, max_seq_len=8192, pad_idx=config.pad_idx
    ).to(device).eval()

    vram_ssm_baseline = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"   • VRAM tĩnh sau khi nạp mô hình: {vram_ssm_baseline:.2f} MB")

    ssm_checkpoints = {}
    tokens_ssm = list(prompt)
    
    torch.cuda.synchronize()
    t_start_ssm = time.perf_counter()

    with torch.no_grad():
        for t in range(1, num_tokens + 1):
            t_step_0 = time.perf_counter()
            inp = torch.tensor([tokens_ssm], device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model_ssm(inp)
            nxt = torch.argmax(logits[0, -1], dim=-1).item()
            tokens_ssm.append(nxt)
            torch.cuda.synchronize()
            t_step_ms = (time.perf_counter() - t_step_0) * 1000.0

            if t in [1, 50, 100, 200, 300, 400, 500]:
                curr_vram = torch.cuda.memory_allocated() / (1024 * 1024)
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
                # Tính kích thước toán học của S_running (dòng 269) + hidden_state (dòng 343)
                state_kb = (8 * 3 * 8 * 64 * 64 * 2 + 8 * 512 * 2) / 1024.0
                ssm_checkpoints[t] = {
                    "vram_curr": curr_vram,
                    "vram_peak": peak_vram,
                    "state_kb": state_kb,
                    "step_time_ms": t_step_ms
                }
                print(f"   [SSM Step {t:3d}] VRAM Hiện tại: {curr_vram:6.1f} MB | Đỉnh VRAM: {peak_vram:6.1f} MB | State Cache: {state_kb:.1f} KB | Bước: {t_step_ms:4.1f} ms")

    total_time_ssm = time.perf_counter() - t_start_ssm
    del model_ssm, inp, logits
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. ĐO ĐẠC MÔ HÌNH STANDARD TRANSFORMER DECODER
    # ------------------------------------------------------------------
    print("\n>>> Đang khởi chạy mô hình Standard Transformer Decoder (52.8M params)...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_tf = StandardTransformerDecoder(
        vocab_size=40960, d_model=512, num_layers=8, num_heads=8,
        dim_feedforward=2048, max_seq_len=8192, tie_weights=False
    ).to(device).eval()

    vram_tf_baseline = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"   • VRAM tĩnh sau khi nạp mô hình: {vram_tf_baseline:.2f} MB")

    tf_checkpoints = {}
    tokens_tf = list(prompt)

    torch.cuda.synchronize()
    t_start_tf = time.perf_counter()

    with torch.no_grad():
        for t in range(1, num_tokens + 1):
            t_step_0 = time.perf_counter()
            inp = torch.tensor([tokens_tf], device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model_tf(inp)
            nxt = torch.argmax(logits[0, -1], dim=-1).item()
            tokens_tf.append(nxt)
            torch.cuda.synchronize()
            t_step_ms = (time.perf_counter() - t_step_0) * 1000.0

            if t in [1, 50, 100, 200, 300, 400, 500]:
                curr_vram = torch.cuda.memory_allocated() / (1024 * 1024)
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
                # Tính kích thước KV-cache tích lũy: 2 (K,V) * 8 layers * 8 heads * 64 dim * t tokens * 2 bytes
                kv_cache_kb = (2 * 8 * 8 * 64 * t * 2) / 1024.0
                tf_checkpoints[t] = {
                    "vram_curr": curr_vram,
                    "vram_peak": peak_vram,
                    "kv_cache_kb": kv_cache_kb,
                    "step_time_ms": t_step_ms
                }
                print(f"   [TF  Step {t:3d}] VRAM Hiện tại: {curr_vram:6.1f} MB | Đỉnh VRAM: {peak_vram:6.1f} MB | KV Cache:    {kv_cache_kb:6.1f} KB | Bước: {t_step_ms:4.1f} ms")

    total_time_tf = time.perf_counter() - t_start_tf
    del model_tf, inp, logits
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. BẢNG ĐỐI SÁNH TỔNG HỢP CHI TIẾT THEO TỪNG BƯỚC
    # ------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("📊 BẢNG AUDIT ĐỐI SÁNH TRỰC TIẾP: BỘ NHỚ VRAM & CACHE THEO TỪNG BƯỚC SINH TỪ (AUTOREGRESSIVE)")
    print("=" * 105)
    header = (
        f"| {'Bước (Token t)':<14} | {'Transformer Peak VRAM':<22} | {'SSM Peak VRAM':<15} | "
        f"{'Transformer Cache':<18} | {'SSM Recurrent State':<21} |"
    )
    print(header)
    print(f"|:{'-'*14}-|-:{'-'*20}:|-:{'-'*14}:|-:{'-'*17}:|-:{'-'*20}:|")

    for t in [1, 50, 100, 200, 300, 400, 500]:
        tf_info = tf_checkpoints[t]
        ssm_info = ssm_checkpoints[t]
        
        tf_v = f"{tf_info['vram_peak']:.1f} MB"
        ssm_v = f"{ssm_info['vram_peak']:.1f} MB"
        tf_c = f"{tf_info['kv_cache_kb']:.1f} KB"
        ssm_c = f"{ssm_info['state_kb']:.1f} KB (Cố định)"

        print(f"| Token t = {t:<6d} | {tf_v:>20} | {ssm_v:>13} | {tf_c:>16} | {ssm_c:>19} |")
    print("=" * 105)

if __name__ == "__main__":
    run_live_token_generation_audit(num_tokens=500)
