import time
import torch
from config import config
from models.transformer_decoder import StandardTransformerDecoder
from models.micro_head_ssm import EvolvingHeadSSMModel

def benchmark_inference_vram():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("🔍 ĐỐI SÁNH CHI TIẾT BỘ NHỚ VRAM TRONG GIAI ĐOẠN SUY DIỄN (INFERENCE / GENERATION)")
    print(f"Thiết bị: {torch.cuda.get_device_name(0)} (Blackwell SM120, BF16)")
    print("=" * 95)

    # -------------------------------------------------------------
    # PHẦN 1: BỘ NHỚ TĨNH CỦA TRỌNG SỐ MÔ HÌNH (STATIC WEIGHTS ONLY)
    # -------------------------------------------------------------
    print("\n--- PHẦN 1: BỘ NHỚ TĨNH CHỈ LƯU TRỌNG SỐ (EVAL MODE, KHÔNG CÓ OPTIMIZER/GRADIENTS) ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    tf = StandardTransformerDecoder(vocab_size=40960, d_model=512, num_layers=8, max_seq_len=8192, tie_weights=False).to(device).eval()
    tf_params = sum(p.numel() for p in tf.parameters())
    tf_weights_vram = torch.cuda.memory_allocated() / (1024 * 1024)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ssm = EvolvingHeadSSMModel(vocab_size=40960, d_model=512, num_micro_heads=8, num_layers=8, max_seq_len=8192, pad_idx=3).to(device).eval()
    ssm_params = sum(p.numel() for p in ssm.parameters())
    ssm_weights_vram = (torch.cuda.memory_allocated() / (1024 * 1024)) - tf_weights_vram

    print(f"• Standard Transformer ({tf_params/1e6:.2f}M params): Trọng số thuần = {tf_weights_vram:.1f} MB")
    print(f"• Evolving Hybrid SSM   ({ssm_params/1e6:.2f}M params): Trọng số thuần = {torch.cuda.memory_allocated() / (1024 * 1024) - tf_weights_vram:.1f} MB")

    del tf, ssm
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # PHẦN 2: BỘ NHỚ TRẠNG THÁI SINH TỪNG TỪ (RECURRENT STATE VS KV CACHE)
    # -------------------------------------------------------------
    print("\n--- PHẦN 2: LÝ THUYẾT VÀ KÍCH THƯỚC BỘ NHỚ TRẠNG THÁI CACHE THEO SỐ TỪ SINH RA ---")
    print(f"| {'Số token đã sinh (T)':<22} | {'Transformer KV Cache':<22} | {'Hybrid SSM Recurrent State':<28} | {'Chênh lệch Cache':<18} |")
    print(f"|:{'-'*22}-|-:{'-'*20}:|-:{'-'*26}:|-:{'-'*18}:|")

    # Tính toán kích thước toán học chính xác:
    # Transformer KV Cache = 2 (K & V) * num_layers (8) * num_heads (8) * head_dim (64) * T * 2 bytes (BF16)
    # SSM State = num_layers (8) * num_clusters (3) * num_heads (8) * head_dim (64) * head_dim (64) * 2 bytes (BF16)
    ssm_state_bytes = 8 * 3 * 8 * 64 * 64 * 2 # Cố định hoàn toàn
    ssm_state_kb = ssm_state_bytes / 1024.0

    token_steps = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    for T in token_steps:
        tf_kv_bytes = 2 * 8 * 8 * 64 * T * 2
        tf_kv_kb = tf_kv_bytes / 1024.0
        tf_str = f"{tf_kv_kb/1024.0:.2f} MB" if tf_kv_kb >= 1024 else f"{tf_kv_kb:.0f} KB"
        ssm_str = f"{ssm_state_kb:.0f} KB (Cố định)"
        diff_str = f"SSM nhỏ hơn {tf_kv_bytes / ssm_state_bytes:.1f}x"
        print(f"| {T:<22,d} | {tf_str:>20} | {ssm_str:>26} | {diff_str:>18} |")

    # -------------------------------------------------------------
    # PHẦN 3: ĐO THỰC NGHIỆM TRÊN GPU QUA CÁC ĐỘ DÀI PROMPT VĂN BẢN (PREFILL)
    # -------------------------------------------------------------
    print("\n--- PHẦN 3: ĐO THỰC NGHIỆM VRAM ĐỈNH KHI NẠP ĐOẠN VĂN BẢN VÀO MÔ HÌNH (NO_GRAD) ---")
    
    tf = StandardTransformerDecoder(vocab_size=40960, d_model=512, num_layers=8, max_seq_len=8192, tie_weights=False).to(device).eval()
    ssm = EvolvingHeadSSMModel(vocab_size=40960, d_model=512, num_micro_heads=8, num_layers=8, max_seq_len=8192, pad_idx=3).to(device).eval()

    prompt_lengths = [256, 512, 1024, 2048, 4096, 8192]
    
    print(f"| {'Độ dài đoạn văn N':<20} | {'Transformer Infer VRAM':<24} | {'Hybrid SSM Infer VRAM':<24} | {'So sánh VRAM':<22} |")
    print(f"|:{'-'*20}-|-:{'-'*22}:|-:{'-'*22}:|-:{'-'*22}:|")

    with torch.no_grad():
        for N in prompt_lengths:
            x = torch.randint(4, 40960, (1, N), device=device)
            
            # Transformer
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = tf(x)
            v_tf = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # SSM
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = ssm(x)
            v_ssm = torch.cuda.max_memory_allocated() / (1024 * 1024)

            print(f"| {N:<20,d} | {v_tf:>21.1f} MB | {v_ssm:>21.1f} MB | {'Tương đương' if abs(v_tf-v_ssm)<150 else ('SSM nhỏ hơn' if v_ssm<v_tf else 'TF nhỏ hơn'):>22} |")

    del tf, ssm
    torch.cuda.empty_cache()
    print("\n" + "=" * 95)

if __name__ == "__main__":
    benchmark_inference_vram()
