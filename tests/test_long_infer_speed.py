import time
import torch
from models.transformer_decoder import StandardTransformerDecoder
from models.micro_head_ssm import EvolvingHeadSSMModel

def benchmark_long_inference(seq_lengths=[128, 256, 512, 1024, 2048]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*75)
    print("BENCHMARK SUY LUẬN TỰ HỒI QUY (AUTOREGRESSIVE INFERENCE) TRÊN ĐOẠN DÀI")
    print("="*75)
    
    tf = StandardTransformerDecoder(vocab_size=40960, d_model=512, num_layers=8).to(device).eval()
    hybrid = EvolvingHeadSSMModel(vocab_size=40960, d_model=512, num_layers=8, num_micro_heads=8).to(device).eval()

    print(f"{'Độ dài chuỗi (Tokens)':<25} | {'Transformer (ms/tok)':<22} | {'Hybrid SSM (ms/tok)':<22} | {'Tăng tốc':<10}")
    print("-" * 85)

    with torch.no_grad():
        for seq_len in seq_lengths:
            dummy_input = torch.randint(10, 30000, (1, seq_len), device=device)

            # Transformer timing
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                _ = tf(dummy_input)
            torch.cuda.synchronize()
            tf_time = (time.time() - t0) / 10 * 1000

            # Hybrid SSM timing
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                _ = hybrid(dummy_input)
            torch.cuda.synchronize()
            hybrid_time = (time.time() - t0) / 10 * 1000

            speedup = tf_time / max(hybrid_time, 1e-6)
            print(f"{seq_len:<25} | {tf_time:<22.2f} | {hybrid_time:<22.2f} | {speedup:<10.2f}x")

if __name__ == "__main__":
    benchmark_long_inference()
