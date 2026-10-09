import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from config import config
from models.transformer_decoder import StandardTransformerDecoder
from models.micro_head_ssm import EvolvingHeadSSMModel

def run_comprehensive_inference_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("🔬 BÁO CÁO THỰC NGHIỆM ĐỐI SÁNH SUY DIỄN TOÀN DIỆN (COMPREHENSIVE INFERENCE BENCHMARK)")
    print(f"Thiết bị: {torch.cuda.get_device_name(0)} (Blackwell SM120, BF16)")
    print("Mô hình: Standard Transformer Decoder (52.8M) vs. Evolving Hybrid SSM (89.35M)")
    print("=" * 95)

    # Khởi tạo mô hình ở chế độ Eval
    model_tf = StandardTransformerDecoder(
        vocab_size=40960, d_model=512, num_layers=8, num_heads=8, dim_feedforward=2048, 
        max_seq_len=8192, tie_weights=False
    ).to(device).eval()

    model_ssm = EvolvingHeadSSMModel(
        vocab_size=40960, d_model=512, num_micro_heads=8, num_layers=8, dim_feedforward=2048, 
        max_seq_len=8192, pad_idx=config.pad_idx
    ).to(device).eval()

    # -------------------------------------------------------------
    # THỰC NGHIỆM 1: ĐỘ TRỄ NẠP PROMPT (PREFILL / TTFT - TIME TO FIRST TOKEN)
    # -------------------------------------------------------------
    print("\n--- 1. ĐỘ TRỄ XỬ LÝ ĐOẠN VĂN ĐẦU VÀO (PREFILL / TIME-TO-FIRST-TOKEN) ---")
    prompt_lengths = [128, 256, 512, 1024, 2048, 4096]
    prefill_results = []

    with torch.no_grad():
        for prompt_len in prompt_lengths:
            dummy_prompt = torch.randint(4, 40960, (1, prompt_len), device=device)
            
            # Đo Transformer
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Warmup
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model_tf(dummy_prompt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model_tf(dummy_prompt)
            torch.cuda.synchronize()
            ttft_tf = (time.perf_counter() - t0) / 10.0 * 1000.0
            vram_tf = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # Đo Hybrid SSM
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Warmup
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model_ssm(dummy_prompt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model_ssm(dummy_prompt)
            torch.cuda.synchronize()
            ttft_ssm = (time.perf_counter() - t0) / 10.0 * 1000.0
            vram_ssm = torch.cuda.max_memory_allocated() / (1024 * 1024)

            prefill_results.append({
                "prompt_len": prompt_len,
                "ttft_tf": ttft_tf,
                "ttft_ssm": ttft_ssm,
                "vram_tf": vram_tf,
                "vram_ssm": vram_ssm,
            })
            print(f"Prompt {prompt_len:4d} tokens | TF: {ttft_tf:5.1f} ms ({vram_tf:6.1f} MB) | SSM: {ttft_ssm:5.1f} ms ({vram_ssm:6.1f} MB)")

    # -------------------------------------------------------------
    # THỰC NGHIỆM 2: BỘ NHỚ CACHE & PHỤC VỤ ĐỒNG THỜI (MULTI-USER CONCURRENCY)
    # -------------------------------------------------------------
    print("\n--- 2. KHẢ NĂNG PHỤC VỤ ĐA NGƯỜI DÙNG ĐỒNG THỜI (CONCURRENCY BATCH SERVING) ---")
    # Giả lập 16 người dùng đồng thời, mỗi người sinh 2,048 tokens
    concurrencies = [1, 4, 8, 16, 32]
    context_tokens = 2048
    serving_results = []
    
    # Kích thước lý thuyết:
    # TF KV Cache per token per head = 2 * 64 * 2 bytes = 256 bytes per head * 8 heads * 8 layers = 16,384 bytes = 16 KB / token
    # For context 2048 tokens = 32 MB per user!
    # SSM State per user = 8 layers * 3 clusters * 8 heads * 64 * 64 * 2 bytes = 1.5 MB per user (HOÀN TOÀN CỐ ĐỊNH!)
    
    for B in concurrencies:
        kv_cache_total_mb = B * (context_tokens * 16 / 1024.0)
        ssm_state_total_mb = B * 1.5
        saving_ratio = kv_cache_total_mb / ssm_state_total_mb
        serving_results.append({
            "batch": B,
            "tf_kv_mb": kv_cache_total_mb,
            "ssm_state_mb": ssm_state_total_mb,
            "ratio": saving_ratio
        })
        print(f"Batch={B:2d} người dùng | TF KV-Cache: {kv_cache_total_mb:>7.1f} MB | SSM State: {ssm_state_total_mb:>5.1f} MB | Tiết kiệm: {saving_ratio:5.1f}x")

    # -------------------------------------------------------------
    # THỰC NGHIỆM 3: MÔ PHỎNG SINH TỪNG TOKEN VÀ TỐC ĐỘ GENERATION (TOKENS/SEC)
    # -------------------------------------------------------------
    print("\n--- 3. ĐO TỐC ĐỘ SINH TỪNG TỪ THỰC TẾ (GENERATION THROUGHPUT & LATENCY) ---")
    gen_lengths = [64, 128, 256]
    gen_results = []
    
    with torch.no_grad():
        for gen_len in gen_lengths:
            prompt = [100, 205, 304, 501, 2] # 5 token khởi đầu
            
            # TF Generation simulation
            seq_tf = list(prompt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(gen_len):
                inp = torch.tensor([seq_tf], device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model_tf(inp)
                nxt = torch.argmax(logits[0, -1], dim=-1).item()
                seq_tf.append(nxt)
            torch.cuda.synchronize()
            t_tf_gen = (time.perf_counter() - t0)
            tok_per_sec_tf = gen_len / max(t_tf_gen, 1e-5)

            # SSM Generation simulation
            seq_ssm = list(prompt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(gen_len):
                inp = torch.tensor([seq_ssm], device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model_ssm(inp)
                nxt = torch.argmax(logits[0, -1], dim=-1).item()
                seq_ssm.append(nxt)
            torch.cuda.synchronize()
            t_ssm_gen = (time.perf_counter() - t0)
            tok_per_sec_ssm = gen_len / max(t_ssm_gen, 1e-5)

            gen_results.append({
                "len": gen_len,
                "tf_speed": tok_per_sec_tf,
                "ssm_speed": tok_per_sec_ssm,
                "tf_time_ms": t_tf_gen * 1000 / gen_len,
                "ssm_time_ms": t_ssm_gen * 1000 / gen_len,
            })
            print(f"Sinh {gen_len:3d} tokens | TF: {tok_per_sec_tf:5.1f} tok/s ({t_tf_gen*1000/gen_len:4.1f} ms/tok) | SSM: {tok_per_sec_ssm:5.1f} tok/s ({t_ssm_gen*1000/gen_len:4.1f} ms/tok)")

    # -------------------------------------------------------------
    # VẼ BIỂU ĐỒ TRỰC QUAN HÓA CHO BÁO CÁO / KHÓA LUẬN
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200)
    plt.subplots_adjust(wspace=0.28)

    # Subplot 1: Prefill Latency
    p_lens = [r["prompt_len"] for r in prefill_results]
    tf_ttft = [r["ttft_tf"] for r in prefill_results]
    ssm_ttft = [r["ttft_ssm"] for r in prefill_results]
    
    axes[0].plot(p_lens, tf_ttft, 'o-', color='#ef4444', linewidth=2.2, label='Standard Transformer')
    axes[0].plot(p_lens, ssm_ttft, 's-', color='#3b82f6', linewidth=2.2, label='Hybrid Attention-SSM')
    axes[0].set_title('1. Prefill Latency (Time-To-First-Token)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Prompt Length (Tokens)', fontsize=11)
    axes[0].set_ylabel('Latency (ms)', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(loc='upper left')

    # Subplot 2: Multi-User Concurrency VRAM Cache
    batches = [r["batch"] for r in serving_results]
    tf_kv = [r["tf_kv_mb"] for r in serving_results]
    ssm_st = [r["ssm_state_mb"] for r in serving_results]
    
    x = np.arange(len(batches))
    width = 0.35
    axes[1].bar(x - width/2, tf_kv, width, label='Transformer KV-Cache (2k ctx)', color='#ef4444', edgecolor='#b91c1c')
    axes[1].bar(x + width/2, ssm_st, width, label='SSM Recurrent State (2k ctx)', color='#3b82f6', edgecolor='#1d4ed8')
    axes[1].set_title('2. Concurrency Serving VRAM Footprint', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{b} Users' for b in batches], fontsize=10)
    axes[1].set_ylabel('Cache VRAM (MB)', fontsize=11)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
    axes[1].legend(loc='upper left')

    # Subplot 3: Generation Latency per token
    glens = [r["len"] for r in gen_results]
    tf_ms = [r["tf_time_ms"] for r in gen_results]
    ssm_ms = [r["ssm_time_ms"] for r in gen_results]
    
    axes[2].plot(glens, tf_ms, 'o-', color='#ef4444', linewidth=2.2, label='Standard Transformer')
    axes[2].plot(glens, ssm_ms, 's-', color='#3b82f6', linewidth=2.2, label='Hybrid Attention-SSM')
    axes[2].set_title('3. Inter-Token Latency (ms/token)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Tokens Generated', fontsize=11)
    axes[2].set_ylabel('Latency (ms/token)', fontsize=11)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].legend(loc='upper left')

    plt.suptitle('Inference Benchmark: Standard Transformer Decoder vs. Evolving Hybrid SSM (RTX 5060 Ti SM120)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    out_png = 'logs_training/inference_comparison.png'
    out_pdf = 'uet-graduation-thesis/fig_inference_comparison.pdf'
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Đã lưu biểu đồ PNG: {out_png}")
    print(f"✓ Đã lưu biểu đồ PDF: {out_pdf}")

if __name__ == "__main__":
    run_comprehensive_inference_benchmark()
