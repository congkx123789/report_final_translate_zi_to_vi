import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from config import config
from vocab_tokenizer import VocabTokenizer
from dataset_loader import get_dataloader
from models.transformer_decoder import StandardTransformerDecoder
from models.micro_head_ssm import EvolvingHeadSSMModel
from models.chunked_loss import ChunkedLinearCrossEntropy

def run_long_seq_comparison(num_steps: int = 50, min_tokens: int = 80, max_seq_len: int = 512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"BẮT ĐẦU TEST TRAIN SO SÁNH TRÊN CÁC ĐOẠN VĂN BẢN DÀI (>= {min_tokens} TOKENS)")
    print(f"Thiết bị: {device} | Kích thước batch hiệu dụng: 16 (Micro-batch 4, Accum 4)")
    print(f"Tổng số bước (Steps): {num_steps}")
    print("=" * 75)

    tokenizer = VocabTokenizer()
    
    # DataLoader streaming từ tập dataset thực tế
    base_loader, _ = get_dataloader(
        data_dir=config.dataset_dir,
        batch_size=1,
        streaming=True,
        shuffle=True
    )

    print(f"Đang thu thập các cặp câu dài (>= {min_tokens} tokens) từ tập dữ liệu thực tế...")
    long_samples = []
    start_collect = time.time()
    
    for seq_batch, src_lens in base_loader:
        # seq_batch shape: (1, seq_len)
        seq_len = seq_batch.shape[1]
        zh_len = src_lens[0].item()
        vi_len = seq_len - zh_len - 1
        
        # Chỉ chọn các đoạn dài: câu tiếng Trung >= min_tokens hoặc tổng câu >= 120 tokens
        if zh_len >= min_tokens and seq_len <= max_seq_len:
            long_samples.append((seq_batch[0].tolist(), zh_len))
            if len(long_samples) % 100 == 0:
                print(f"   Đã lọc được {len(long_samples)} mẫu dài... (Mẫu gần nhất: ZH={zh_len} tok, Tổng={seq_len} tok)")
            if len(long_samples) >= num_steps * 4 * 4:  # Đủ batch cho toàn bộ test
                break
                
        if time.time() - start_collect > 60 and len(long_samples) >= num_steps * 4:
            break

    print(f"Thu thập thành công {len(long_samples)} mẫu văn bản dài! (Thời gian lọc: {time.time()-start_collect:.1f}s)")

    # Khởi tạo mô hình Transformer Decoder-Only (O(N^2))
    model_transformer = StandardTransformerDecoder(
        vocab_size=40960, d_model=512, num_layers=8, num_heads=8, dim_feedforward=2048
    ).to(device)

    # Khởi tạo mô hình Evolving-Head Hybrid SSM (O(N))
    model_hybrid = EvolvingHeadSSMModel(
        vocab_size=40960, d_model=512, num_layers=8, num_micro_heads=8, dim_feedforward=2048
    ).to(device)

    opt_tf = AdamW(model_transformer.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=0.01)

    decay_params, no_decay_params = [], []
    for name, param in model_hybrid.named_parameters():
        if param.dim() < 2 or "bias" in name or "norm" in name or "A_log" in name or "decay_logits" in name or "elevation_logits" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    opt_hybrid = AdamW([{"params": decay_params, "weight_decay": 0.01}, {"params": no_decay_params, "weight_decay": 0.0}], lr=3e-4, betas=(0.9, 0.98))

    chunked_criterion = ChunkedLinearCrossEntropy(chunk_size=512, ignore_index=config.pad_idx)

    models = [
        ("Standard Transformer Decoder-Only (O(N^2))", model_transformer, opt_tf),
        ("Evolving-Head Hybrid SSM (O(N))", model_hybrid, opt_hybrid),
    ]

    results = {}
    micro_batch_size = 4
    accum_steps = 4

    for model_name, model, opt in models:
        print(f"\n" + "-"*75)
        print(f"TIẾN HÀNH HUẤN LUYỆN: {model_name}")
        print(f"-"*75)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.train()
        opt.zero_grad()

        sample_idx = 0
        step_losses = []
        step_times = []
        step_vrams = []
        accum_loss = 0.0
        accum_tokens = 0
        global_step = 0

        start_train_t = time.time()

        for step in range(num_steps):
            t_step_start = time.time()
            
            for accum_idx in range(accum_steps):
                # Tạo micro-batch từ long_samples
                batch_samples = []
                batch_zh_lens = []
                for _ in range(micro_batch_size):
                    s, zh_l = long_samples[sample_idx % len(long_samples)]
                    sample_idx += 1
                    batch_samples.append(s)
                    batch_zh_lens.append(zh_l)

                max_l = max(len(s) for s in batch_samples)
                padded_batch = [s + [config.pad_idx] * (max_l - len(s)) for s in batch_samples]
                
                seq_batch = torch.tensor(padded_batch, dtype=torch.long, device=device)
                B, N_tot = seq_batch.shape

                input_seq = seq_batch[:, :-1]
                target_seq = seq_batch[:, 1:]

                loss_mask = torch.zeros_like(target_seq, dtype=torch.float32, device=device)
                for b in range(B):
                    start_vi = batch_zh_lens[b] + 1
                    loss_mask[b, start_vi:] = (target_seq[b, start_vi:] != config.pad_idx).float()

                masked_target = target_seq.clone()
                masked_target[loss_mask == 0] = config.pad_idx

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    hidden_states, _ = model(input_seq, return_hidden=True)
                    loss = chunked_criterion(hidden_states, model.generator.weight, masked_target)
                    scaled_loss = loss / float(accum_steps)

                scaled_loss.backward()
                num_valid = float((masked_target != config.pad_idx).sum().item())
                accum_loss += loss.item() * num_valid
                accum_tokens += num_valid

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            step_time = (time.time() - t_step_start) * 1000  # ms
            step_loss = accum_loss / max(1.0, accum_tokens)
            accum_loss = 0.0
            accum_tokens = 0
            
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

            step_losses.append(step_loss)
            step_times.append(step_time)
            step_vrams.append(peak_vram)
            global_step += 1

            if global_step % 10 == 0 or global_step == 1:
                ppl = math.exp(min(step_loss, 20.0))
                print(f"Step [{global_step:2d}/{num_steps}] | Loss: {step_loss:7.4f} | PPL: {ppl:8.2f} | StepTime: {step_time:6.1f}ms | Peak VRAM: {peak_vram:6.1f}MB")

        results[model_name] = {
            "initial_loss": step_losses[0],
            "loss_step_25": step_losses[24] if len(step_losses) >= 25 else step_losses[-1],
            "final_loss": step_losses[-1],
            "initial_ppl": math.exp(min(step_losses[0], 20.0)),
            "final_ppl": math.exp(min(step_losses[-1], 20.0)),
            "avg_step_time_ms": sum(step_times) / len(step_times),
            "peak_vram_mb": max(step_vrams),
        }

    print("\n" + "="*80)
    print("BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH HUẤN LUYỆN TRÊN ĐOẠN DÀI (LONG SEQUENCES)")
    print("="*80)
    header = f"{'Kiến trúc Mô hình':<35} | {'Loss Đầu':<9} | {'Loss Cuối':<9} | {'PPL Cuối':<9} | {'Thời gian (ms/step)':<19} | {'VRAM Đỉnh':<10}"
    print(header)
    print("-" * 80)
    for name, r in results.items():
        row = f"{name:<35} | {r['initial_loss']:<9.4f} | {r['final_loss']:<9.4f} | {r['final_ppl']:<9.2f} | {r['avg_step_time_ms']:<19.1f} | {r['peak_vram_mb']:<10.1f}"
        print(row)
    print("="*80)

if __name__ == "__main__":
    run_long_seq_comparison(num_steps=50, min_tokens=80, max_seq_len=512)
