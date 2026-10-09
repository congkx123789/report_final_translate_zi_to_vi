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

def run_comparison(num_steps: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"Running Decoder-Only Comparison on Device: {device}")
    print(f"Steps: {num_steps}, Effective Batch Size: 32 (Micro-batch 8, Accum 4)")
    print(f"============================================================")

    # 1. DataLoader
    dataloader, tokenizer = get_dataloader(
        data_dir=config.dataset_dir,
        batch_size=8,
        streaming=True,
        shuffle=True
    )
    data_iter = iter(dataloader)

    # 2. Models
    model_transformer = StandardTransformerDecoder(
        vocab_size=40960, d_model=512, num_layers=8, num_heads=8, dim_feedforward=2048
    ).to(device)

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
        ("Standard Transformer Decoder-Only (O(N^2))", model_transformer, opt_tf, False),
        ("Evolving-Head Hybrid Attention-SSM (O(N))", model_hybrid, opt_hybrid, True),
    ]

    results = {}

    for model_name, model, opt, is_hybrid in models:
        print(f"\n--- Training: {model_name} ---")
        model.train()
        opt.zero_grad()
        
        step_losses = []
        step_times = []
        accum_loss = 0.0
        accum_tokens = 0
        global_step = 0

        start_t = time.time()
        
        for batch_idx in range(num_steps * 4):
            try:
                seq_batch, src_lens = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                seq_batch, src_lens = next(data_iter)

            seq_batch = seq_batch.to(device)
            B, N_tot = seq_batch.shape

            input_seq = seq_batch[:, :-1]
            target_seq = seq_batch[:, 1:]

            loss_mask = torch.zeros_like(target_seq, dtype=torch.float32, device=device)
            for b in range(B):
                start_vi = src_lens[b].item() + 1
                loss_mask[b, start_vi:] = (target_seq[b, start_vi:] != config.pad_idx).float()

            masked_target = target_seq.clone()
            masked_target[loss_mask == 0] = config.pad_idx

            t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                hidden_states, _ = model(input_seq, return_hidden=True)
                loss = chunked_criterion(hidden_states, model.generator.weight, masked_target)
                scaled_loss = loss / 4.0

            scaled_loss.backward()
            num_valid = float((masked_target != config.pad_idx).sum().item())
            accum_loss += loss.item() * num_valid
            accum_tokens += num_valid

            if (batch_idx + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step_loss = accum_loss / max(1.0, accum_tokens)
                accum_loss = 0.0
                accum_tokens = 0
                step_time = (time.time() - t0) * 4 * 1000  # ms per effective batch
                step_losses.append(step_loss)
                step_times.append(step_time)
                global_step += 1

                if global_step % 10 == 0 or global_step == 1:
                    ppl = math.exp(min(step_loss, 20.0))
                    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    print(f"Step [{global_step:2d}/{num_steps}] | Loss: {step_loss:7.4f} | PPL: {ppl:8.2f} | StepTime: {step_time:6.1f}ms | Peak VRAM: {vram_mb:6.1f}MB")

        total_elapsed = time.time() - start_t
        results[model_name] = {
            "initial_loss": step_losses[0],
            "final_loss": step_losses[-1],
            "loss_history": step_losses,
            "avg_step_time_ms": sum(step_times) / len(step_times),
            "total_time_s": total_elapsed,
        }

    print("\n" + "="*70)
    print(f"{'MÔ HÌNH':<42} | {'LOSS ĐẦU':<10} | {'LOSS BƯỚC 50':<12} | {'TỐC ĐỘ (ms/step)':<15}")
    print("="*70)
    for name, res in results.items():
        print(f"{name:<42} | {res['initial_loss']:<10.4f} | {res['final_loss']:<12.4f} | {res['avg_step_time_ms']:<15.1f}")
    print("="*70)

if __name__ == "__main__":
    run_comparison(num_steps=50)
