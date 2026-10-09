import argparse
import os
import time
import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import config
from vocab_tokenizer import VocabTokenizer
from dataset_loader import get_dataloader
from models.transformer_decoder import StandardTransformerDecoder
from models.chunked_loss import ChunkedLinearCrossEntropy

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def get_transformer_adamw_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Standard Production AdamW Optimizer setup for Transformer.
    Applies weight decay 0.01 to 2D weight matrices; 0.0 to 1D biases and LayerNorms.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(optimizer_grouped_parameters, lr=lr, betas=betas, eps=eps)


class TransformerLRScheduler:
    """
    Linear Warmup + Cosine Decay Scheduler matching SSMLRScheduler logic exactly.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 3000,
        total_steps: int = 100000,
        peak_lr: float = 3e-4,
        min_lr: float = 1e-5
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed_lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
            return decayed_lr / self.peak_lr

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def step(self, last_step: int = None):
        if last_step is not None:
            self.scheduler.last_epoch = last_step
        self.scheduler.step()

    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


def log_print(msg: str, log_file: str = "logs_training/train_transformer_baseline.log"):
    print(msg)
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TransformerLRScheduler,
    global_step: int,
    epoch: int,
    best_loss: float,
    ckpt_path: str,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    checkpoint = {
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Standard Transformer Decoder-Only on 34.3M Dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200000, help="Target training steps (e.g. 200,000)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU micro-step")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps (effective batch 32)")
    parser.add_argument("--save-interval", type=int, default=5000, help="Checkpoint interval")
    parser.add_argument("--save-dir", type=str, default="checkpoints_transformer")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--log-file", type=str, default="logs_training/train_transformer_baseline.log")
    parser.add_argument("--data-path", type=str, default="dataset/tagged_output,dataset/glossary_tagged_output,dataset/master_finetune")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    log_print("=" * 80, args.log_file)
    log_print(f"🚀 BẮT ĐẦU HUẤN LUYỆN: STANDARD TRANSFORMER DECODER-ONLY (O(N^2))", args.log_file)
    log_print(f"Thiết bị: {device} | Target Steps: {args.steps:,} | Batch Size hiệu dụng: {args.batch_size * args.accum_steps}", args.log_file)
    log_print(f"Dataset: {args.data_path}", args.log_file)
    log_print("=" * 80, args.log_file)

    dataloader, tokenizer = get_dataloader(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        streaming=True,
        shuffle=True,
    )

    model = StandardTransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_encoder_layers,
        num_heads=config.nhead,
        dim_feedforward=config.dim_feedforward,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        tie_weights=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_print(f"Tổng tham số Transformer Decoder: {total_params:,} (~{total_params/1e6:.2f}M) [8 Layers, d_model=512, d_ff=2048, Untied]", args.log_file)

    optimizer = get_transformer_adamw_optimizer(
        model=model,
        lr=args.peak_lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    scheduler = TransformerLRScheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
    )

    start_step = 0
    start_epoch = 1
    best_loss = float("inf")
    latest_ckpt = os.path.join(args.save_dir, "transformer_latest.pt")

    if args.resume and os.path.exists(latest_ckpt):
        log_print(f"🔄 Đang phục hồi checkpoint Transformer từ '{latest_ckpt}'...", args.log_file)
        chk = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(chk["model_state_dict"])
        if "optimizer_state_dict" in chk:
            optimizer.load_state_dict(chk["optimizer_state_dict"])
        if "scheduler_state_dict" in chk:
            scheduler.load_state_dict(chk["scheduler_state_dict"])
        start_step = chk.get("global_step", 0)
        start_epoch = chk.get("epoch", 1)
        best_loss = chk.get("best_loss", float("inf"))
        log_print(f"   ✓ Phục hồi thành công! Tiếp tục từ Step {start_step:,}...", args.log_file)

    chunked_criterion = ChunkedLinearCrossEntropy(chunk_size=512, ignore_index=config.pad_idx)

    model.train()
    optimizer.zero_grad()

    global_step = start_step
    accum_loss = 0.0
    accum_tokens = 0
    total_tokens = 0
    total_sentences = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        log_print(f"\n--- Epoch {epoch}/{args.epochs} ---", args.log_file)
        epoch_steps = 0

        for batch_idx, (seq_batch, src_lens) in enumerate(dataloader):
            if global_step >= args.steps:
                break

            seq_batch = seq_batch.to(device, non_blocking=True)
            src_lens = src_lens.to(device, non_blocking=True)
            B, N_tot = seq_batch.shape

            input_seq = seq_batch[:, :-1]
            target_seq = seq_batch[:, 1:]

            # GPU Vectorized loss mask creation (0ms Python CPU overhead)
            seq_positions = torch.arange(N_tot - 1, device=device).unsqueeze(0).expand(B, -1)
            loss_mask = (seq_positions > src_lens.unsqueeze(1)) & (target_seq != config.pad_idx)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                masked_target = torch.where(loss_mask, target_seq, config.pad_idx)
                hidden_states, _ = model(input_seq, return_hidden=True)
                loss = chunked_criterion(hidden_states, model.generator.weight, masked_target)
                scaled_loss = loss / float(args.accum_steps)

            num_valid = float((masked_target != config.pad_idx).sum().item())
            if num_valid > 0:
                scaled_loss.backward()
                accum_loss += loss.item() * num_valid
                accum_tokens += num_valid
                total_tokens += num_valid
                total_sentences += B

            if (batch_idx + 1) % args.accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                curr_step_loss = accum_loss / max(1.0, accum_tokens)
                accum_loss = 0.0
                accum_tokens = 0
                global_step += 1
                epoch_steps += 1

                if curr_step_loss < best_loss and global_step > 50:
                    best_loss = curr_step_loss
                    best_path = os.path.join(args.save_dir, "transformer_best.pt")
                    save_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, best_path)

                if global_step % args.log_interval == 0 or global_step == start_step + 1:
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens / max(elapsed, 0.001)
                    sents_per_sec = total_sentences / max(elapsed, 0.001)
                    ppl = math.exp(min(curr_step_loss, 20.0))
                    current_lr = scheduler.get_last_lr()
                    alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    res_mb = torch.cuda.memory_reserved() / (1024 * 1024)

                    msg = (
                        f"Step [{global_step:7d}/{args.steps}] | Loss: {curr_step_loss:7.4f} | PPL: {ppl:8.2f} | "
                        f"LR: {current_lr:.2e} | Speed: {sents_per_sec:6.1f} sent/s ({tokens_per_sec:7.1f} tok/s) | "
                        f"VRAM: {alloc_mb:.0f}/{res_mb:.0f}MB"
                    )
                    log_print(msg, args.log_file)

                if global_step % args.save_interval == 0:
                    step_path = os.path.join(args.save_dir, f"transformer_step_{global_step}.pt")
                    save_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, latest_ckpt)
                    save_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, step_path)
                    log_print(f"💾 [CHECKPOINT SAVED] Transformer Step {global_step:,} -> '{latest_ckpt}'", args.log_file)

        if global_step >= args.steps:
            break

    save_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, latest_ckpt)
    log_print(f"✅ HOÀN TẤT HUẤN LUYỆN TRANSFORMER 200K STEPS! Checkpoint: {latest_ckpt}", args.log_file)


if __name__ == "__main__":
    main()
