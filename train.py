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
from models.micro_head_ssm import EvolvingHeadSSMModel, load_checkpoint_compatible
from models.chunked_loss import ChunkedLinearCrossEntropy

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def get_standard_adamw_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Standard Production AdamW Optimizer setup for Transformers & SSMs.
    Separates 2D weight matrices (weight decay applied) from 1D biases, LayerNorms,
    and SSM continuous parameters A_log / decay_logits (no weight decay).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "norm" in name or "A_log" in name or "decay_logits" in name or "elevation_logits" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=betas, eps=eps)
    return optimizer


class SSMLRScheduler:
    """
    Warmup + Cosine Decay LR Scheduler.
    Decays smoothly across total_steps (default: 100,000 steps to reach min_lr).
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


def compute_masked_loss(logits: torch.Tensor, target_seq: torch.Tensor, loss_mask: torch.Tensor, criterion: nn.Module) -> Tuple[torch.Tensor, float]:
    """
    Computes masked CrossEntropyLoss efficiently without giant memory spikes.
    """
    valid_indices = (loss_mask > 0).nonzero(as_tuple=False)
    if valid_indices.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True), 0.0

    b_idx, n_idx = valid_indices[:, 0], valid_indices[:, 1]
    selected_logits = logits[b_idx, n_idx]
    selected_targets = target_seq[b_idx, n_idx]

    loss = criterion(selected_logits, selected_targets)
    return loss, float(selected_targets.numel())


def log_print(msg: str, log_file: str = "train.log"):
    """Prints message to stdout and appends to train.log file."""
    print(msg)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: SSMLRScheduler,
    global_step: int,
    epoch: int,
    best_loss: float,
    ckpt_path: str,
):
    """
    Detailed Checkpoint Saver: Stores complete model weights, optimizer, scheduler,
    step count, epoch index, and best loss metadata.
    """
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    checkpoint = {
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": {
            "d_model": config.d_model,
            "num_layers": config.num_encoder_layers,
            "num_micro_heads": 16,
        }
    }
    torch.save(checkpoint, ckpt_path)


def train_pipeline(
    model: EvolvingHeadSSMModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: SSMLRScheduler,
    criterion: nn.Module,
    device: torch.device,
    tokenizer: Optional[VocabTokenizer] = None,
    start_step: int = 0,
    start_epoch: int = 1,
    best_loss: float = float("inf"),
    epochs: int = 5,
    max_steps: int = None,
    log_interval: int = 20,
    save_interval: int = 5000,
    grad_accum_steps: int = 4,
    use_bf16: bool = config.use_bf16,
    clip_grad: float = config.clip_grad,
    log_file: str = "train.log",
    save_dir: str = "checkpoints",
    use_chunked_loss: bool = False,
    checkpointing: bool = False,
):
    model.train()
    total_loss = 0.0
    global_step = start_step
    total_tokens = 0
    total_sentences = 0
    start_time = time.time()

    banner = (
        "\n==========================================================================\n"
        "🚀 EVOLVING MICRO-HEAD SSM PRODUCTION TRAINING ACTIVE\n"
        f"   Epochs: {epochs} | Peak LR: {scheduler.peak_lr:.1e} -> Min LR: {scheduler.min_lr:.1e} at Step {scheduler.total_steps:,}\n"
        f"   AdamW Betas: (0.9, 0.98) | Weight Decay: {config.weight_decay} | Grad Accum: {grad_accum_steps}\n"
        f"   Save Interval: Every {save_interval:,} steps | Save Dir: {save_dir}\n"
        f"   Logging File: {log_file}\n"
        "=========================================================================="
    )
    log_print(banner, log_file)

    optimizer.zero_grad()
    accum_loss = 0.0
    accum_tokens = 0

    for epoch in range(start_epoch, epochs + 1):
        log_print(f"\n--- Epoch {epoch}/{epochs} ---", log_file)
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (seq_batch, src_lens) in enumerate(dataloader):
            if max_steps is not None and global_step >= max_steps:
                break

            seq_batch = seq_batch.to(device)
            B, N_tot = seq_batch.shape

            input_seq = seq_batch[:, :-1]
            target_seq = seq_batch[:, 1:]

            loss_mask = torch.zeros_like(target_seq, dtype=torch.float32, device=device)
            for b in range(B):
                # Vietnamese target tokens start at index src_lens[b] + 1 (after EOS and BOS)
                start_vi = src_lens[b].item() + 1
                loss_mask[b, start_vi:] = (target_seq[b, start_vi:] != config.pad_idx).float()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                if use_chunked_loss:
                    masked_target = target_seq.clone()
                    masked_target[loss_mask == 0] = config.pad_idx
                    hidden_states, _ = model(input_seq, return_hidden=True, use_checkpointing=checkpointing)
                    chunked_criterion = ChunkedLinearCrossEntropy(chunk_size=512, ignore_index=config.pad_idx)
                    loss = chunked_criterion(hidden_states, model.generator.weight, masked_target, model.generator.bias)
                    scaled_loss = loss / grad_accum_steps
                    num_valid_tokens = float((masked_target != config.pad_idx).sum().item())
                    sum_loss = loss * max(1.0, num_valid_tokens)
                else:
                    logits, _ = model(input_seq, use_checkpointing=checkpointing)
                    sum_loss, num_valid_tokens = compute_masked_loss(logits, target_seq, loss_mask, criterion)
                    scaled_loss = (sum_loss / max(1.0, float(num_valid_tokens))) / grad_accum_steps

            if num_valid_tokens > 0:
                scaled_loss.backward()
                accum_loss += sum_loss.item()
                accum_tokens += num_valid_tokens
                total_tokens += num_valid_tokens
                total_sentences += B

            if (batch_idx + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                curr_step_loss = accum_loss / max(1, accum_tokens)
                accum_loss = 0.0
                accum_tokens = 0
                total_loss += curr_step_loss
                epoch_loss += curr_step_loss
                global_step += 1
                epoch_steps += 1

                if curr_step_loss < best_loss and global_step > 50:
                    best_loss = curr_step_loss
                    best_path = os.path.join(save_dir, "ssm_model_best.pt")
                    save_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, best_path)

                if global_step % log_interval == 0 or global_step == start_step + 1:
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens / max(elapsed, 0.001)
                    sents_per_sec = total_sentences / max(elapsed, 0.001)
                    ppl = math.exp(min(curr_step_loss, 20.0))
                    current_lr = scheduler.get_last_lr()

                    vram_str = ""
                    if torch.cuda.is_available():
                        alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                        res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                        vram_str = f" | VRAM: {alloc_mb:.0f}/{res_mb:.0f}MB"

                    msg = (
                        f"Step [{global_step:7d}] | Loss: {curr_step_loss:.4f} | PPL: {ppl:8.2f} | "
                        f"LR: {current_lr:.2e} | Speed: {sents_per_sec:6.1f} sent/s ({tokens_per_sec:7.1f} tok/s) | "
                        f"StepTime: {elapsed/max(epoch_steps, 1)*1000:5.1f}ms{vram_str}"
                    )
                    log_print(msg, log_file)

                if global_step % save_interval == 0:
                    latest_path = os.path.join(save_dir, "ssm_model_latest.pt")
                    step_path = os.path.join(save_dir, f"ssm_model_step_{global_step}.pt")
                    save_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, latest_path)
                    save_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, step_path)
                    log_print(f"\n💾 [CHECKPOINT SAVED] Step {global_step:,} -> '{latest_path}' & '{step_path}'", log_file)
                    
                    # Live sample translation check inside train log
                    if tokenizer is not None:
                        model.eval()
                        with torch.no_grad():
                            sample_zh = ["你好", "我们需要继续努力"]
                            for zh in sample_zh:
                                src_ids = tokenizer.encode_zh(zh, add_bos_eos=False)
                                if src_ids:
                                    prompt = src_ids + [config.eos_idx, config.bos_idx]
                                    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
                                    gen_ids = []
                                    for _ in range(32):
                                        logits, _ = model(prompt_tensor)
                                        nxt = torch.argmax(logits[0, -1], dim=-1).item()
                                        if nxt == config.eos_idx or nxt == config.pad_idx:
                                            break
                                        gen_ids.append(nxt)
                                        prompt_tensor = torch.cat([prompt_tensor, torch.tensor([[nxt]], device=device)], dim=1)
                                    vi_res = tokenizer.decode_vi(gen_ids)
                                    log_print(f"   🔮 Sample Trans: 🇨🇳 '{zh}' -> 🇻🇳 '{vi_res}'", log_file)
                        model.train()
                        log_print("", log_file)

                accum_loss = 0.0
                accum_tokens = 0

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        log_print(f"\n--- Epoch [{epoch}/{epochs}] Completed --- Average Loss: {avg_epoch_loss:.4f}", log_file)
        epoch_ckpt_path = os.path.join(save_dir, f"ssm_model_epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, epoch_ckpt_path)
        log_print(f"💾 [EPOCH CHECKPOINT SAVED] Epoch {epoch} -> '{epoch_ckpt_path}'\n", log_file)

    latest_path = os.path.join(save_dir, "ssm_model_latest.pt")
    save_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, latest_path)
    log_print(f"\n✅ All {epochs} Epochs Completed! Total Steps: {global_step:,}. Final Checkpoint saved to: {latest_path}", log_file)


def main():
    parser = argparse.ArgumentParser(description="Train Evolving Micro-Head SSM Model on Full Dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs over dataset (default: 5)")
    parser.add_argument("--steps", type=int, default=None, help="Max steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU micro-step (default: 16)")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--save-interval", type=int, default=5000, help="Checkpoint save interval (default: 5000)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--log-interval", type=int, default=20, help="Log output interval steps (default: 20)")
    parser.add_argument("--peak-lr", type=float, default=3e-4, help="Peak Learning Rate after Warmup")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum Learning Rate at end of Cosine Decay")
    parser.add_argument("--warmup-steps", type=int, default=3000, help="Warmup steps count")
    parser.add_argument("--total-steps", type=int, default=100000, help="Total steps for Cosine Decay to Min LR")
    parser.add_argument("--no-bf16", action="store_true", help="Disable BF16 mixed precision")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset directly from gz files")
    parser.add_argument("--max-samples", type=int, default=100000, help="Max samples to preload into RAM memory for ultra-fast training")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers (default: 4)")
    parser.add_argument("--log-file", type=str, default="train.log", help="Log output file path (default: train.log)")
    parser.add_argument("--data-path", type=str, default=None, help="Custom dataset path or jsonl.gz file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint file")
    parser.add_argument("--finetune", action="store_true", help="Fine-tuning mode")
    parser.add_argument("--resume", action="store_true", help="Resume mode")
    parser.add_argument("--binary-data", action="store_true", help="Use 0ms memory-mapped pre-tokenized binary dataset (.bin)")
    parser.add_argument("--use-chunked-loss", action="store_true", help="Enable ChunkedLinearCrossEntropy low-VRAM loss")
    parser.add_argument("--checkpointing", action="store_true", help="Enable Activation Checkpointing for 70%% VRAM savings")
    args = parser.parse_args()

    device = torch.device(config.device)
    use_bf16 = config.use_bf16 and not args.no_bf16

    if device.type == "cuda":
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    log_print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", args.log_file)
    log_print(f"Hardware Specs: BF16={use_bf16} | SM120 Blackwell TensorCores=Enabled | CuDNN Benchmark=Enabled", args.log_file)

    save_dir = args.save_dir or config.checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)
    target_data_dir = args.data_path or config.dataset_dir

    if args.binary_data:
        from dataset_binary_loader import get_binary_dataloader
        dataloader = get_binary_dataloader(
            data_dir=target_data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        tokenizer = VocabTokenizer()
        log_print(f"⚡ [BINARY MEMMAP DATASET] 0ms Memory-Mapped Binary DataLoader Initialized ({len(dataloader.dataset):,} samples, {len(dataloader):,} batches)", args.log_file)
    elif args.streaming:
        dataloader, tokenizer = get_dataloader(
            data_dir=target_data_dir,
            batch_size=args.batch_size,
            streaming=True,
            shuffle=True,
        )
        log_print(f"Streaming DataLoader Initialized across all dataset files in {target_data_dir}", args.log_file)
    else:
        log_print(f"Preloading up to {args.max_samples:,} sentence pairs from {target_data_dir} into RAM memory...", args.log_file)
        dataloader, tokenizer = get_dataloader(
            data_dir=target_data_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            streaming=False,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        log_print(f"✓ ParallelTextDataset Initialized with {len(dataloader.dataset):,} samples ({len(dataloader):,} batches, {args.num_workers} workers, pin_memory=True).", args.log_file)

    model = EvolvingHeadSSMModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        num_micro_heads=config.nhead,
        num_layers=config.num_encoder_layers,
        window_sizes=(5, 20, None),
        dim_feedforward=config.dim_feedforward,
        deep_ssm_expand_factor=2.0,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        pad_idx=config.pad_idx,
    ).to(device)

    optimizer = get_standard_adamw_optimizer(
        model=model,
        lr=args.peak_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    scheduler = SSMLRScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        peak_lr=args.peak_lr,
        min_lr=args.min_lr
    )

    start_step = 0
    start_epoch = 1
    best_loss = float("inf")
    ckpt_path = args.checkpoint or (os.path.join(save_dir, "ssm_model_latest.pt") if (args.resume and os.path.exists(os.path.join(save_dir, "ssm_model_latest.pt"))) else None)

    if ckpt_path and os.path.exists(ckpt_path):
        log_print(f"📦 Loading detailed checkpoint from: '{ckpt_path}'", args.log_file)
        chk = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = chk.get("model_state_dict", chk)

        load_checkpoint_compatible(model, state_dict)

        chk_step = chk.get("global_step", 0)
        best_loss = chk.get("best_loss", float("inf"))

        if args.finetune:
            start_step = 0
            start_epoch = 1
            log_print(f"🎯 [FINE-TUNING MODE] Loaded model weights from step {chk_step:,}. Resetting step to 0.", args.log_file)
        else:
            start_step = chk_step
            start_epoch = chk.get("epoch", 1)
            if "optimizer_state_dict" in chk and args.resume:
                try:
                    optimizer.load_state_dict(chk["optimizer_state_dict"])
                    log_print("   ✓ Successfully restored Optimizer state.", args.log_file)
                except Exception as e:
                    log_print(f"   Warning: Could not load optimizer state: {e}", args.log_file)
            if "scheduler_state_dict" in chk and args.resume:
                try:
                    scheduler.load_state_dict(chk["scheduler_state_dict"])
                    log_print("   ✓ Successfully restored LR Scheduler state.", args.log_file)
                except Exception as e:
                    log_print(f"   Warning: Could not load scheduler state: {e}", args.log_file)

            log_print(f"🔄 [RESUME MODE] Restored checkpoint! Resuming training from Epoch {start_epoch}, Global Step {start_step:,}...", args.log_file)

    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx, reduction="sum")

    train_pipeline(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        tokenizer=tokenizer,
        start_step=start_step,
        start_epoch=start_epoch,
        best_loss=best_loss,
        epochs=args.epochs,
        max_steps=args.steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        grad_accum_steps=args.accum_steps,
        use_bf16=use_bf16,
        clip_grad=config.clip_grad,
        log_file=args.log_file,
        save_dir=save_dir,
        use_chunked_loss=args.use_chunked_loss,
        checkpointing=args.checkpointing,
    )

if __name__ == "__main__":
    main()
