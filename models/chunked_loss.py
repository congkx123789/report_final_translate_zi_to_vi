import torch
import torch.nn as nn
import torch.nn.functional as F

class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    """
    Memory-efficient Chunked Linear Projection + Cross Entropy Loss.
    Computes X @ W^T and CrossEntropyLoss in token-chunks (e.g. 512 tokens),
    reducing peak VRAM allocation from gigabytes to a few megabytes.
    """
    @staticmethod
    def forward(ctx, x, weight, target, bias=None, chunk_size=512, ignore_index=3):
        # x: (N_total, D) where N_total = B * N
        # weight: (V, D)
        # target: (N_total,)
        # bias: (V,) or None
        N_total, D = x.shape
        V, _ = weight.shape

        ctx.save_for_backward(x, weight, target, bias)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index

        total_loss = 0.0
        total_valid_tokens = 0

        with torch.no_grad():
            for k in range(0, N_total, chunk_size):
                x_k = x[k : k + chunk_size]
                y_k = target[k : k + chunk_size]

                logits_k = F.linear(x_k, weight, bias)
                valid_mask = (y_k != ignore_index)
                n_valid_k = valid_mask.sum().item()

                if n_valid_k > 0:
                    loss_k = F.cross_entropy(
                        logits_k, y_k, ignore_index=ignore_index, reduction='sum'
                    )
                    total_loss += loss_k.item()
                    total_valid_tokens += n_valid_k

        ctx.total_valid_tokens = max(1, total_valid_tokens)
        final_loss = torch.tensor(total_loss / ctx.total_valid_tokens, device=x.device, dtype=x.dtype)
        return final_loss

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, target, bias = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        ignore_index = ctx.ignore_index
        N_valid = ctx.total_valid_tokens

        N_total, D = x.shape
        V, _ = weight.shape

        dx = torch.zeros_like(x)
        dweight = torch.zeros_like(weight)
        dbias = torch.zeros_like(bias) if bias is not None else None

        scale = grad_output.item() / float(N_valid)

        for k in range(0, N_total, chunk_size):
            x_k = x[k : k + chunk_size]
            y_k = target[k : k + chunk_size]

            # Recompute logits for chunk k in FP32/BF16
            logits_k = F.linear(x_k, weight, bias)
            probs_k = F.softmax(logits_k, dim=-1)

            # Compute dL/dLogits
            valid_mask = (y_k != ignore_index).unsqueeze(-1) # (C_k, 1)
            
            # One-hot subtraction: probs - 1 for target index
            target_clamped = torch.where(y_k == ignore_index, 0, y_k)
            one_hot = F.one_hot(target_clamped, num_classes=V).to(probs_k.dtype)

            dlogits_k = (probs_k - one_hot) * valid_mask * scale # (C_k, V)

            # Gradients for x_k and weight
            dx[k : k + chunk_size] = dlogits_k @ weight
            dweight += dlogits_k.transpose(0, 1) @ x_k

            if bias is not None:
                dbias += dlogits_k.sum(dim=0)

        return dx, dweight, None, dbias, None, None


class ChunkedLinearCrossEntropy(nn.Module):
    """
    Module wrapper for ChunkedLinearCrossEntropyFunction.
    Replaces `nn.Linear(d_model, vocab_size)` + `nn.CrossEntropyLoss()` for low VRAM usage.
    """
    def __init__(self, chunk_size: int = 512, ignore_index: int = 3):
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        # hidden_states: (B, N, D) -> reshape to (B*N, D)
        # weight: (V, D)
        # target: (B, N) -> reshape to (B*N,)
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            B, N, D = orig_shape
            x_flat = hidden_states.reshape(B * N, D)
            target_flat = target.reshape(B * N)
        else:
            x_flat = hidden_states
            target_flat = target

        return ChunkedLinearCrossEntropyFunction.apply(
            x_flat, weight, target_flat, bias, self.chunk_size, self.ignore_index
        )
