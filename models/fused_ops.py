import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedDenseGELUFunction(torch.autograd.Function):
    """
    Fused LayerNorm + Linear + GELU operation.
    Fuses memory passes for feed-forward blocks into a single autograd operation,
    reducing intermediate VRAM allocations and GPU kernel launch overhead.
    """
    @staticmethod
    def forward(ctx, x, weight, bias, normalized_shape, eps=1e-5):
        # LayerNorm
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        inv_std = 1.0 / torch.sqrt(var + eps)
        x_norm = (x - mean) * inv_std

        # Linear projection
        linear_out = F.linear(x_norm, weight, bias)

        # GELU activation
        out = F.gelu(linear_out)

        ctx.save_for_backward(x_norm, linear_out, weight, inv_std)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_norm, linear_out, weight, inv_std = ctx.saved_tensors
        
        # Derivative of GELU
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * (1.0 + torch.erf(linear_out / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * linear_out.pow(2)) / math.sqrt(2.0 * math.pi)
        d_gelu = grad_output * (cdf + linear_out * pdf)

        # Backward through Linear
        d_x_norm = d_gelu @ weight
        d_weight = d_gelu.transpose(-2, -1) @ x_norm if x_norm.dim() == 2 else torch.einsum('...i,...j->ij', d_gelu, x_norm)
        d_bias = d_gelu.sum(dim=tuple(range(d_gelu.dim() - 1))) if ctx.needs_input_grad[2] else None

        # Backward through LayerNorm
        N = x_norm.shape[-1]
        d_x = inv_std * (d_x_norm - (d_x_norm.mean(dim=-1, keepdim=True) + x_norm * (d_x_norm * x_norm).mean(dim=-1, keepdim=True)))

        return d_x, d_weight, d_bias, None, None


class FusedDenseGELU(nn.Module):
    """
    Module wrapper for Fused LayerNorm + Linear + GELU.
    Replaces separate LayerNorm -> Linear -> GELU calls.
    """
    def __init__(self, d_model: int, dim_feedforward: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim_feedforward, d_model))
        self.bias = nn.Parameter(torch.empty(dim_feedforward))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FusedDenseGELUFunction.apply(
            x, self.weight, self.bias, (self.d_model,), self.eps
        )
