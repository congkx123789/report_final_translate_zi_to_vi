import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class TransformerDecoderBlock(nn.Module):
    """
    Standard Transformer Decoder Layer with Causal Scaled Dot-Product Softmax Attention.
    O(N^2) time & memory complexity baseline.
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads
        d_k = self.head_dim

        norm_x = self.norm1(x)

        q = self.q_proj(norm_x).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        k = self.k_proj(norm_x).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        v = self.v_proj(norm_x).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)

        # Native Hardware-Accelerated FlashAttention (F.scaled_dot_product_attention)
        # Fully optimizes SM120 Blackwell Tensor Cores & BF16
        dropout_p = self.dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=causal_mask, dropout_p=dropout_p, is_causal=(causal_mask is None)
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        attn_out = self.out_proj(attn_out)

        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class StandardTransformerDecoder(nn.Module):
    """
    Decoder-Only Standard Transformer (GPT / Marian Style) with identical parameter size,
    vocabulary (40,960 tokens), and embedding dimension as EvolvingHeadSSMModel.
    """
    def __init__(
        self,
        vocab_size: int = 40960,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        max_seq_len: int = 4080,
        dropout: float = 0.1,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.tie_weights = tie_weights
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size, bias=True)

        # Match Hybrid SSM architecture (Untied weights: separate embedding and generator)
        if tie_weights:
            self.generator.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,
        return_hidden: bool = False,
        use_checkpointing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        B, N = tokens.shape
        pos = torch.arange(0, N, device=tokens.device).unsqueeze(0)

        x = self.tok_emb(tokens) * math.sqrt(self.d_model) + self.pos_emb(pos)
        x = self.dropout(x)

        for block in self.blocks:
            if use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, None, use_reentrant=False)
            else:
                x = block(x, causal_mask=None)

        x = self.final_norm(x)
        if return_hidden:
            return x, None
        logits = self.generator(x)
        return logits, None
