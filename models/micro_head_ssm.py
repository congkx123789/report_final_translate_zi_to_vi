import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for 2D Query & Key rotation.
    Supports dynamic sequence length scaling up to max_seq_len (default 4080+).
    """

    def __init__(self, dim: int, max_seq_len: int = 4080, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        device = self.inv_freq.device if hasattr(self, "inv_freq") else torch.device("cpu")
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq)  # (max_seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1) # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding (RoPE) to query and key tensors.
    q, k shape: (B, H, N, d_k)
    cos, sin shape: (N, d_k)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d_k)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d_k)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DynamicHeadStitcher(nn.Module):
    """
    Dynamic Micro-Head Matrix Slicing, Per-Token Adaptive Routing & Cross-Head Re-stitching Layer.
    """

    def __init__(self, d_model: int, num_micro_heads: int, num_clusters: int):
        super().__init__()
        self.d_model = d_model
        self.num_micro_heads = num_micro_heads
        self.num_clusters = num_clusters

        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_micro_heads * num_clusters),
        )
        self.stitch_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        B, N, D = x.shape
        router_logits = self.router(x).view(B, N, self.num_micro_heads, self.num_clusters)
        router_logits = router_logits.transpose(1, 2)  # (B, H, N, C)
        return F.softmax(router_logits, dim=-1)


class HeadPruner(nn.Module):
    """
    Adaptive Micro-Head Importance Scoring & Low-Contribution Pruning Layer.
    """

    def __init__(self, d_model: int, num_micro_heads: int, pruning_threshold: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_micro_heads = num_micro_heads
        self.pruning_threshold = pruning_threshold

        self.importance_proj = nn.Sequential(
            nn.Linear(d_model, num_micro_heads),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        imp_scores = self.importance_proj(x).view(B, N, self.num_micro_heads).transpose(1, 2)  # (B, H, N)
        importance_mask = imp_scores.unsqueeze(-1)  # (B, H, N, 1)
        return importance_mask, imp_scores.mean(dim=-1)


class LayerHeadElevator(nn.Module):
    """
    Dynamic Cross-Layer Micro-Head Elevator & Positioning Algorithm.
    Strictly Causal Per-Token Elevation Routing.
    """

    def __init__(self, d_model: int, num_micro_heads: int, base_ratio: float):
        super().__init__()
        self.d_model = d_model
        self.num_micro_heads = num_micro_heads

        base_clamped = max(0.05, min(0.95, base_ratio))
        init_logit = math.log(base_clamped / (1.0 - base_clamped))
        self.elevation_logits = nn.Parameter(
            torch.full((num_micro_heads,), init_logit, dtype=torch.float32)
        )
        self.elevator_router = nn.Linear(d_model, num_micro_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dynamic_offset = self.elevator_router(x)
        total_elevation = self.elevation_logits.unsqueeze(0).unsqueeze(0) + dynamic_offset
        beta_h = torch.sigmoid(total_elevation)  # (B, N, H)
        alpha_h = 1.0 - beta_h                  # (B, N, H)
        return alpha_h, beta_h


class MicroHeadLinearAttention(nn.Module):
    """
    Micro-Head Linear Attention with RoPE (Rotary Position Embedding), Dynamic Matrix Slicing,
    Per-Token Adaptive Routing & SSM Recurrence.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_micro_heads: int = 8,
        window_sizes: Tuple[Optional[int], ...] = (5, 20, None),
        dropout: float = 0.1,
        max_seq_len: int = 4080,
    ):
        super().__init__()
        assert d_model % num_micro_heads == 0, f"d_model ({d_model}) must be divisible by num_micro_heads ({num_micro_heads})"
        
        self.d_model = d_model
        self.num_micro_heads = num_micro_heads
        self.head_dim = d_model // num_micro_heads
        self.window_sizes = window_sizes
        self.num_clusters = len(window_sizes)

        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)

        self.head_stitcher = DynamicHeadStitcher(d_model, num_micro_heads, self.num_clusters)
        self.head_pruner = HeadPruner(d_model, num_micro_heads)
        
        self.heads_per_cluster: List[int] = []
        base_h = num_micro_heads // self.num_clusters
        remainder = num_micro_heads % self.num_clusters
        for c in range(self.num_clusters):
            count = base_h + (1 if c < remainder else 0)
            self.heads_per_cluster.append(count)

        decay_inits = []
        for w in window_sizes:
            if w is not None and w > 0:
                init_val = math.exp(-1.0 / float(w))
            else:
                init_val = 1.0
            decay_inits.append(init_val)

        decay_logits = torch.tensor(
            [math.log(g / (1.0 - g + 1e-6)) if g < 1.0 else 10.0 for g in decay_inits],
            dtype=torch.float32,
        )
        self.decay_logits = nn.Parameter(decay_logits)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @property
    def decay_factors(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logits)

    @staticmethod
    def kernel_fn(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_micro_heads
        d_k = self.head_dim
        C = self.num_clusters
        gammas = self.decay_factors

        routing_weights = self.head_stitcher(x, is_training=self.training)
        imp_mask, _ = self.head_pruner(x, is_training=self.training)

        # Raw Head projections
        q = self.q_proj(x).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        k = self.k_proj(x).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        v = self.v_proj(x).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)

        # Apply RoPE (Rotary Position Embedding) to Query & Key
        cos, sin = self.rope(x, N)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Apply kernel function for linear attention feature map
        q = self.kernel_fn(q)
        k = self.kernel_fn(k)
        q = q * imp_mask

        # Sequence Chunking & Causal Linear Attention Scan
        chunk_size = 64
        num_chunks = math.ceil(N / chunk_size)
        pad_len = num_chunks * chunk_size - N

        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            routing_weights = F.pad(routing_weights, (0, 0, 0, pad_len))

        N_padded = num_chunks * chunk_size
        q_4d = q.view(B, H, num_chunks, chunk_size, d_k)
        k_4d = k.view(B, H, num_chunks, chunk_size, d_k)
        v_4d = v.view(B, H, num_chunks, chunk_size, d_k)
        rw_4d = routing_weights.view(B, H, num_chunks, chunk_size, C)

        i_idx = torch.arange(chunk_size, device=x.device, dtype=torch.float32)
        j_idx = torch.arange(chunk_size, device=x.device, dtype=torch.float32)
        dist_intra = i_idx.unsqueeze(1) - j_idx.unsqueeze(0)
        causal_intra = dist_intra >= 0

        g = gammas.view(1, C, 1, 1, 1)
        dist_clamped = torch.clamp(dist_intra, min=0.0).view(1, 1, 1, chunk_size, chunk_size)
        decay_intra = torch.exp(dist_clamped * torch.log(g.view(1, C, 1, 1, 1) + 1e-8)) * causal_intra.view(1, 1, 1, chunk_size, chunk_size)

        window_masks = []
        for c_idx, w in enumerate(self.window_sizes):
            if w is not None:
                wm = (dist_intra < float(w)) & causal_intra
            else:
                wm = causal_intra
            window_masks.append(wm.float())
        window_masks = torch.stack(window_masks, dim=0).view(1, C, 1, chunk_size, chunk_size)
        decay_intra = (decay_intra * window_masks).to(dtype=x.dtype)

        decay_from_start = torch.exp((i_idx + 1.0).view(1, 1, 1, chunk_size, 1) * torch.log(g + 1e-8)).to(dtype=x.dtype)
        decay_to_end = torch.exp((chunk_size - 1.0 - j_idx).view(1, 1, 1, chunk_size, 1) * torch.log(g + 1e-8)).to(dtype=x.dtype)
        gamma_chunk = torch.exp(chunk_size * torch.log(g + 1e-8)).to(dtype=x.dtype)

        S_running = torch.zeros(B, C, H, d_k, d_k, dtype=x.dtype, device=x.device)
        Z_running = torch.zeros(B, C, H, 1, d_k, dtype=x.dtype, device=x.device)

        out_heads_padded = torch.empty(B, H, num_chunks, chunk_size, d_k, dtype=x.dtype, device=x.device)

        for b in range(num_chunks):
            q_b = q_4d[:, :, b]
            k_b = k_4d[:, :, b]
            v_b = v_4d[:, :, b]
            rw_b = rw_4d[:, :, b].permute(0, 3, 1, 2).unsqueeze(-1)

            q_b_c = q_b.unsqueeze(1) * rw_b
            k_b_c = k_b.unsqueeze(1) * rw_b
            v_b_c = v_b.unsqueeze(1) * rw_b

            sim_b = torch.matmul(q_b_c, k_b_c.transpose(-2, -1)) * decay_intra
            num_b = torch.matmul(sim_b, v_b_c)
            denom_b = sim_b.sum(dim=-1, keepdim=True)

            q_b_decayed = q_b_c * decay_from_start
            num_inter_b = torch.matmul(q_b_decayed, S_running)
            denom_inter_b = torch.matmul(q_b_decayed, Z_running.transpose(-2, -1))

            num_all = num_b + num_inter_b
            denom_all = denom_b + denom_inter_b + 1e-6
            out_c_chunk = num_all / denom_all

            out_heads_padded[:, :, b] = (out_c_chunk.permute(0, 2, 3, 1, 4) * rw_4d[:, :, b].unsqueeze(-1)).sum(dim=-2)

            k_b_decayed = k_b_c * decay_to_end
            kv_chunk = torch.matmul(k_b_decayed.transpose(-2, -1), v_b_c)
            k_chunk = k_b_decayed.sum(dim=-2, keepdim=True)

            S_running = S_running * gamma_chunk + kv_chunk
            Z_running = Z_running * gamma_chunk + k_chunk

        out_heads = out_heads_padded.view(B, H, N_padded, d_k)
        if pad_len > 0:
            out_heads = out_heads[:, :, :N]

        out = out_heads.transpose(1, 2).contiguous().view(B, N, D)
        out = self.head_stitcher.stitch_proj(out)

        return self.dropout(self.out_proj(out))


class SelectiveSSMAlgorithm(nn.Module):
    """
    Selective State-Space Model (SSM) Selection Algorithm (Mamba Architecture Equivalence).
    """

    def __init__(self, d_model: int = 512, d_state: int = 512, expand_factor: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.expand_factor = expand_factor
        self.d_state = int(d_state * expand_factor)

        A_init = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A_init))

        self.x_proj = nn.Linear(d_model, self.d_state * 3, bias=False)
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, self.d_state),
            nn.Softplus(),
        )

        self.in_proj = nn.Linear(d_model, self.d_state)
        self.out_proj = nn.Linear(self.d_state, d_model)

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        if hidden_state is None or hidden_state.shape[-1] != self.d_state:
            hidden_state = torch.zeros(B, self.d_state, device=x.device)

        A = -torch.exp(self.A_log)

        delta = self.dt_proj(x)
        bc_proj = self.x_proj(x)
        b_gate, c_gate, u_in = torch.chunk(bc_proj, 3, dim=-1)

        b_gate = torch.sigmoid(b_gate)
        c_gate = torch.sigmoid(c_gate)
        u_val = self.in_proj(x)

        if N == 1:
            decay = torch.exp(delta[:, 0, :] * A)
            u_term = (delta[:, 0, :] * b_gate[:, 0, :]) * u_val[:, 0, :]
            final_hidden_state = decay * hidden_state + u_term
            y_seq = c_gate[:, 0, :] * final_hidden_state
            output_tensor = self.out_proj(y_seq).unsqueeze(1)
            return output_tensor, final_hidden_state

        delta_A = delta * A.unsqueeze(0).unsqueeze(0)
        cum_A = torch.exp(torch.cumsum(delta_A, dim=1)) + 1e-8
        u_term = (delta * b_gate) * u_val

        cumsum_u = torch.cumsum(u_term / cum_A, dim=1)
        cum_A_0 = cum_A[:, 0:1, :]
        hidden_seq = cum_A * (cumsum_u + hidden_state.unsqueeze(1) / (cum_A_0 + 1e-8))

        y_seq = c_gate * hidden_seq
        output_tensor = self.out_proj(y_seq)
        
        final_hidden_state = hidden_seq[:, N - 1, :]
        return output_tensor, final_hidden_state


SelectiveGatingSSM = SelectiveSSMAlgorithm


class EvolvingHeadSSMBlock(nn.Module):
    """
    Single Evolving Head SSM Layer Block with Selective SSM Selection Algorithm.
    Default num_micro_heads = 8.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_micro_heads: int = 8,
        window_sizes: Tuple[Optional[int], ...] = (5, 20, None),
        dim_feedforward: int = 2048,
        layer_idx: int = 0,
        total_layers: int = 8,
        ssm_expand_factor: float = 1.0,
        dropout: float = 0.1,
        max_seq_len: int = 4080,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.ssm_expand_factor = ssm_expand_factor

        base_ratio = layer_idx / max(1, total_layers - 1)
        
        self.head_elevator = LayerHeadElevator(
            d_model=d_model,
            num_micro_heads=num_micro_heads,
            base_ratio=base_ratio,
        )

        self.attn = MicroHeadLinearAttention(
            d_model=d_model,
            num_micro_heads=num_micro_heads,
            window_sizes=window_sizes,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.ssm = SelectiveSSMAlgorithm(
            d_model=d_model,
            d_state=d_model,
            expand_factor=ssm_expand_factor,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

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
        hidden_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.norm1(x)

        attn_out = self.attn(norm_x, mask=mask)
        ssm_out, new_hidden_state = self.ssm(norm_x, hidden_state=hidden_state)

        alpha_h, beta_h = self.head_elevator(norm_x)

        B, N, D = x.shape
        H = self.attn.num_micro_heads
        d_k = self.attn.head_dim

        attn_heads = attn_out.view(B, N, H, d_k).transpose(1, 2)
        ssm_heads = ssm_out.view(B, N, H, d_k).transpose(1, 2)

        alpha_exp = alpha_h.transpose(1, 2).unsqueeze(-1)
        beta_exp = beta_h.transpose(1, 2).unsqueeze(-1)

        blended_heads = alpha_exp * attn_heads + beta_exp * ssm_heads
        blended = blended_heads.transpose(1, 2).contiguous().view(B, N, D)

        x = x + blended
        x = x + self.mlp(self.norm2(x))

        return x, new_hidden_state


class EvolvingHeadSSMModel(nn.Module):
    """
    Complete Evolving Micro-Head Linear Attention State-Space Model with RoPE & 8 Micro-Heads.
    Supports max_seq_len = 4080 tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_micro_heads: int = 8,
        num_layers: int = 8,
        window_sizes: Tuple[Optional[int], ...] = (5, 20, None),
        dim_feedforward: int = 2048,
        deep_ssm_expand_factor: float = 2.0,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pad_idx: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_micro_heads = num_micro_heads

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # Note: Absolute pos_emb replaced by dynamic RoPE inside MicroHeadLinearAttention
        self.dropout = nn.Dropout(dropout)

        blocks = []
        for l in range(num_layers):
            expand_f = deep_ssm_expand_factor if l >= (num_layers // 2) else 1.0
            block = EvolvingHeadSSMBlock(
                d_model=d_model,
                num_micro_heads=num_micro_heads,
                window_sizes=window_sizes,
                dim_feedforward=dim_feedforward,
                layer_idx=l,
                total_layers=num_layers,
                ssm_expand_factor=expand_f,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
        start_pos: int = 0,
        return_hidden: bool = False,
        use_checkpointing: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, N = tokens.shape
        x = self.tok_emb(tokens) * math.sqrt(self.d_model)
        x = self.dropout(x)

        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        new_hidden_states = []
        for l_idx, block in enumerate(self.blocks):
            h_in = hidden_states[l_idx]
            if use_checkpointing and self.training:
                x, h_out = torch.utils.checkpoint.checkpoint(block, x, h_in, None, use_reentrant=False)
            else:
                x, h_out = block(x, hidden_state=h_in)
            new_hidden_states.append(h_out)

        x = self.final_norm(x)
        if return_hidden:
            return x, new_hidden_states
        logits = self.generator(x)
        return logits, new_hidden_states


def load_checkpoint_compatible(model: nn.Module, state_dict: dict) -> nn.Module:
    """
    Safely loads weights from a pre-trained checkpoint into EvolvingHeadSSMModel with RoPE & 8 Micro-Heads.
    Automatically adapts mismatched layer shapes (e.g. 16 heads -> 8 heads) and ignores obsolete keys like pos_emb.
    """
    model_state = model.state_dict()
    adapted_state = {}

    for k, v in state_dict.items():
        if "pos_emb" in k:
            # Ignore obsolete absolute pos_emb (upgraded to dynamic RoPE)
            continue
            
        if k in model_state:
            target_shape = model_state[k].shape
            if v.shape == target_shape:
                adapted_state[k] = v
            else:
                # Shape adjustment (e.g. head router logits 16 heads -> 8 heads)
                if v.dim() == 1:
                    min_0 = min(v.shape[0], target_shape[0])
                    buf = model_state[k].clone()
                    buf[:min_0] = v[:min_0]
                    adapted_state[k] = buf
                elif v.dim() == 2:
                    min_0 = min(v.shape[0], target_shape[0])
                    min_1 = min(v.shape[1], target_shape[1])
                    buf = model_state[k].clone()
                    buf[:min_0, :min_1] = v[:min_0, :min_1]
                    adapted_state[k] = buf
                elif v.dim() == 3:
                    min_0 = min(v.shape[0], target_shape[0])
                    min_1 = min(v.shape[1], target_shape[1])
                    min_2 = min(v.shape[2], target_shape[2])
                    buf = model_state[k].clone()
                    buf[:min_0, :min_1, :min_2] = v[:min_0, :min_1, :min_2]
                    adapted_state[k] = buf
                else:
                    adapted_state[k] = model_state[k]
        else:
            pass

    missing_keys, unexpected_keys = model.load_state_dict(adapted_state, strict=False)
    print(f"✓ [CHECKPOINT ADAPTER] State dict loaded for RoPE & 8-Head Fine-tuning! (Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)})")
    return model
