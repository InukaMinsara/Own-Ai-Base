import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(
            -1,
            keepdim=True,
        )
        return self.weight * x * torch.rsqrt(
            variance + self.eps
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        if head_dim % 2:
            raise ValueError(
                "RoPE requires an even head dimension."
            )

        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        freqs = torch.outer(
            positions,
            inv_freq,
        )

        self.register_buffer(
            "cos",
            freqs.cos()[None, None, :, :],
            persistent=False,
        )
        self.register_buffer(
            "sin",
            freqs.sin()[None, None, :, :],
            persistent=False,
        )

    def forward(self, q, k):
        t = q.size(2)
        cos = self.cos[:, :, :t, :]
        sin = self.sin[:, :, :t, :]

        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]
        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]

        q = torch.stack(
            [
                q_even * cos - q_odd * sin,
                q_even * sin + q_odd * cos,
            ],
            dim=-1,
        ).flatten(-2)

        k = torch.stack(
            [
                k_even * cos - k_odd * sin,
                k_even * sin + k_odd * cos,
            ],
            dim=-1,
        ).flatten(-2)

        return q, k


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout,
        max_seq_len,
    ):
        super().__init__()

        if d_model % n_heads:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(
            d_model,
            d_model * 3,
        )
        self.out_proj = nn.Linear(
            d_model,
            d_model,
        )
        self.dropout = dropout
        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=max_seq_len,
        )

    def forward(self, x):
        b, t, c = x.shape

        q, k, v = self.qkv(x).chunk(
            3,
            dim=-1,
        )

        q = q.view(
            b,
            t,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = k.view(
            b,
            t,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = v.view(
            b,
            t,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        q, k = self.rope(q, k)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=True,
        )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(b, t, c)
        )

        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        hidden = int(
            (8 * d_model) / 3
        )
        hidden = (
            (hidden + 63) // 64
        ) * 64

        self.gate = nn.Linear(
            d_model,
            hidden,
        )
        self.up = nn.Linear(
            d_model,
            hidden,
        )
        self.down = nn.Linear(
            hidden,
            d_model,
        )
        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):
        return self.dropout(
            self.down(
                F.silu(self.gate(x))
                * self.up(x)
            )
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout,
        max_seq_len,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            dropout,
            max_seq_len,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(
            d_model,
            dropout,
        )

    def forward(self, x):
        x = x + self.attn(
            self.norm1(x)
        )
        x = x + self.mlp(
            self.norm2(x)
        )
        return x


class OwnAIv8(nn.Module):
    """~30M parameter local decoder Transformer.

    Default profile:
      d_model=384, heads=8, layers=16, context=1024.

    It uses RMSNorm + RoPE + SwiGLU and weight tying.
    """

    def __init__(
        self,
        vocab_size,
        block_size=1024,
        d_model=384,
        n_heads=8,
        n_layers=16,
        dropout=0.1,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.block_size = int(block_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.gradient_checkpointing = bool(
            gradient_checkpointing
        )

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )
        self.dropout = nn.Dropout(
            dropout
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dropout,
                    block_size,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = RMSNorm(
            d_model
        )
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(
            module,
            (nn.Linear, nn.Embedding),
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )
            if (
                isinstance(module, nn.Linear)
                and module.bias is not None
            ):
                nn.init.zeros_(
                    module.bias
                )

    def _forward_block(self, block, x):
        if (
            self.training
            and self.gradient_checkpointing
        ):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                block,
                x,
                use_reentrant=False,
            )
        return block(x)

    def forward(self, idx, targets=None):
        _, t = idx.shape

        if t > self.block_size:
            raise ValueError(
                f"Sequence length {t} exceeds context {self.block_size}"
            )

        x = self.token_embedding(idx)
        x = self.dropout(x)

        for block in self.blocks:
            x = self._forward_block(
                block,
                x,
            )

        logits = self.lm_head(
            self.final_norm(x)
        )

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
            )

        return logits, loss
