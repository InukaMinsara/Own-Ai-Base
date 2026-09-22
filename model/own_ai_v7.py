import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        hidden = int((8 * d_model) / 3)
        hidden = ((hidden + 63) // 64) * 64

        self.gate = nn.Linear(d_model, hidden)
        self.up = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(
            self.down(
                F.silu(self.gate(x)) * self.up(x)
            )
        )


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            dropout,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class OwnAIv7(nn.Module):
    """Scalable local decoder-only Transformer.

    The default profile is ~20M parameters and a 512-token context.
    Change the constructor for larger models when hardware permits.
    """

    def __init__(
        self,
        vocab_size,
        block_size=512,
        d_model=384,
        n_heads=8,
        n_layers=12,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        _, t = idx.shape

        if t > self.block_size:
            raise ValueError(
                f"Sequence length {t} exceeds context {self.block_size}"
            )

        positions = torch.arange(
            t,
            device=idx.device,
        )

        x = self.token_embedding(idx)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_norm(x))
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss
