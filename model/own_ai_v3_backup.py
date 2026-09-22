import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.0):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(
            torch.ones(block_size, block_size, dtype=torch.bool)
        )

        self.register_buffer(
            "mask",
            mask.view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attention = attention.masked_fill(
            ~self.mask[:, :, :T, :T],
            float("-inf")
        )

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        y = attention @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model,
            n_heads,
            block_size
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class OwnAI(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        d_model=128,
        n_heads=4,
        n_layers=4
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            block_size,
            d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                block_size
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.output = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                "Input is longer than block_size"
            )

        positions = torch.arange(
            T,
            device=idx.device
        )

        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.output(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )

        return logits, loss


if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = OwnAI(
        vocab_size=128,
        block_size=64,
        d_model=128,
        n_heads=4,
        n_layers=4
    ).to(device)

    input_ids = torch.randint(
        0,
        128,
        (2, 64),
        device=device
    )

    targets = torch.randint(
        0,
        128,
        (2, 64),
        device=device
    )

    logits, loss = model(
        input_ids,
        targets
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("================================")
    print("        OWN AI MODEL TEST")
    print("================================")
    print("Device:", device)
    print("Parameters:", f"{parameters:,}")
    print("Output shape:", tuple(logits.shape))
    print("Test loss:", round(loss.item(), 4))
    print("================================")
    print("OWN AI MODEL: PASSED")