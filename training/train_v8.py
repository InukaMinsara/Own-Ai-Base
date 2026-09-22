from pathlib import Path
import os
import random
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.build_v8_dataset import main as build_dataset
from tokenizer.tokenizer_v8 import OwnTokenizerV8
from model.own_ai_v8 import OwnAIv8


DATA_FILE = ROOT / "data" / "processed" / "corpus_v8.txt"
MODEL_FILE = ROOT / "checkpoints" / "own_ai_v8_pretrain_best.pt"
TOKENIZER_FILE = ROOT / "checkpoints" / "tokenizer_v8.json"

VOCAB_SIZE = int(os.getenv("OWN_AI_V8_VOCAB", "4096"))
BLOCK_SIZE = int(os.getenv("OWN_AI_V8_CONTEXT", "1024"))
D_MODEL = int(os.getenv("OWN_AI_V8_DIM", "384"))
N_HEADS = int(os.getenv("OWN_AI_V8_HEADS", "8"))
N_LAYERS = int(os.getenv("OWN_AI_V8_LAYERS", "16"))
BATCH_SIZE = int(os.getenv("OWN_AI_V8_BATCH", "1"))
GRAD_ACCUM = int(os.getenv("OWN_AI_V8_ACCUM", "8"))
MAX_STEPS = int(os.getenv("OWN_AI_V8_STEPS", "5000"))
LR = float(os.getenv("OWN_AI_V8_LR", "0.00025"))
WEIGHT_DECAY = float(os.getenv("OWN_AI_V8_WEIGHT_DECAY", "0.05"))
USE_CHECKPOINTING = os.getenv(
    "OWN_AI_V8_CHECKPOINTING",
    "1",
).lower() not in {"0", "false", "no"}
SEED = 42
EVAL_INTERVAL = 100


def batch(data, batch_size, block_size):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            "Corpus is smaller than the requested context window."
        )

    starts = torch.randint(
        0,
        max_start,
        (batch_size,),
    )

    x = torch.stack(
        [
            data[i:i + block_size]
            for i in starts
        ]
    )
    y = torch.stack(
        [
            data[i + 1:i + block_size + 1]
            for i in starts
        ]
    )
    return x, y


@torch.no_grad()
def evaluate(model, data, device, block_size, batch_size):
    model.eval()
    values = []
    repeats = min(20, max(4, len(data) // max(block_size, 1)))

    for _ in range(repeats):
        x, y = batch(
            data,
            batch_size,
            block_size,
        )
        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)
        values.append(loss.item())

    model.train()
    return sum(values) / max(1, len(values))


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if not DATA_FILE.exists():
        build_dataset()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    text = DATA_FILE.read_text(
        encoding="utf-8"
    )

    tokenizer = OwnTokenizerV8(
        vocab_size=VOCAB_SIZE
    )
    tokenizer.train(text)

    token_ids = torch.tensor(
        tokenizer.encode(
            text,
            add_special_tokens=True,
        ),
        dtype=torch.long,
    )

    split = int(len(token_ids) * 0.95)
    train_data = token_ids[:split]
    val_data = token_ids[split:]

    model = OwnAIv8(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.1,
        gradient_checkpointing=USE_CHECKPOINTING,
    ).to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("=" * 64)
    print("OWN AI v8 PRETRAINING")
    print("=" * 64)
    print("Device:", device)
    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
    print("Tokens:", len(token_ids))
    print("Train tokens:", len(train_data))
    print("Validation tokens:", len(val_data))
    print("Vocabulary:", tokenizer.vocab_size)
    print("Model:", f"{D_MODEL}d / {N_HEADS} heads / {N_LAYERS} layers")
    print("Context:", BLOCK_SIZE)
    print("Parameters:", f"{params:,}")
    print("Batch:", BATCH_SIZE, "x", GRAD_ACCUM)
    print("Steps:", MAX_STEPS)
    print("Gradient checkpointing:", USE_CHECKPOINTING)
    print("=" * 64)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device == "cuda",
    )

    best = float("inf")
    model.train()

    for step in range(MAX_STEPS):
        optimizer.zero_grad(
            set_to_none=True
        )

        running = 0.0

        for _ in range(GRAD_ACCUM):
            x, y = batch(
                train_data,
                BATCH_SIZE,
                BLOCK_SIZE,
            )
            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast(
                "cuda",
                enabled=device == "cuda",
            ):
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            running += loss.item()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        if step % EVAL_INTERVAL == 0:
            val = evaluate(
                model,
                val_data,
                device,
                BLOCK_SIZE,
                BATCH_SIZE,
            )
            print(
                f"Step {step:05d} | "
                f"Train: {running:.4f} | "
                f"Val: {val:.4f}"
            )

            if val < best:
                best = val
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocab_size": tokenizer.vocab_size,
                        "block_size": BLOCK_SIZE,
                        "d_model": D_MODEL,
                        "n_heads": N_HEADS,
                        "n_layers": N_LAYERS,
                        "dropout": 0.1,
                        "stage": "v8_pretrain",
                        "best_val_loss": best,
                    },
                    MODEL_FILE,
                )
                tokenizer.save(
                    TOKENIZER_FILE
                )
                print("  Saved best v8 pretraining checkpoint.")

    val = evaluate(
        model,
        val_data,
        device,
        BLOCK_SIZE,
        BATCH_SIZE,
    )

    if val < best:
        torch.save(
            {
                "model_state": model.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "block_size": BLOCK_SIZE,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "dropout": 0.1,
                "stage": "v8_pretrain",
                "best_val_loss": val,
            },
            MODEL_FILE,
        )
        tokenizer.save(TOKENIZER_FILE)

    print("=" * 64)
    print("v8 PRETRAINING FINISHED")
    print("Best validation loss:", f"{min(best, val):.4f}")
    print("Model:", MODEL_FILE)
    print("=" * 64)


if __name__ == "__main__":
    main()
