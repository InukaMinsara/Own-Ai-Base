from pathlib import Path
import os
import random
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.build_instruction_dataset_v7 import main as build_instructions
from data.build_corpus_v7 import main as build_corpus
from tokenizer.tokenizer_v7 import OwnTokenizerV7
from model.own_ai_v7 import OwnAIv7


DATA_FILE = ROOT / "data" / "processed" / "corpus_v7.txt"
MODEL_FILE = ROOT / "checkpoints" / "own_ai_v7_pretrain_best.pt"
TOKENIZER_FILE = ROOT / "checkpoints" / "tokenizer_v7.json"

VOCAB_SIZE = int(os.getenv("OWN_AI_V7_VOCAB", "2048"))
BLOCK_SIZE = int(os.getenv("OWN_AI_V7_CONTEXT", "512"))
D_MODEL = int(os.getenv("OWN_AI_V7_DIM", "384"))
N_HEADS = int(os.getenv("OWN_AI_V7_HEADS", "8"))
N_LAYERS = int(os.getenv("OWN_AI_V7_LAYERS", "12"))
BATCH_SIZE = int(os.getenv("OWN_AI_V7_BATCH", "2"))
GRAD_ACCUM = int(os.getenv("OWN_AI_V7_ACCUM", "4"))
MAX_STEPS = int(os.getenv("OWN_AI_V7_STEPS", "3000"))
LR = float(os.getenv("OWN_AI_V7_LR", "0.0003"))
WEIGHT_DECAY = 0.01
SEED = 42
EVAL_INTERVAL = 100


def batch(data, batch_size):
    max_start = len(data) - BLOCK_SIZE - 1

    if max_start <= 0:
        raise ValueError(
            "Corpus is smaller than the requested context window."
        )

    positions = torch.randint(
        0,
        max_start,
        (batch_size,),
    )

    x = torch.stack(
        [
            data[i:i + BLOCK_SIZE]
            for i in positions
        ]
    )

    y = torch.stack(
        [
            data[i + 1:i + BLOCK_SIZE + 1]
            for i in positions
        ]
    )

    return x, y


@torch.no_grad()
def eval_loss(model, data, device):
    model.eval()
    values = []

    for _ in range(10):
        x, y = batch(data, BATCH_SIZE)
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

    if not (
        ROOT / "data" / "processed" / "instructions_v7.jsonl"
    ).exists():
        build_instructions()

    build_corpus()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    text = DATA_FILE.read_text(
        encoding="utf-8"
    )

    tokenizer = OwnTokenizerV7(
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

    print("=" * 56)
    print("OWN AI v7 PRETRAINING")
    print("=" * 56)
    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("Vocabulary:", tokenizer.vocab_size)
    print("Tokens:", len(token_ids))
    print("Train tokens:", len(train_data))
    print("Val tokens:", len(val_data))
    print("Model:", D_MODEL, "dim /", N_HEADS, "heads /", N_LAYERS, "layers")
    print("Context:", BLOCK_SIZE)
    print("Batch:", BATCH_SIZE, "x", GRAD_ACCUM)
    print("Steps:", MAX_STEPS)
    print("=" * 56)

    model = OwnAIv7(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.1,
    ).to(device)

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device == "cuda",
    )

    best_val = float("inf")

    model.train()

    for step in range(MAX_STEPS):
        optimizer.zero_grad(set_to_none=True)

        for _ in range(GRAD_ACCUM):
            x, y = batch(
                train_data,
                BATCH_SIZE,
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

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            val = eval_loss(
                model,
                val_data,
                device,
            )

            print(
                f"Step {step:04d} | "
                f"Val: {val:.4f}"
            )

            if val < best_val:
                best_val = val

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocab_size": tokenizer.vocab_size,
                        "block_size": BLOCK_SIZE,
                        "d_model": D_MODEL,
                        "n_heads": N_HEADS,
                        "n_layers": N_LAYERS,
                        "dropout": 0.1,
                        "stage": "v7_pretrain",
                        "best_val_loss": best_val,
                    },
                    MODEL_FILE,
                )

                tokenizer.save(
                    TOKENIZER_FILE
                )

                print("  Saved best v7 checkpoint.")

    print("=" * 56)
    print("v7 PRETRAINING FINISHED")
    print("Best validation loss:", f"{best_val:.4f}")
    print("Model:", MODEL_FILE)
    print("=" * 56)


if __name__ == "__main__":
    main()
