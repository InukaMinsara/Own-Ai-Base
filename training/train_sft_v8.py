from pathlib import Path
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer_v8 import OwnTokenizerV8
from model.own_ai_v8 import OwnAIv8


INSTRUCTION_FILE = ROOT / "data" / "processed" / "sft_v8.jsonl"
PRETRAIN_CANDIDATES = [
    (
        ROOT / "checkpoints" / "own_ai_v8_streaming_best.pt",
        ROOT / "checkpoints" / "tokenizer_v8_streaming.json",
    ),
    (
        ROOT / "checkpoints" / "own_ai_v8_pretrain_best.pt",
        ROOT / "checkpoints" / "tokenizer_v8.json",
    ),
]
OUT_MODEL = ROOT / "checkpoints" / "own_ai_v8_sft_best.pt"
OUT_TOKENIZER = ROOT / "checkpoints" / "tokenizer_v8_sft.json"

BLOCK_SIZE = int(os.getenv("OWN_AI_V8_CONTEXT", "1024"))
BATCH_SIZE = int(os.getenv("OWN_AI_V8_SFT_BATCH", "1"))
ACCUM = int(os.getenv("OWN_AI_V8_SFT_ACCUM", "8"))
EPOCHS = int(os.getenv("OWN_AI_V8_SFT_EPOCHS", "3"))
MAX_BATCHES = int(os.getenv("OWN_AI_V8_SFT_MAX_BATCHES", "0"))
LOG_EVERY = int(os.getenv("OWN_AI_V8_SFT_LOG_EVERY", "100"))
LR = float(os.getenv("OWN_AI_V8_SFT_LR", "0.00005"))
USE_CHECKPOINTING = True
SEED = 42


def load_examples():
    rows = []

    with INSTRUCTION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            q = str(
                item.get("instruction", "")
            ).strip()
            a = str(
                item.get("response", "")
            ).strip()
            group = str(
                item.get("group_id", q)
            )

            if q and a:
                rows.append(
                    {
                        "question": q,
                        "answer": a,
                        "group_id": group,
                    }
                )

    if not rows:
        raise ValueError(
            "No v8 instruction examples found."
        )

    return rows


def split_by_group(rows):
    groups = {}
    for row in rows:
        groups.setdefault(
            row["group_id"],
            [],
        ).append(row)

    keys = list(groups)
    random.Random(SEED).shuffle(keys)

    val_groups = max(
        1,
        int(len(keys) * 0.05),
    )

    val_keys = set(
        keys[:val_groups]
    )

    train = []
    val = []

    for key, items in groups.items():
        (val if key in val_keys else train).extend(items)

    return train, val


def encode_example(tokenizer, row):
    prefix = (
        f"User: {row['question']}\n"
        "Assistant:"
    )
    answer = " " + row["answer"]

    prefix_ids = tokenizer.encode(prefix)
    answer_ids = tokenizer.encode(answer)

    tokens = (
        [tokenizer.bos_id]
        + prefix_ids
        + answer_ids
        + [tokenizer.eos_id]
    )

    mask = (
        [0] * (1 + len(prefix_ids))
        + [1] * len(answer_ids)
        + [1]
    )

    tokens = tokens[: BLOCK_SIZE + 1]
    mask = mask[: BLOCK_SIZE + 1]

    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(mask, dtype=torch.float32),
    )


def collate(encoded, pad_id):
    max_len = max(
        len(tokens)
        for tokens, _ in encoded
    )
    max_len = min(
        BLOCK_SIZE + 1,
        max_len,
    )

    xs = []
    ys = []
    masks = []

    for tokens, mask in encoded:
        tokens = tokens[:max_len]
        mask = mask[:max_len]

        missing = max_len - len(tokens)
        if missing:
            tokens = torch.cat(
                [
                    tokens,
                    torch.full(
                        (missing,),
                        pad_id,
                        dtype=torch.long,
                    ),
                ]
            )
            mask = torch.cat(
                [
                    mask,
                    torch.zeros(
                        missing,
                        dtype=torch.float32,
                    ),
                ]
            )

        xs.append(tokens[:-1])
        ys.append(tokens[1:])
        masks.append(mask[1:])

    return (
        torch.stack(xs),
        torch.stack(ys),
        torch.stack(masks),
    )


def masked_loss(model, x, y, mask):
    logits, _ = model(x)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).reshape_as(mask)

    return (
        (token_loss * mask).sum()
        / mask.sum().clamp_min(1.0)
    )


@torch.no_grad()
def evaluate(model, rows, tokenizer, device):
    model.eval()
    total = 0.0
    count = 0

    encoded = [
        encode_example(tokenizer, row)
        for row in rows
    ]

    for start in range(
        0,
        len(encoded),
        BATCH_SIZE,
    ):
        batch = encoded[
            start:start + BATCH_SIZE
        ]
        x, y, mask = collate(
            batch,
            tokenizer.pad_id,
        )

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        total += masked_loss(
            model,
            x,
            y,
            mask,
        ).item()
        count += 1

    model.train()
    return total / max(1, count)


def main():
    global BLOCK_SIZE

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if not INSTRUCTION_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {INSTRUCTION_FILE}"
        )

    base_model = None
    base_tokenizer = None

    for candidate_model, candidate_tokenizer in PRETRAIN_CANDIDATES:
        if candidate_model.exists() and candidate_tokenizer.exists():
            base_model = candidate_model
            base_tokenizer = candidate_tokenizer
            break

    if base_model is None:
        raise FileNotFoundError(
            "No v8 pretrained checkpoint/tokenizer found."
        )

    rows = load_examples()
    train_rows, val_rows = split_by_group(
        rows
    )

    tokenizer = OwnTokenizerV8(
        vocab_size=4096
    )
    tokenizer.load(
        base_tokenizer
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        base_model,
        map_location=device,
        weights_only=False,
    )

    block_size = int(
        checkpoint["block_size"]
    )

    model = OwnAIv8(
        vocab_size=checkpoint["vocab_size"],
        block_size=block_size,
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint.get(
            "dropout",
            0.1,
        ),
        gradient_checkpointing=USE_CHECKPOINTING,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device == "cuda",
    )

    best = float("inf")

    print("=" * 64)
    print("OWN AI v8 TRUE INSTRUCTION SFT")
    print("=" * 64)
    print("Device:", device)
    print("Examples:", len(rows))
    print("Train:", len(train_rows))
    print("Validation:", len(val_rows))
    print("Context:", BLOCK_SIZE)
    print("Batch:", BATCH_SIZE, "x", ACCUM)
    print("=" * 64)

    # Match the context used by the pretrained checkpoint.
    BLOCK_SIZE = block_size

    for epoch in range(EPOCHS):
        random.shuffle(train_rows)
        running = 0.0
        optimizer.zero_grad(
            set_to_none=True
        )
        batch_count = 0
        effective_rows = (
            len(train_rows)
            if MAX_BATCHES <= 0
            else min(
                len(train_rows),
                MAX_BATCHES * BATCH_SIZE,
            )
        )

        for index in range(
            0,
            effective_rows,
            BATCH_SIZE,
        ):
            batch_rows = train_rows[
                index:index + BATCH_SIZE
            ]

            encoded = [
                encode_example(
                    tokenizer,
                    row,
                )
                for row in batch_rows
            ]

            x, y, mask = collate(
                encoded,
                tokenizer.pad_id,
            )

            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            with torch.amp.autocast(
                "cuda",
                enabled=device == "cuda",
            ):
                loss = masked_loss(
                    model,
                    x,
                    y,
                    mask,
                )
                scaled_loss = loss / ACCUM

            scaler.scale(
                scaled_loss
            ).backward()

            running += loss.item()
            batch_count += 1

            if LOG_EVERY > 0 and batch_count % LOG_EVERY == 0:
                print(
                    f"Epoch {epoch + 1:02d}/{EPOCHS} | "
                    f"Batch {batch_count}/{max(1, (effective_rows + BATCH_SIZE - 1) // BATCH_SIZE)} | "
                    f"Loss: {loss.item():.4f}"
                )

            should_step = (
                ((index // BATCH_SIZE) + 1) % ACCUM == 0
                or index + BATCH_SIZE >= effective_rows
            )

            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(
                    set_to_none=True
                )

        val = evaluate(
            model,
            val_rows,
            tokenizer,
            device,
        )

        train_avg = running / max(
            1,
            max(1, (effective_rows + BATCH_SIZE - 1) // BATCH_SIZE),
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train: {train_avg:.4f} | "
            f"Val: {val:.4f}"
        )

        if val < best:
            best = val

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": checkpoint["vocab_size"],
                    "block_size": checkpoint["block_size"],
                    "d_model": checkpoint["d_model"],
                    "n_heads": checkpoint["n_heads"],
                    "n_layers": checkpoint["n_layers"],
                    "dropout": checkpoint.get("dropout", 0.1),
                    "stage": "v8_sft",
                    "best_val_loss": best,
                    "instruction_examples": len(rows),
                },
                OUT_MODEL,
            )

            tokenizer.save(
                OUT_TOKENIZER
            )

            print("  Saved best v8 SFT checkpoint.")

    print("=" * 64)
    print("v8 SFT FINISHED")
    print("Best validation loss:", f"{best:.4f}")
    print("Max batches:", MAX_BATCHES if MAX_BATCHES > 0 else "all")
    print("Model:", OUT_MODEL)
    print("=" * 64)


if __name__ == "__main__":
    main()
