from pathlib import Path
import json
import os
import random
import re
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer_v7 import OwnTokenizerV7
from model.own_ai_v7 import OwnAIv7


INSTRUCTION_FILE = ROOT / "data" / "processed" / "instructions_v7.jsonl"
BASE_MODEL = ROOT / "checkpoints" / "own_ai_v7_pretrain_best.pt"
BASE_TOKENIZER = ROOT / "checkpoints" / "tokenizer_v7.json"

OUT_MODEL = ROOT / "checkpoints" / "own_ai_v7_sft_best.pt"
OUT_TOKENIZER = ROOT / "checkpoints" / "tokenizer_v7_sft.json"

BLOCK_SIZE = int(os.getenv("OWN_AI_V7_CONTEXT", "512"))
BATCH_SIZE = int(os.getenv("OWN_AI_V7_SFT_BATCH", "2"))
EPOCHS = int(os.getenv("OWN_AI_V7_SFT_EPOCHS", "5"))
LR = float(os.getenv("OWN_AI_V7_SFT_LR", "0.00005"))
PATIENCE = 5
SEED = 42


def load_examples():
    examples = []

    with INSTRUCTION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = str(
                item.get("instruction", "")
            ).strip()

            answer = str(
                item.get("response", "")
            ).strip()

            if question and answer:
                examples.append(
                    (question, answer)
                )

    if not examples:
        raise ValueError(
            "No instruction examples found."
        )

    return examples


def encode_example(tokenizer, question, answer):
    prefix = (
        f"User: {question}\n"
        "Assistant:"
    )

    answer_text = f" {answer}"

    prefix_ids = tokenizer.encode(prefix)
    answer_ids = tokenizer.encode(
        answer_text
    )

    tokens = (
        [tokenizer.bos_id]
        + prefix_ids
        + answer_ids
        + [tokenizer.eos_id]
    )

    mask = (
        [0] * (1 + len(prefix_ids))
        + [1] * (len(answer_ids) + 1)
    )

    tokens = tokens[
        :BLOCK_SIZE + 1
    ]

    mask = mask[
        :BLOCK_SIZE + 1
    ]

    return (
        torch.tensor(
            tokens,
            dtype=torch.long,
        ),
        torch.tensor(
            mask,
            dtype=torch.float32,
        ),
    )


def collate(batch, pad_id):
    max_len = max(
        len(item[0])
        for item in batch
    )

    max_len = min(
        BLOCK_SIZE + 1,
        max_len,
    )

    xs = []
    ys = []
    masks = []

    for tokens, mask in batch:
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
        token_loss * mask
    ).sum() / mask.sum().clamp_min(1.0)


def make_encoded(examples, tokenizer):
    result = [
        encode_example(
            tokenizer,
            question,
            answer,
        )
        for question, answer in examples
    ]

    return result


@torch.no_grad()
def evaluate(model, encoded, tokenizer, device):
    model.eval()

    values = []

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

        values.append(
            masked_loss(
                model,
                x,
                y,
                mask,
            ).item()
        )

    model.train()

    return sum(values) / max(
        1,
        len(values),
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    for path in (
        INSTRUCTION_FILE,
        BASE_MODEL,
        BASE_TOKENIZER,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = OwnTokenizerV7(
        vocab_size=2048
    )

    tokenizer.load(
        BASE_TOKENIZER
    )

    examples = load_examples()
    random.shuffle(examples)

    split = max(
        1,
        int(len(examples) * 0.95),
    )

    train_examples = examples[:split]
    val_examples = examples[split:]

    train_encoded = make_encoded(
        train_examples,
        tokenizer,
    )

    val_encoded = make_encoded(
        val_examples,
        tokenizer,
    )

    checkpoint = torch.load(
        BASE_MODEL,
        map_location=device,
        weights_only=False,
    )

    model = OwnAIv7(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint.get(
            "dropout",
            0.1,
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device == "cuda",
    )

    best = float("inf")
    bad = 0

    print("=" * 60)
    print("OWN AI v7 TRUE INSTRUCTION SFT")
    print("=" * 60)
    print("Device:", device)
    print("Examples:", len(examples))
    print("Train:", len(train_encoded))
    print("Validation:", len(val_encoded))
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("Vocabulary:", tokenizer.vocab_size)
    print("Context:", BLOCK_SIZE)
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(train_encoded)

        running = []

        for start in range(
            0,
            len(train_encoded),
            BATCH_SIZE,
        ):
            batch = train_encoded[
                start:start + BATCH_SIZE
            ]

            x, y, mask = collate(
                batch,
                tokenizer.pad_id,
            )

            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

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

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            running.append(
                loss.item()
            )

        val = evaluate(
                model,
                val_encoded,
                tokenizer,
                device,
            )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train: {sum(running)/max(1,len(running)):.4f} | "
            f"Val: {val:.4f}"
        )

        if val < best:
            best = val
            bad = 0

            torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocab_size": checkpoint["vocab_size"],
                        "block_size": checkpoint["block_size"],
                        "d_model": checkpoint["d_model"],
                        "n_heads": checkpoint["n_heads"],
                        "n_layers": checkpoint["n_layers"],
                        "dropout": checkpoint.get("dropout", 0.1),
                        "stage": "v7_sft",
                        "best_val_loss": best,
                        "instruction_examples": len(examples),
                    },
                    OUT_MODEL,
                )

                tokenizer.save(
                    OUT_TOKENIZER
                )

            print("  Saved best v7 SFT checkpoint.")
        else:
            bad += 1

            if bad >= PATIENCE:
                print("  Early stopping.")
                break

    print("=" * 60)
    print("v7 SFT FINISHED")
    print("Best validation loss:", f"{best:.4f}")
    print("Model:", OUT_MODEL)
    print("=" * 60)


if __name__ == "__main__":
    main()
