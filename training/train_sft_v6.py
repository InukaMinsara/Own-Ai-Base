from pathlib import Path
import random
import re
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import OwnTokenizer
from model.own_ai import OwnAI

BASE_CHECKPOINT = ROOT / "checkpoints" / "own_ai_best.pt"
BASE_TOKENIZER = ROOT / "checkpoints" / "tokenizer_best.json"
INSTRUCTION_FILE = ROOT / "data" / "processed" / "instructions.txt"

OUT_CHECKPOINT = ROOT / "checkpoints" / "own_ai_v6_sft_best.pt"
OUT_TOKENIZER = ROOT / "checkpoints" / "tokenizer_v6.json"

BLOCK_SIZE = 256
BATCH_SIZE = 4
MAX_STEPS = 1200
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
TRAIN_RATIO = 0.9
EVAL_INTERVAL = 50
PATIENCE = 6
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_instruction_file(path):
    text = path.read_text(encoding="utf-8")
    examples = []

    for chunk in text.split("<BOS>"):
        chunk = chunk.strip()
        if not chunk:
            continue

        chunk = chunk.split("<EOS>", 1)[0].strip()

        user_match = re.search(
            r"User:\s*(.*?)\s*Assistant:",
            chunk,
            re.S,
        )

        if not user_match:
            continue

        user_text = user_match.group(1).strip()
        answer_text = chunk[user_match.end():].strip()

        if user_text and answer_text:
            examples.append((user_text, answer_text))

    unique = []
    seen = set()

    for user_text, answer_text in examples:
        key = (user_text, answer_text)

        if key not in seen:
            seen.add(key)
            unique.append(key)

    if not unique:
        raise ValueError(
            f"No valid instruction examples found in {path}"
        )

    return unique


def make_example_ids(tokenizer, user_text, answer_text):
    bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]

    prefix = f"User: {user_text}\nAssistant:"
    answer = f" {answer_text}"

    prefix_ids = tokenizer.encode(prefix)
    answer_ids = tokenizer.encode(answer)

    tokens = (
        [bos_id]
        + prefix_ids
        + answer_ids
        + [eos_id]
    )

    loss_mask = (
        [0] * (1 + len(prefix_ids))
        + [1] * (len(answer_ids) + 1)
    )

    if len(tokens) > BLOCK_SIZE + 1:
        tokens = tokens[:BLOCK_SIZE + 1]
        loss_mask = loss_mask[:BLOCK_SIZE + 1]

    if sum(loss_mask[1:]) == 0:
        return None

    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(loss_mask, dtype=torch.float32),
    )


def split_examples(examples):
    examples = list(examples)
    random.shuffle(examples)

    if len(examples) < 2:
        return examples, examples

    split = max(
        1,
        int(len(examples) * TRAIN_RATIO),
    )

    split = min(
        split,
        len(examples) - 1,
    )

    return (
        examples[:split],
        examples[split:],
    )


def encoded_examples(examples, tokenizer):
    usable = []

    for user_text, answer_text in examples:
        item = make_example_ids(
            tokenizer,
            user_text,
            answer_text,
        )

        if item is not None:
            usable.append(item)

    if not usable:
        raise ValueError(
            "No usable examples after tokenization."
        )

    return usable


def collate(batch, pad_id):
    max_len = min(
        BLOCK_SIZE + 1,
        max(len(item[0]) for item in batch),
    )

    xs = []
    ys = []
    masks = []

    for tokens, mask in batch:
        tokens = tokens[:max_len]
        mask = mask[:max_len]

        if len(tokens) < max_len:
            pad = max_len - len(tokens)

            tokens = torch.cat(
                [
                    tokens,
                    torch.full(
                        (pad,),
                        pad_id,
                        dtype=torch.long,
                    ),
                ]
            )

            mask = torch.cat(
                [
                    mask,
                    torch.zeros(
                        pad,
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


def make_batch_stream(
    encoded,
    batch_size,
    pad_id,
    device,
):
    while True:
        order = list(range(len(encoded)))
        random.shuffle(order)

        for start in range(
            0,
            len(order),
            batch_size,
        ):
            indices = order[
                start:start + batch_size
            ]

            batch = [
                encoded[index]
                for index in indices
            ]

            x, y, mask = collate(
                batch,
                pad_id,
            )

            yield (
                x.to(device),
                y.to(device),
                mask.to(device),
            )


def masked_loss(model, x, y, mask):
    logits, _ = model(x)

    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).reshape_as(mask)

    denom = mask.sum().clamp_min(1.0)

    return (
        (per_token * mask).sum()
        / denom
    )


@torch.no_grad()
def evaluate(
    model,
    encoded,
    pad_id,
    device,
):
    model.eval()

    losses = []

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
            pad_id,
        )

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        losses.append(
            masked_loss(
                model,
                x,
                y,
                mask,
            ).item()
        )

    model.train()

    return (
        sum(losses)
        / max(1, len(losses))
    )


def main():
    seed_everything(SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 56)
    print("OWN AI v6 — TRUE INSTRUCTION SFT")
    print("=" * 56)

    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    for required in (
        BASE_CHECKPOINT,
        BASE_TOKENIZER,
        INSTRUCTION_FILE,
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing required file: {required}"
            )

    tokenizer = OwnTokenizer(
        vocab_size=512,
    )

    tokenizer.load(BASE_TOKENIZER)

    examples = parse_instruction_file(
        INSTRUCTION_FILE
    )

    train_examples, val_examples = split_examples(
        examples
    )

    train_encoded = encoded_examples(
        train_examples,
        tokenizer,
    )

    val_encoded = encoded_examples(
        val_examples,
        tokenizer,
    )

    print(
        "Instruction pairs:",
        len(examples),
    )

    print(
        "Train pairs:",
        len(train_encoded),
    )

    print(
        "Validation pairs:",
        len(val_encoded),
    )

    print(
        "Vocabulary:",
        tokenizer.vocab_size,
    )

    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model = OwnAI(
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

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    print(
        "Starting from:",
        BASE_CHECKPOINT,
    )

    print(
        "Loss target:",
        "assistant response only",
    )

    print("-" * 56)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_STEPS,
        )
    )

    pad_id = tokenizer.token_to_id[
        tokenizer.pad_token
    ]

    stream = make_batch_stream(
        train_encoded,
        BATCH_SIZE,
        pad_id,
        device,
    )

    best_val = float("inf")
    bad_evals = 0

    OUT_CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for step in range(MAX_STEPS):
        model.train()

        x, y, mask = next(stream)

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = masked_loss(
            model,
            x,
            y,
            mask,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()
        scheduler.step()

        if (
            step % EVAL_INTERVAL == 0
            or step == MAX_STEPS - 1
        ):
            train_loss = loss.item()

            val_loss = evaluate(
                model,
                val_encoded,
                pad_id,
                device,
            )

            print(
                f"Step {step:04d} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.7f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                bad_evals = 0

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocab_size": checkpoint["vocab_size"],
                        "block_size": checkpoint["block_size"],
                        "d_model": checkpoint["d_model"],
                        "n_heads": checkpoint["n_heads"],
                        "n_layers": checkpoint["n_layers"],
                        "dropout": checkpoint.get(
                            "dropout",
                            0.1,
                        ),
                        "stage": "instruction_sft_v6",
                        "base_checkpoint": str(
                            BASE_CHECKPOINT
                        ),
                        "best_val_loss": best_val,
                        "instruction_pairs": len(
                            examples
                        ),
                        "masking": (
                            "assistant_response_only"
                        ),
                    },
                    OUT_CHECKPOINT,
                )

                tokenizer.save(
                    OUT_TOKENIZER
                )

                print(
                    "  Saved new v6 SFT checkpoint."
                )

            else:
                bad_evals += 1

                print(
                    f"  No improvement "
                    f"({bad_evals}/{PATIENCE})"
                )

                if bad_evals >= PATIENCE:
                    print(
                        "  Early stopping."
                    )
                    break

    print("-" * 56)
    print("v6 SFT finished.")
    print(
        "Best validation loss:",
        f"{best_val:.4f}",
    )

    print(
        "Model:",
        OUT_CHECKPOINT,
    )

    print(
        "Tokenizer:",
        OUT_TOKENIZER,
    )

    print("=" * 56)


if __name__ == "__main__":
    main()
