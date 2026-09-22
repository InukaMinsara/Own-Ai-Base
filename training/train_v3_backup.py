from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import OwnTokenizer
from model.own_ai import OwnAI


DATA_FILE = ROOT / "data" / "processed" / "train.txt"
CHECKPOINT_DIR = ROOT / "checkpoints"

BLOCK_SIZE = 64
BATCH_SIZE = 16

MAX_STEPS = 1500

LEARNING_RATE = 0.0005

TRAIN_RATIO = 0.9

EVAL_INTERVAL = 100

PATIENCE = 3


def get_batch(data, batch_size, block_size, device):

    max_start = len(data) - block_size - 1

    positions = torch.randint(
        0,
        max_start,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in positions
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in positions
    ])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    device
):

    model.eval()

    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = []

        for _ in range(20):

            x, y = get_batch(
                data,
                BATCH_SIZE,
                BLOCK_SIZE,
                device
            )

            _, loss = model(x, y)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("================================")
    print("       OWN AI TRAINING v3")
    print("================================")
    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    text = DATA_FILE.read_text(
        encoding="utf-8"
    )

    tokenizer = OwnTokenizer(
        vocab_size=512
    )

    tokenizer.train(text)

    tokens = torch.tensor(
        tokenizer.encode(
            text,
            add_special_tokens=True
        ),
        dtype=torch.long
    )

    split = int(
        len(tokens) * TRAIN_RATIO
    )

    train_data = tokens[:split]
    val_data = tokens[split:]

    print("Vocabulary:", tokenizer.vocab_size)
    print("Total tokens:", len(tokens))
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))
    print("Block size:", BLOCK_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Maximum steps:", MAX_STEPS)
    print("Learning rate:", LEARNING_RATE)
    print("================================")

    model = OwnAI(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        d_model=128,
        n_heads=4,
        n_layers=4
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_STEPS
    )

    best_val_loss = float("inf")
    bad_evaluations = 0

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.train()

    for step in range(MAX_STEPS):

        x, y = get_batch(
            train_data,
            BATCH_SIZE,
            BLOCK_SIZE,
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        _, loss = model(x, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()
        scheduler.step()

        if step % EVAL_INTERVAL == 0:

            losses = estimate_loss(
                model,
                train_data,
                val_data,
                device
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Step {step:04d} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss
                bad_evaluations = 0

                checkpoint_path = (
                    CHECKPOINT_DIR
                    / "own_ai_best.pt"
                )

                torch.save(
                    {
                        "model_state":
                            model.state_dict(),

                        "vocab_size":
                            tokenizer.vocab_size,

                        "block_size":
                            BLOCK_SIZE,

                        "d_model":
                            128,

                        "n_heads":
                            4,

                        "n_layers":
                            4,

                        "best_val_loss":
                            best_val_loss
                    },
                    checkpoint_path
                )

                tokenizer.save(
                    CHECKPOINT_DIR
                    / "tokenizer_best.json"
                )

                print(
                    "  ✓ New best checkpoint saved"
                )

            else:

                bad_evaluations += 1

                print(
                    f"  No improvement "
                    f"({bad_evaluations}/{PATIENCE})"
                )

                if bad_evaluations >= PATIENCE:

                    print(
                        "  Early stopping triggered."
                    )

                    break

    print("================================")
    print("TRAINING FINISHED")
    print("================================")
    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )
    print(
        "Best model:",
        CHECKPOINT_DIR / "own_ai_best.pt"
    )
    print(
        "Best tokenizer:",
        CHECKPOINT_DIR / "tokenizer_best.json"
    )
    print("================================")


if __name__ == "__main__":
    main()