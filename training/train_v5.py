from pathlib import Path
import sys
import random

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import OwnTokenizer
from model.own_ai import OwnAI


# ============================================================
# PATHS
# ============================================================

DATA_FILE = ROOT / "data" / "processed" / "train_v5.txt"

CHECKPOINT_DIR = ROOT / "checkpoints"

MODEL_CHECKPOINT = (
    CHECKPOINT_DIR / "own_ai_v5_best.pt"
)

TOKENIZER_FILE = (
    CHECKPOINT_DIR / "tokenizer_v5.json"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

BLOCK_SIZE = 256

BATCH_SIZE = 4

MAX_STEPS = 2000

LEARNING_RATE = 0.0001

TRAIN_RATIO = 0.9

EVAL_INTERVAL = 100

PATIENCE = 4

VOCAB_SIZE = 512


# ============================================================
# MODEL SETTINGS
# ============================================================

D_MODEL = 256

N_HEADS = 8

N_LAYERS = 8

DROPOUT = 0.1


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# LOAD INSTRUCTION EXAMPLES
# ============================================================

def load_examples(path):

    text = path.read_text(
        encoding="utf-8"
    )

    raw_examples = text.split(
        "<BOS>"
    )

    examples = []

    for example in raw_examples:

        example = example.strip()

        if not example:
            continue

        if "<EOS>" in example:

            example = example.split(
                "<EOS>",
                1
            )[0]

        example = example.strip()

        if example:
            examples.append(
                example
            )

    return examples


# ============================================================
# TOKENIZE EXAMPLES
# ============================================================

def encode_examples(
    examples,
    tokenizer
):

    all_tokens = []

    for example in examples:

        # Every example gets real BOS/EOS IDs.
        tokens = []

        tokens.append(
            tokenizer.token_to_id[
                tokenizer.bos_token
            ]
        )

        tokens.extend(
            tokenizer.encode(
                example
            )
        )

        tokens.append(
            tokenizer.token_to_id[
                tokenizer.eos_token
            ]
        )

        all_tokens.extend(
            tokens
        )

    return torch.tensor(
        all_tokens,
        dtype=torch.long
    )


# ============================================================
# BATCH
# ============================================================

def get_batch(
    data,
    batch_size,
    block_size,
    device
):

    max_start = (
        len(data)
        - block_size
        - 1
    )

    if max_start <= 0:

        raise ValueError(
            "Dataset is smaller than BLOCK_SIZE."
        )

    positions = torch.randint(
        0,
        max_start,
        (batch_size,)
    )

    x = torch.stack(
        [
            data[
                i:i + block_size
            ]
            for i in positions
        ]
    )

    y = torch.stack(
        [
            data[
                i + 1:i + block_size + 1
            ]
            for i in positions
        ]
    )

    return (
        x.to(device),
        y.to(device)
    )


# ============================================================
# VALIDATION
# ============================================================

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

            _, loss = model(
                x,
                y
            )

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "================================"
    )

    print(
        "       OWN AI TRAINING v5"
    )

    print(
        "================================"
    )

    print(
        "Device:",
        device
    )

    if device == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load examples
    # --------------------------------------------------------

    examples = load_examples(
        DATA_FILE
    )

    print(
        "Instruction examples:",
        len(examples)
    )

    # --------------------------------------------------------
    # Load complete dataset text
    # --------------------------------------------------------

    text = DATA_FILE.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Train tokenizer
    # --------------------------------------------------------

    tokenizer = OwnTokenizer(
        vocab_size=VOCAB_SIZE
    )

    tokenizer.train(text)

    print(
        "Vocabulary:",
        tokenizer.vocab_size
    )

    print(
        "BOS ID:",
        tokenizer.token_to_id[
            tokenizer.bos_token
        ]
    )

    print(
        "EOS ID:",
        tokenizer.token_to_id[
            tokenizer.eos_token
        ]
    )

    # --------------------------------------------------------
    # Encode each instruction example
    # --------------------------------------------------------

    tokens = encode_examples(
        examples,
        tokenizer
    )

    print(
        "Total tokens:",
        len(tokens)
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    split = int(
        len(tokens)
        * TRAIN_RATIO
    )

    train_data = tokens[
        :split
    ]

    val_data = tokens[
        split:
    ]

    print(
        "Training tokens:",
        len(train_data)
    )

    print(
        "Validation tokens:",
        len(val_data)
    )

    print(
        "Block size:",
        BLOCK_SIZE
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Maximum steps:",
        MAX_STEPS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "================================"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = OwnAI(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_STEPS
        )
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    bad_evaluations = 0

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.train()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for step in range(
        MAX_STEPS
    ):

        x, y = get_batch(
            train_data,
            BATCH_SIZE,
            BLOCK_SIZE,
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        _, loss = model(
            x,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        scheduler.step()

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if (
            step % EVAL_INTERVAL
            == 0
        ):

            losses = estimate_loss(
                model,
                train_data,
                val_data,
                device
            )

            train_loss = (
                losses["train"]
            )

            val_loss = (
                losses["val"]
            )

            current_lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            print(
                f"Step {step:04d} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # ------------------------------------------------
            # New best
            # ------------------------------------------------

            if (
                val_loss
                < best_val_loss
            ):

                best_val_loss = (
                    val_loss
                )

                bad_evaluations = 0

                torch.save(
                    {
                        "model_state":
                            model.state_dict(),

                        "vocab_size":
                            tokenizer.vocab_size,

                        "block_size":
                            BLOCK_SIZE,

                        "d_model":
                            D_MODEL,

                        "n_heads":
                            N_HEADS,

                        "n_layers":
                            N_LAYERS,

                        "dropout":
                            DROPOUT,

                        "best_val_loss":
                            best_val_loss
                    },
                    MODEL_CHECKPOINT
                )

                tokenizer.save(
                    TOKENIZER_FILE
                )

                print(
                    "  New best v5 checkpoint saved"
                )

            else:

                bad_evaluations += 1

                print(
                    f"  No improvement "
                    f"({bad_evaluations}/"
                    f"{PATIENCE})"
                )

                if (
                    bad_evaluations
                    >= PATIENCE
                ):

                    print(
                        "  Early stopping triggered."
                    )

                    break

    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "================================"
    )

    print(
        "       TRAINING FINISHED"
    )

    print(
        "================================"
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )

    print(
        "Best model:",
        MODEL_CHECKPOINT
    )

    print(
        "Best tokenizer:",
        TOKENIZER_FILE
    )

    print(
        "================================"
    )


if __name__ == "__main__":

    main()