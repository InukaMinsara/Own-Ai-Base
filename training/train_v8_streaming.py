from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.build_token_shards_v8 import OUT_ROOT, main as build_shards
from tokenizer.tokenizer_v8 import OwnTokenizerV8
from model.own_ai_v8 import OwnAIv8


MODEL_FILE = ROOT / "checkpoints" / "own_ai_v8_streaming_best.pt"
TOKENIZER_FILE = ROOT / "checkpoints" / "tokenizer_v8_streaming.json"

VOCAB_SIZE = int(os.getenv("OWN_AI_V8_VOCAB", "4096"))
BLOCK_SIZE = int(os.getenv("OWN_AI_V8_CONTEXT", "1024"))
D_MODEL = int(os.getenv("OWN_AI_V8_DIM", "384"))
N_HEADS = int(os.getenv("OWN_AI_V8_HEADS", "8"))
N_LAYERS = int(os.getenv("OWN_AI_V8_LAYERS", "16"))
BATCH_SIZE = int(os.getenv("OWN_AI_V8_BATCH", "1"))
GRAD_ACCUM = int(os.getenv("OWN_AI_V8_ACCUM", "8"))
MAX_STEPS = int(os.getenv("OWN_AI_V8_STEPS", "5000"))
LR = float(os.getenv("OWN_AI_V8_LR", "0.00025"))
WEIGHT_DECAY = float(
    os.getenv(
        "OWN_AI_V8_WEIGHT_DECAY",
        "0.05",
    )
)
EVAL_INTERVAL = int(
    os.getenv(
        "OWN_AI_V8_EVAL_INTERVAL",
        "100",
    )
)
EVAL_BATCHES = int(
    os.getenv(
        "OWN_AI_V8_EVAL_BATCHES",
        "16",
    )
)
SEED = 42
USE_CHECKPOINTING = os.getenv(
    "OWN_AI_V8_CHECKPOINTING",
    "1",
).lower() not in {
    "0",
    "false",
    "no",
}


def read_summary():
    path = OUT_ROOT / "dataset_summary.json"
    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None


def load_shards(split):
    shards = []

    for path in sorted(
        (OUT_ROOT / split).glob("shard-*.bin")
    ):
        meta_path = path.with_suffix(
            ".json"
        )

        try:
            metadata = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )
            tokens = int(
                metadata["tokens"]
            )
        except (
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ):
            continue

        if tokens >= BLOCK_SIZE + 1:
            shards.append(
                {
                    "path": path,
                    "tokens": tokens,
                }
            )

    if not shards:
        raise ValueError(
            f"No usable {split} token shards found."
        )

    return shards


class ShardStore:
    def __init__(self, split):
        self.shards = load_shards(split)
        self.mmaps = [
            np.memmap(
                item["path"],
                dtype=np.uint32,
                mode="r",
            )
            for item in self.shards
        ]

        self.weights = np.asarray(
            [
                len(mm)
                for mm in self.mmaps
            ],
            dtype=np.float64,
        )
        self.weights /= self.weights.sum()

    def sample_batch(
        self,
        batch_size,
        block_size,
        device,
    ):
        x_values = []
        y_values = []

        for _ in range(batch_size):
            index = int(
                np.random.choice(
                    len(self.mmaps),
                    p=self.weights,
                )
            )
            data = self.mmaps[index]

            start = random.randint(
                0,
                len(data) - block_size - 1,
            )

            x_values.append(
                np.asarray(
                    data[
                        start:start + block_size
                    ],
                    dtype=np.int64,
                )
            )
            y_values.append(
                np.asarray(
                    data[
                        start + 1:
                        start + block_size + 1
                    ],
                    dtype=np.int64,
                )
            )

        x = torch.from_numpy(
            np.stack(x_values)
        ).to(
            device=device,
            dtype=torch.long,
        )
        y = torch.from_numpy(
            np.stack(y_values)
        ).to(
            device=device,
            dtype=torch.long,
        )

        return x, y

    def close(self):
        self.mmaps.clear()


@torch.no_grad()
def evaluate(
    model,
    store,
    device,
):
    model.eval()
    values = []

    for _ in range(EVAL_BATCHES):
        x, y = store.sample_batch(
            BATCH_SIZE,
            BLOCK_SIZE,
            device,
        )

        _, loss = model(
            x,
            y,
        )
        values.append(
            loss.item()
        )

    model.train()
    return sum(values) / max(
        1,
        len(values),
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    summary = read_summary()

    if summary is None:
        build_shards()
    else:
        print(
            "Using existing token-shard dataset:",
            OUT_ROOT,
        )

    tokenizer_path = (
        OUT_ROOT / "tokenizer_v8.json"
    )

    tokenizer = OwnTokenizerV8(
        vocab_size=VOCAB_SIZE
    )
    tokenizer.load(
        tokenizer_path
    )

    train_store = ShardStore("train")
    val_store = ShardStore("val")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    print("=" * 68)
    print("OWN AI v8 STREAMING PRETRAINING")
    print("=" * 68)
    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Vocabulary:", tokenizer.vocab_size)
    print("Model:", f"{D_MODEL}d / {N_HEADS} heads / {N_LAYERS} layers")
    print("Context:", BLOCK_SIZE)
    print("Parameters:", f"{params:,}")
    print("Train shards:", len(train_store.shards))
    print("Val shards:", len(val_store.shards))
    print("Batch:", BATCH_SIZE, "x", GRAD_ACCUM)
    print("Steps:", MAX_STEPS)
    print("=" * 68)

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

    for step in range(
        MAX_STEPS
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        running = 0.0

        for _ in range(
            GRAD_ACCUM
        ):
            x, y = train_store.sample_batch(
                BATCH_SIZE,
                BLOCK_SIZE,
                device,
            )

            with torch.amp.autocast(
                "cuda",
                enabled=device == "cuda",
            ):
                _, loss = model(
                    x,
                    y,
                )
                scaled = loss / GRAD_ACCUM

            scaler.scale(
                scaled
            ).backward()

            running += loss.item()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        scaler.step(
            optimizer
        )
        scaler.update()

        if (
            step % EVAL_INTERVAL == 0
            or step == MAX_STEPS - 1
        ):
            val = evaluate(
                model,
                val_store,
                device,
            )

            print(
                f"Step {step:05d} | "
                f"Train: {running / max(1, GRAD_ACCUM):.4f} | "
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
                        "stage": "v8_streaming_pretrain",
                        "best_val_loss": best,
                    },
                    MODEL_FILE,
                )

                tokenizer.save(
                    TOKENIZER_FILE
                )

                print(
                    "  Saved best streaming checkpoint."
                )

    train_store.close()
    val_store.close()

    print("=" * 68)
    print("STREAMING PRETRAINING FINISHED")
    print("Best validation loss:", f"{best:.4f}")
    print("Model:", MODEL_FILE)
    print("=" * 68)


if __name__ == "__main__":
    main()
