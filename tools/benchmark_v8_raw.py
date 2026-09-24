from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.own_ai_v8 import OwnAIv8
from tokenizer.tokenizer_v8 import OwnTokenizerV8


DEFAULT_CHECKPOINT = (
    ROOT / "checkpoints" / "own_ai_v8_streaming_best.pt"
)
DEFAULT_TOKENIZER = (
    ROOT / "data" / "processed" / "token_shards_v8"
    / "tokenizer_v8.json"
)

PROMPTS = [
    "Hello, who are you?",
    "Explain what a computer program is in simple words.",
    "සිංහලෙන් ශ්‍රී ලංකාව ගැන වාක්‍ය දෙකක් ලියන්න.",
    "What is the difference between RAM and storage?",
    "def add(a, b):",
    "2 + 2 =",
]


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=80,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
):
    ids = tokenizer.encode(
        prompt,
        add_special_tokens=True,
    )

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.block_size:]
        logits, _ = model(context)
        logits = logits[:, -1, :]
        logits = logits / max(temperature, 1e-5)

        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            cutoff = values[:, [-1]]
            logits = torch.where(
                logits < cutoff,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        if top_p and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                logits,
                descending=True,
            )
            sorted_probs = F.softmax(
                sorted_logits,
                dim=-1,
            )
            cumulative = torch.cumsum(
                sorted_probs,
                dim=-1,
            )
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits[remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter(
                1,
                sorted_indices,
                sorted_logits,
            )

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

        if next_token.item() == tokenizer.eos_id:
            break

    return tokenizer.decode(
        input_ids[0].tolist()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Raw V8 checkpoint benchmark."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
    )
    parser.add_argument(
        "--tokenizer",
        default=str(DEFAULT_TOKENIZER),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    tokenizer = OwnTokenizerV8(
        vocab_size=int(checkpoint["vocab_size"])
    )
    tokenizer.load(tokenizer_path)

    model = OwnAIv8(
        vocab_size=tokenizer.vocab_size,
        block_size=int(checkpoint["block_size"]),
        d_model=int(checkpoint["d_model"]),
        n_heads=int(checkpoint["n_heads"]),
        n_layers=int(checkpoint["n_layers"]),
        dropout=float(checkpoint.get("dropout", 0.1)),
        gradient_checkpointing=False,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.eval()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("=" * 68)
    print("OWN AI v8 RAW MODEL BENCHMARK")
    print("=" * 68)
    print("Checkpoint:", checkpoint_path)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Parameters:", f"{params:,}")
    print("Vocabulary:", tokenizer.vocab_size)
    print("Context:", model.block_size)
    print(
        "Best validation loss:",
        f"{float(checkpoint.get('best_val_loss', 0.0)):.4f}",
    )
    print("=" * 68)

    for index, prompt in enumerate(PROMPTS, 1):
        output = generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print()
        print(f"[{index}] PROMPT")
        print(prompt)
        print("RESPONSE")
        print(output)

    print()
    print("=" * 68)
    print("RAW MODEL BENCHMARK FINISHED")
    print("=" * 68)


if __name__ == "__main__":
    main()
