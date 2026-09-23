from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer_v8 import OwnTokenizerV8
from model.own_ai_v8 import OwnAIv8


CHECKPOINT = ROOT / "checkpoints" / "own_ai_v8_streaming_best.pt"
TOKENIZER = ROOT / "checkpoints" / "tokenizer_v8_streaming.json"


PROMPTS = [
    "hi",
    "What is a transformer?",
    "Explain Python simply.",
    "What is artificial intelligence?",
    "සිංහලෙන් කෘතිම බුද්ධිය කියන්නේ මොකක්ද?",
    "සිංහලෙන් ගණිතය ගැන කෙටි විස්තරයක් දෙන්න.",
]


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=80):
    ids = [
        tokenizer.bos_id,
        *tokenizer.encode(prompt),
    ]
    ids = ids[-(model.block_size - 1):]

    generated = ids[:]

    for _ in range(max_new_tokens):
        x = torch.tensor(
            [generated[-model.block_size:]],
            dtype=torch.long,
            device=device,
        )

        logits, _ = model(x)
        next_logits = logits[:, -1, :] / 0.8

        probabilities = torch.softmax(
            next_logits,
            dim=-1,
        )
        next_id = torch.multinomial(
            probabilities,
            1,
        ).item()

        generated.append(next_id)

        if next_id == tokenizer.eos_id:
            break

    answer = tokenizer.decode(
        generated[len(ids):]
    ).strip()

    return answer


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if not TOKENIZER.exists():
        raise FileNotFoundError(TOKENIZER)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    tokenizer = OwnTokenizerV8(
        vocab_size=checkpoint["vocab_size"]
    )
    tokenizer.load(TOKENIZER)

    model = OwnAIv8(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint.get("dropout", 0.1),
        gradient_checkpointing=False,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    model.eval()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("=" * 68)
    print("OWN AI v8 STREAMING CHECKPOINT BENCHMARK")
    print("=" * 68)
    print("Checkpoint:", CHECKPOINT)
    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Parameters:", f"{params:,}")
    print("Vocabulary:", tokenizer.vocab_size)
    print("Context:", model.block_size)
    print("=" * 68)

    for prompt in PROMPTS:
        print()
        print("USER:", prompt)
        try:
            answer = generate(
                model,
                tokenizer,
                prompt,
                device,
            )
            print("OWN AI:", answer)
        except Exception as exc:
            print("ERROR:", exc)

    print()
    print("=" * 68)
    print("Benchmark finished.")
    print("=" * 68)


if __name__ == "__main__":
    main()
