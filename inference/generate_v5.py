from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import OwnTokenizer
from model.own_ai import OwnAI


CHECKPOINT = ROOT / "checkpoints" / "own_ai_v5_best.pt"
TOKENIZER_FILE = ROOT / "checkpoints" / "tokenizer_v5.json"


def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=120,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1
):

    model.eval()

    bos_id = tokenizer.token_to_id[
        tokenizer.bos_token
    ]

    eos_id = tokenizer.token_to_id[
        tokenizer.eos_token
    ]

    prompt_text = (
        "User: "
        + prompt
        + "\nAssistant:"
    )

    prompt_ids = tokenizer.encode(
        prompt_text
    )

    input_ids = [
        bos_id
    ] + prompt_ids

    generated = input_ids.copy()

    for _ in range(max_new_tokens):

        context = generated[
            -model.block_size:
        ]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            logits, _ = model(x)

        logits = logits[:, -1, :]

        # Temperature
        logits = logits / temperature

        # Repetition penalty
        for token_id in set(generated):

            logits[0, token_id] /= repetition_penalty

        # Top-K
        if top_k is not None:

            values, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )

            cutoff = values[:, -1].unsqueeze(-1)

            logits = torch.where(
                logits < cutoff,
                torch.full_like(
                    logits,
                    float("-inf")
                ),
                logits
            )

        # Top-P
        if top_p is not None:

            sorted_logits, sorted_indices = torch.sort(
                logits,
                descending=True
            )

            probabilities = torch.softmax(
                sorted_logits,
                dim=-1
            )

            cumulative = torch.cumsum(
                probabilities,
                dim=-1
            )

            remove = cumulative > top_p

            remove[:, 1:] = remove[:, :-1].clone()

            remove[:, 0] = False

            sorted_logits[remove] = float("-inf")

            logits = torch.full_like(
                logits,
                float("-inf")
            )

            logits.scatter_(
                1,
                sorted_indices,
                sorted_logits
            )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        generated.append(next_token)

        if next_token == eos_id:
            break

    # Only decode newly generated tokens
    new_tokens = generated[
        len(input_ids):
    ]

    output = tokenizer.decode(
        new_tokens
    )

    # Clean common formatting artifacts
    if "User:" in output:
        output = output.split(
            "User:",
            1
        )[0]

    if "Assistant:" in output:
        output = output.replace(
            "Assistant:",
            "",
            1
        )

    return output.strip()


def main():

    print("================================")
    print("        OWN AI v5 CHAT")
    print("================================")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("--------------------------------")

    if not CHECKPOINT.exists():

        print(
            "ERROR: Checkpoint not found:"
        )

        print(CHECKPOINT)

        return

    if not TOKENIZER_FILE.exists():

        print(
            "ERROR: Tokenizer not found:"
        )

        print(TOKENIZER_FILE)

        return

    tokenizer = OwnTokenizer(
        vocab_size=512
    )

    tokenizer.load(
        TOKENIZER_FILE
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model = OwnAI(
        vocab_size=checkpoint[
            "vocab_size"
        ],
        block_size=checkpoint[
            "block_size"
        ],
        d_model=checkpoint[
            "d_model"
        ],
        n_heads=checkpoint[
            "n_heads"
        ],
        n_layers=checkpoint[
            "n_layers"
        ],
        dropout=checkpoint.get(
            "dropout",
            0.1
        )
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameters:,}"
    )

    print(
        "Vocabulary:",
        tokenizer.vocab_size
    )

    print(
        "Context:",
        checkpoint["block_size"]
    )

    print(
        "Embedding:",
        checkpoint["d_model"]
    )

    print(
        "Heads:",
        checkpoint["n_heads"]
    )

    print(
        "Layers:",
        checkpoint["n_layers"]
    )

    print(
        "Best validation loss:",
        f'{checkpoint["best_val_loss"]:.4f}'
    )

    print("--------------------------------")
    print("Model loaded successfully.")
    print("Type 'exit' to quit.")
    print("================================")

    while True:

        try:

            prompt = input("\nYou: ")

        except KeyboardInterrupt:

            print("\nBye!")

            break

        if prompt.lower().strip() == "exit":

            print("Bye!")

            break

        if not prompt.strip():

            continue

        answer = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=120,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1
        )

        print(
            "AI:",
            answer
        )


if __name__ == "__main__":
    main()