from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import OwnTokenizer
from model.own_ai import OwnAI


CHECKPOINT = ROOT / "checkpoints" / "own_ai_best.pt"
TOKENIZER_FILE = ROOT / "checkpoints" / "tokenizer_best.json"


# ============================================================
# GENERATION
# ============================================================

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1
):
    model.eval()

    ids = tokenizer.encode(prompt)

    if not ids:
        return prompt

    device = next(model.parameters()).device

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only the latest context
            context = input_ids[:, -model.block_size:]

            # Forward pass
            logits, _ = model(context)

            # Last-token logits
            logits = logits[:, -1, :]

            # ------------------------------------------------
            # Repetition penalty
            # ------------------------------------------------

            if repetition_penalty != 1.0:

                previous_tokens = input_ids[0].unique()

                for token_id in previous_tokens:

                    token_id = token_id.item()

                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # ------------------------------------------------
            # Temperature
            # ------------------------------------------------

            logits = logits / max(
                temperature,
                1e-5
            )

            # ------------------------------------------------
            # Top-K sampling
            # ------------------------------------------------

            if top_k is not None and top_k > 0:

                k = min(
                    top_k,
                    logits.size(-1)
                )

                values, _ = torch.topk(
                    logits,
                    k
                )

                minimum = values[:, [-1]]

                logits = torch.where(
                    logits < minimum,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            # ------------------------------------------------
            # Top-P / Nucleus sampling
            # ------------------------------------------------

            if top_p is not None and top_p < 1.0:

                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True
                )

                sorted_probs = F.softmax(
                    sorted_logits,
                    dim=-1
                )

                cumulative_probs = torch.cumsum(
                    sorted_probs,
                    dim=-1
                )

                remove_tokens = (
                    cumulative_probs > top_p
                )

                # Keep at least the most probable token
                remove_tokens[:, 1:] = (
                    remove_tokens[:, :-1].clone()
                )

                remove_tokens[:, 0] = False

                sorted_logits[
                    remove_tokens
                ] = float("-inf")

                logits = torch.zeros_like(
                    logits
                ).scatter(
                    1,
                    sorted_indices,
                    sorted_logits
                )

            # ------------------------------------------------
            # Convert logits to probabilities
            # ------------------------------------------------

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Safety check
            if torch.isnan(probabilities).any():
                break

            # ------------------------------------------------
            # Sample next token
            # ------------------------------------------------

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

            # ------------------------------------------------
            # EOS stop
            # ------------------------------------------------

            if hasattr(
                tokenizer,
                "eos_token_id"
            ):

                if (
                    next_token.item()
                    == tokenizer.eos_token_id
                ):
                    break

    # Decode complete sequence
    result = tokenizer.decode(
        input_ids[0].tolist()
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = OwnTokenizer()

    tokenizer.load(
        TOKENIZER_FILE
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Own AI v4 architecture
    #
    # The checkpoint metadata from the old training script
    # may contain old v3 architecture values.
    #
    # Therefore we explicitly load the v4 architecture here.
    # --------------------------------------------------------

    model = OwnAI(
        vocab_size=checkpoint["vocab_size"],
        block_size=256,
        d_model=256,
        n_heads=8,
        n_layers=8,
        dropout=0.1
    ).to(device)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    # --------------------------------------------------------
    # Startup information
    # --------------------------------------------------------

    print()
    print("================================")
    print("          OWN AI v4 CHAT")
    print("================================")

    print(
        "Device:",
        device
    )

    if device == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Parameters:",
        f"{parameters:,}"
    )

    print(
        "Vocabulary:",
        checkpoint["vocab_size"]
    )

    print(
        "Context:",
        model.block_size
    )

    print(
        "Embedding:",
        model.d_model
    )

    print(
        "Heads:",
        8
    )

    print(
        "Layers:",
        8
    )

    print(
        "Best validation loss:",
        f"{checkpoint['best_val_loss']:.4f}"
    )

    print("--------------------------------")
    print("Model loaded successfully.")
    print("Type 'exit' to quit.")
    print("================================")

    # --------------------------------------------------------
    # Chat loop
    # --------------------------------------------------------

    while True:

        try:

            prompt = input(
                "\nYou: "
            )

        except KeyboardInterrupt:

            print(
                "\nAI: Goodbye!"
            )

            break

        except EOFError:

            print(
                "\nAI: Goodbye!"
            )

            break

        # Exit
        if prompt.lower().strip() == "exit":

            print(
                "AI: Goodbye!"
            )

            break

        # Empty input
        if not prompt.strip():
            continue

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1
        )

        print(
            "AI:",
            response
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()