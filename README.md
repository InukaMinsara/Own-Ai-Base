# Own-Ai-Base

A from-scratch small Transformer language model project built with PyTorch.

## Model

- Decoder-only causal Transformer
- 8 layers
- 8 attention heads
- 256-dimensional embeddings
- 256-token context window
- 512-token vocabulary
- approximately 6.5M parameters
- weight-tied language-model head
- CUDA support through PyTorch

## Own AI v6

v6 adds the pieces that make the model usable as a local assistant:

1. Base language-model pretraining
2. True instruction SFT with assistant-response-only loss masking
3. Local retrieval over project data
4. Conversation memory
5. Browser chat interface

### Train v6

From the project root:

    .venv\Scripts\activate
    python data\build_instructions.py
    python training\train_sft_v6.py

The SFT stage starts from:

    checkpoints\own_ai_best.pt

and creates:

    checkpoints\own_ai_v6_sft_best.pt
    checkpoints\tokenizer_v6.json

The existing v5 experiment is intentionally preserved.

### Run the browser chat

    .venv\Scripts\activate
    python chat\server.py

Then open:

    http://127.0.0.1:8000

The server automatically uses the v6 SFT checkpoint when it exists. Otherwise it falls back to the base checkpoint.

## RAG and memory

The model's learned knowledge comes from training data stored in its weights.

The RAG layer is separate. It searches local .txt and .md files under data at runtime and places relevant text into the prompt. Updating those files does not require retraining the model.

Conversation memory is also runtime state. It keeps recent turns for the current server session.

## Important limitation

This is an original small model, not a copy of GPT, Gemini, or Claude. Its current 6.5M-parameter architecture and small instruction dataset are far below frontier-model scale.

The project is structured so the important parts can keep growing:

data -> tokenizer -> pretraining -> instruction tuning -> retrieval -> memory -> interface

For stronger quality, the next major upgrade is a much larger, clean, legally usable dataset plus a larger model architecture and longer context window.
