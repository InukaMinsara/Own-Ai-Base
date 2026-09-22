# Own-Ai-Base

Own AI is an original local-first AI project built from scratch with PyTorch.

## What it contains

### Current local assistant

- Decoder-only Transformer language model
- Own AI v4/v5/v6 experiments preserved
- True response-only instruction SFT in v6
- Local knowledge retrieval (RAG)
- Persistent accounts and saved chats
- Cross-chat memory recall
- Exact local calculator/tool routing
- Optional web search without an API key
- File upload and local document indexing
- PDF and DOCX text extraction
- Optional local image understanding with BLIP
- Browser voice input and speech output
- Responsive desktop/mobile web UI
- PWA install support
- Localhost-only server by default

### v7 model path

The repository also contains a larger scalable Transformer:

- 512-token context
- 384-dimensional embeddings
- 8 attention heads
- 12 Transformer layers
- RMSNorm
- SwiGLU feed-forward blocks
- approximately 20M parameters
- Unicode-aware ranked BPE tokenizer
- mixed-precision training support

The v7 model is a separate architecture and therefore requires v7 pretraining before v7 SFT.

## Run the current Own AI app

Update the local copy:

    cd /d "D:Own AI"
    git pull origin main

Install optional document/vision dependencies:

    install_v7.bat

Start the local web app:

    start_own_ai.bat

Open:

    http://127.0.0.1:8000

Create a local account, then use the workspace.

## Build a larger v7 model

Generate thousands of structured instruction examples:

    .venv\Scripts\activate
    python data\build_instruction_dataset_v7.py

Build a cleaned training corpus:

    python data\build_corpus_v7.py

Pretrain the larger Transformer:

    python training\train_v7.py

Then instruction-tune it:

    python training\train_sft_v7.py

Convenience launchers are also provided:

    train_v7.bat
    train_sft_v7.bat

The v7 training scripts accept environment variables for model size, context, batch size, gradient accumulation, learning rate and training length.

## Data and knowledge

Put durable local knowledge into:

    data\knowledge\

User-uploaded documents are stored under:

    data\uploads\<username>\

The RAG system indexes raw knowledge and uploaded text, not generated training wrappers or backup datasets.

## No public API

This project does not expose a public cloud API. The browser communicates with the local Python server over localhost because a browser needs a transport layer to talk to the local model.

## Important limitations

This is an original small AI project, not GPT, Gemini or Claude.

A larger dataset improves coverage only when the data is diverse and high quality. Millions of duplicated or templated examples are not equivalent to a high-quality training corpus.

For frontier-level capability, much larger model capacity, much more training data, longer context, better evaluation, and substantial compute are required.

The project is designed to grow toward that direction without replacing the local-first architecture.
