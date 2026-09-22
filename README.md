# Own-Ai-Base

Own AI is an original local-first AI project built from scratch with PyTorch.

## What it contains

### Current local assistant

- Decoder-only Transformer language model
- Own AI v4/v5/v6 experiments preserved
- v7 scalable Transformer path
- v8 larger Transformer path with RoPE, RMSNorm and SwiGLU
- True response-only instruction fine-tuning
- Local knowledge retrieval (RAG)
- Persistent local accounts and saved chats
- Cross-chat memory recall
- Exact local calculator/tool routing
- Optional web search without an AI API key
- File upload and local document indexing
- PDF and DOCX extraction
- Optional local image understanding
- Browser voice input and speech output
- Responsive desktop/mobile web UI
- PWA install support
- Localhost-only server by default
- Google Drive dataset-lake helper with SHA-256 manifests

## v8 brain upgrade

The main development direction is now v8.

Default model profile:

- 4096-token vocabulary
- 1024-token context
- 384-dimensional embeddings
- 8 attention heads
- 16 Transformer layers
- RoPE positional encoding
- RMSNorm
- SwiGLU
- weight tying
- gradient-checkpointing support
- roughly 30M-class parameter scale

The v8 tokenizer is a hybrid word/Unicode tokenizer. Frequent complete pieces can become tokens while Unicode characters from the training corpus are explicitly preserved for fallback.

The v8 dataset builder creates a balanced supervised dataset across:

- general explanations
- coding
- Sinhala
- math
- tool usage
- conversation behavior
- legacy instruction examples

It also builds a clean pretraining corpus from local data.

## Run the app

Update the local copy:

    cd /d "D:\Own AI"
    git pull origin main

Install v8 dependencies:

    install_v8.bat

Run the 25-step smoke test first:

    train_v8_smoke.bat

Build and train the full v8 pipeline:

    train_v8.bat

Or run the phases separately:

    python data\build_v8_dataset.py
    python training\train_v8.py
    python training\train_sft_v8.py

After v8 SFT finishes, start the app:

    start_own_ai.bat

Open:

    http://127.0.0.1:8000

Run the local benchmark:

    python tools\benchmark_v8.py

## Data lake / 4.5 TB cloud storage

Do not download 4.5 TB to the laptop.

Use the cloud space as a dataset lake and keep only the active shard(s) in the local cache. Organize cloud data into owned, open-license, code, education, Sinhala, documents, conversations, images, and other clearly labeled collections.

Local manifest:

    python cloud\google_drive_sync.py manifest data

Upload a dataset folder to Google Drive:

    set GOOGLE_DRIVE_PARENT_ID=YOUR_FOLDER_ID
    python cloud\google_drive_sync.py upload "D:\Own AI\data\to_upload"

Read:

    cloud\DATA_LAKE.md

Keep license/attribution information with every external dataset. Never put credentials, OAuth tokens, API keys, or private secrets into the training corpus.

## No public AI API

The core assistant runs the language model locally. The local web UI talks to the local Python server over localhost.

The optional web-search tool fetches public web results when explicitly requested; it is a tool, not an external AI-model API.

## Important limitations

Own AI is an original small-scale research project, not GPT, Gemini, Claude, or another frontier model.

A huge storage quota does not automatically create a strong model. The useful path is:

high-quality data -> deduplication -> correct tokenizer -> pretraining -> instruction SFT -> evaluation -> retrieval/tools -> iteration.

Millions of duplicated or templated examples are not equivalent to a high-quality corpus, and frontier-level capabilities require vastly more compute, data, engineering, and evaluation.

The project is designed to grow step by step while keeping the core architecture local-first.
