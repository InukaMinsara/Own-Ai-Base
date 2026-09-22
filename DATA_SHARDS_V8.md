# Own AI v8 Mass-Data Shards

The large-data pipeline is designed for multi-gigabyte or multi-terabyte
corpora without loading the whole dataset into laptop RAM.

## Local data sources

By default the shard builder scans:

    data\cache
    data\raw
    data\knowledge

You can override the roots on Windows:

    set OWN_AI_V8_DATA_DIRS=D:\Own AI\data\cache;D:\Own AI\data\knowledge

Keep cloud data in the cloud and sync only the active subset into the local
cache. The shard builder works on that active subset.

## Pipeline

    source files
        ↓
    normalize
        ↓
    content-chunk hashing
        ↓
    exact duplicate removal
        ↓
    deterministic train/validation/test split
        ↓
    tokenizer
        ↓
    uint32 token shards
        ↓
    NumPy memory-mapped training
        ↓
    Own AI v8

Default shard size is 128 MB. Change it with:

    set OWN_AI_V8_SHARD_MB=256

The tokenizer is trained from a bounded sample rather than scanning the full
corpus into memory. Change the sample with:

    set OWN_AI_V8_TOKENIZER_SAMPLE_MB=512

The training loop samples directly from token-shard memory maps. Only the
current mini-batch is transferred to the GPU.

## Commands

Build shards:

    python data\build_token_shards_v8.py

Train:

    python training\train_v8_streaming.py

Or use:

    train_v8_streaming.bat

## Data quality

The split is based on deterministic chunk hashes, and exact duplicate chunks
are removed before tokenization. This is still only a first-stage data
quality filter: semantic duplicates, low-quality pages, boilerplate, and
license issues need further filtering before treating a very large corpus as
high-quality training data.

## 4.5 TB strategy

Do not mirror 4.5 TB to the laptop.

Treat the cloud storage as the source lake. Keep:

    cloud raw data
        → active local cache shard(s)
        → token shard cache
        → training
        → rotate cache

This lets a small GPU repeatedly train on new data slices without requiring
a local disk large enough to hold the entire cloud corpus.
