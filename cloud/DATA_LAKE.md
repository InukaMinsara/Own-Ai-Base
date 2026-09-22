# Own AI v8 Data Lake

Use the 4.5 TB cloud space as a dataset lake, not as one giant training file.

Recommended layout:

    Own-AI-Data/
      00_manifests/
      01_owned/
      02_open_license/
      03_code/
      04_education/
      05_sinhala/
      06_documents/
      07_conversations/
      08_images/
      09_audio/
      10_video_metadata/
      99_quarantine/

Keep raw source data separate from cleaned training data.

## Rules

- Keep source attribution and license information beside downloaded datasets.
- Never mix private credentials, API keys, OAuth tokens, or secrets into the dataset.
- Do not assume a web dataset is free to train on; record its license before ingestion.
- Prefer user-owned, public-domain, or clearly open-licensed sources.
- Deduplicate before training.
- Keep validation and test material isolated from the training folders.
- Store manifests with SHA-256 hashes so large datasets can be verified without copying them again.

## Local cache strategy

The laptop does not need to hold all 4.5 TB.

Use a local cache such as:

    D:\Own AI\data\cache\

Download or copy only the current shard(s), train on them, then rotate the cache.

## Useful commands

Create a local manifest:

    python cloud\google_drive_sync.py manifest data

Upload an owned/open dataset folder:

    python cloud\google_drive_sync.py upload "D:\Own AI\data\to_upload"

Set the Drive parent folder with:

    set GOOGLE_DRIVE_PARENT_ID=YOUR_FOLDER_ID

First-time Drive login uses OAuth credentials from:

    secrets\google_credentials.json

The OAuth token is stored under:

    data\runtime\google_drive_token.json

Both locations are ignored by git when stored in the local project.
