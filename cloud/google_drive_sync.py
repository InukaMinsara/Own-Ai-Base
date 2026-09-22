from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = (
    ROOT / "data" / "processed" / "cloud_manifest.jsonl"
)


def sha256_file(path: Path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path, output: Path):
    root = root.resolve()
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    allowed = {
        ".txt",
        ".md",
        ".json",
        ".jsonl",
        ".csv",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".html",
        ".css",
        ".xml",
        ".yaml",
        ".yml",
        ".sql",
    }

    count = 0

    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in allowed
            ):
                continue

            rel = path.relative_to(root)

            try:
                stat = path.stat()
                digest = sha256_file(path)

                row = {
                    "path": str(rel).replace("\\", "/"),
                    "bytes": stat.st_size,
                    "sha256": digest,
                }

                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1

            except OSError as exc:
                print(
                    "SKIP:",
                    path,
                    "-",
                    exc,
                )

    return count


def google_drive_service():
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Install Google Drive dependencies first: "
            "pip install google-api-python-client google-auth google-auth-oauthlib"
        ) from exc

    token_path = Path(
        os.getenv(
            "GOOGLE_DRIVE_TOKEN",
            str(ROOT / "data" / "runtime" / "google_drive_token.json"),
        )
    )
    credentials_path = Path(
        os.getenv(
            "GOOGLE_DRIVE_CREDENTIALS",
            str(ROOT / "secrets" / "google_credentials.json"),
        )
    )

    scopes = [
        "https://www.googleapis.com/auth/drive.file",
    ]

    credentials = None

    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(
            str(token_path),
            scopes,
        )

    if credentials is None or not credentials.valid:
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise RuntimeError(
                "Install google-auth-oauthlib for first-time Drive login."
            ) from exc

        if not credentials_path.exists():
            raise FileNotFoundError(
                f"Google OAuth credentials not found: {credentials_path}"
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_path),
            scopes,
        )
        credentials = flow.run_local_server(
            port=0
        )

        token_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        token_path.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

    return build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


def ensure_folder(service, name, parent_id=None):
    query_parts = [
        "mimeType='application/vnd.google-apps.folder'",
        f"name='{name.replace(chr(39), chr(92)+chr(39))}'",
        "trashed=false",
    ]

    if parent_id:
        query_parts.append(
            f"'{parent_id}' in parents"
        )

    response = service.files().list(
        q=" and ".join(query_parts),
        spaces="drive",
        fields="files(id,name)",
        pageSize=10,
    ).execute()

    files = response.get(
        "files",
        [],
    )

    if files:
        return files[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }

    if parent_id:
        metadata["parents"] = [parent_id]

    created = service.files().create(
        body=metadata,
        fields="id",
    ).execute()

    return created["id"]


def upload_file(service, path: Path, parent_id: str):
    from googleapiclient.http import MediaFileUpload

    metadata = {
        "name": path.name,
        "parents": [parent_id],
    }

    media = MediaFileUpload(
        str(path),
        mimetype=(
            mimetypes.guess_type(
                str(path)
            )[0]
            or "application/octet-stream"
        ),
        resumable=True,
    )

    return service.files().create(
        body=metadata,
        media_body=media,
        fields="id,name,size",
    ).execute()


def upload_folder(local_root: Path, drive_parent_id: str | None):
    service = google_drive_service()
    root_folder = ensure_folder(
        service,
        local_root.name,
        drive_parent_id,
    )

    folders = {
        Path("."): root_folder,
    }

    uploaded = 0

    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue

        relative_parent = path.parent.relative_to(
            local_root
        )

        parent_id = folders.get(
            relative_parent
        )

        if parent_id is None:
            current = Path(".")
            parent_id = root_folder

            for part in relative_parent.parts:
                current = current / part
                parent_id = folders.get(current)
                if parent_id is None:
                    parent_id = ensure_folder(
                        service,
                        part,
                        folders[current.parent],
                    )
                    folders[current] = parent_id

        result = upload_file(
            service,
            path,
            parent_id,
        )
        uploaded += 1

        print(
            f"UPLOADED {uploaded}: "
            f"{path.relative_to(local_root)} "
            f"-> {result.get('id')}"
        )

    return uploaded


def main():
    parser = argparse.ArgumentParser(
        description="Own AI Google Drive dataset tools."
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    manifest = sub.add_parser(
        "manifest",
        help="Build a SHA-256 manifest for a local dataset.",
    )
    manifest.add_argument(
        "folder",
        nargs="?",
        default=str(ROOT / "data"),
    )
    manifest.add_argument(
        "--output",
        default=str(DEFAULT_MANIFEST),
    )

    upload = sub.add_parser(
        "upload",
        help="Upload a dataset folder to Google Drive.",
    )
    upload.add_argument(
        "folder",
    )
    upload.add_argument(
        "--parent-id",
        default=os.getenv(
            "GOOGLE_DRIVE_PARENT_ID"
        ),
    )

    args = parser.parse_args()

    if args.command == "manifest":
        count = build_manifest(
            Path(args.folder),
            Path(args.output),
        )
        print(
            f"Manifest written: {args.output}"
        )
        print(
            f"Files hashed: {count}"
        )
        return

    folder = Path(args.folder).resolve()

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    count = upload_folder(
        folder,
        args.parent_id,
    )

    print(
        f"Drive upload complete. Files: {count}"
    )


if __name__ == "__main__":
    main()
