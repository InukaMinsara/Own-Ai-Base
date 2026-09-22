from pathlib import Path
import io

from googleapiclient.http import MediaIoBaseDownload

from drive_connector import get_drive_service


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


TEXT_TYPES = {
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/json": ".json",
}

GOOGLE_DOC_TYPE = "application/vnd.google-apps.document"
FOLDER_TYPE = "application/vnd.google-apps.folder"


def find_folder(service, name, parent_id=None):

    query = (
        f"name = '{name}' "
        f"and mimeType = '{FOLDER_TYPE}' "
        "and trashed = false"
    )

    if parent_id:
        query += f" and '{parent_id}' in parents"

    result = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id,name)"
    ).execute()

    files = result.get("files", [])

    return files[0] if files else None


def download_file(service, file_id, output_file):

    request = service.files().get_media(
        fileId=file_id
    )

    buffer = io.BytesIO()

    downloader = MediaIoBaseDownload(
        buffer,
        request
    )

    done = False

    while not done:
        _, done = downloader.next_chunk()

    output_file.write_bytes(
        buffer.getvalue()
    )


def export_google_doc(service, file_id, output_file):

    request = service.files().export_media(
        fileId=file_id,
        mimeType="text/plain"
    )

    buffer = io.BytesIO()

    downloader = MediaIoBaseDownload(
        buffer,
        request
    )

    done = False

    while not done:
        _, done = downloader.next_chunk()

    output_file.write_bytes(
        buffer.getvalue()
    )


def download_text_files(service, folder_id):

    query = (
        f"'{folder_id}' in parents "
        "and trashed = false"
    )

    result = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id,name,mimeType)"
    ).execute()

    files = result.get("files", [])

    downloaded = 0

    for file in files:

        file_id = file["id"]
        name = file["name"]
        mime_type = file["mimeType"]

        if mime_type == FOLDER_TYPE:

            print("Entering folder:", name)

            downloaded += download_text_files(
                service,
                file_id
            )

            continue

        if mime_type == GOOGLE_DOC_TYPE:

            output_name = Path(name).stem + ".txt"
            output_file = RAW_DIR / output_name

            print("Exporting Google Doc:", name)

            export_google_doc(
                service,
                file_id,
                output_file
            )

            downloaded += 1

            continue

        if mime_type in TEXT_TYPES:

            extension = TEXT_TYPES[mime_type]

            output_name = Path(name).stem + extension
            output_file = RAW_DIR / output_name

            print("Downloading:", name)

            download_file(
                service,
                file_id,
                output_file
            )

            downloaded += 1

            continue

        print("Skipping:", name)

    return downloaded


def main():

    print("================================")
    print("     OWN AI DATASET COLLECTOR")
    print("================================")

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    service = get_drive_service()

    folder = find_folder(
        service,
        "OwnAI_Dataset"
    )

    if not folder:

        print(
            "ERROR: OwnAI_Dataset folder not found."
        )

        return

    print(
        "Dataset folder:",
        folder["name"]
    )

    print("--------------------------------")

    count = download_text_files(
        service,
        folder["id"]
    )

    print("--------------------------------")
    print("Downloaded files:", count)
    print("Raw dataset:", RAW_DIR)
    print("================================")
    print("DATASET COLLECTION COMPLETE")


if __name__ == "__main__":
    main()