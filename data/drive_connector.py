from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


ROOT = Path(__file__).resolve().parent.parent

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]

CREDENTIALS_FILE = next(
    ROOT.glob("*.json"),
    None
)

TOKEN_FILE = ROOT / "data" / "token.json"


def get_drive_service():

    credentials = None

    if TOKEN_FILE.exists():
        credentials = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES
        )

    if not credentials or not credentials.valid:

        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

        else:

            if CREDENTIALS_FILE is None:
                raise FileNotFoundError(
                    "OAuth credentials JSON file not found in D:\\Own AI"
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )

            credentials = flow.run_local_server(
                port=0
            )

        TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8"
        )

    return build(
        "drive",
        "v3",
        credentials=credentials
    )


if __name__ == "__main__":

    print("================================")
    print("      OWN AI GOOGLE DRIVE")
    print("================================")

    service = get_drive_service()

    result = service.files().list(
        pageSize=10,
        fields="files(id,name,mimeType)"
    ).execute()

    files = result.get("files", [])

    print("Connected successfully!")
    print("Files found:", len(files))
    print("--------------------------------")

    for file in files:
        print(
            file["name"],
            "|",
            file["mimeType"]
        )

    print("================================")
    print("GOOGLE DRIVE: CONNECTED")