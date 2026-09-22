from drive_connector import get_drive_service

service = get_drive_service()

result = service.files().list(
    q="trashed = false",
    fields="files(name,mimeType,parents)"
).execute()

print("================================")
print("       OWNAI DATASET FILES")
print("================================")

for file in result.get("files", []):
    print(file["name"], "|", file["mimeType"])

print("================================")