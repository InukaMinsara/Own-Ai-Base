from drive_connector import get_drive_service

service = get_drive_service()

result = service.files().list(
    q="'root' in parents and trashed = false",
    fields="files(id,name,mimeType)"
).execute()

print("================================")
print("       GOOGLE DRIVE ROOT")
print("================================")

for file in result.get("files", []):
    print(file["name"], "|", file["mimeType"])

print("================================")