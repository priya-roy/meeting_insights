# drive_utils.py
import os
import io
import re
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from config import ROOT_FOLDER_NAME, LOCAL_SESSION_DIR, SERVICE_ACCOUNT_FILE

os.makedirs(LOCAL_SESSION_DIR, exist_ok=True)

def get_drive_service():
    """Authenticate and return the Google Drive API service."""
    if not SERVICE_ACCOUNT_FILE or not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError("Service account file missing or not found.")
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)

def find_root_folder(service):
    """Find root folder ID by name."""
    q = f"name='{ROOT_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'"
    res = service.files().list(q=q, fields="files(id, name)").execute()
    folders = res.get("files", [])
    if not folders:
        raise FileNotFoundError(
            f"Root folder '{ROOT_FOLDER_NAME}' not found. Please share it with the service account."
        )
    return folders[0]["id"]

def list_date_folders(service, root_folder_id):
    """List all date-based folders under root, sorted by date descending."""
    q = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    res = service.files().list(q=q, fields="files(id, name, createdTime)").execute()
    folders = res.get("files", [])
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_KM")
    valid_folders = []
    for f in folders:
        match = pattern.search(f["name"])
        if match:
            date_obj = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            valid_folders.append((date_obj, f))
    valid_folders.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in valid_folders]

def download_file(service, file_id, file_path):
    """Download a single file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"⬇️  Download progress: {int(status.progress() * 100)}%")
    print(f"✅ Download complete: {file_path}")

def download_latest_session_files():
    """Download the latest Knowledge Meet folder files (PDF and video if available)."""
    service = get_drive_service()
    root_folder_id = find_root_folder(service)
    date_folders = list_date_folders(service, root_folder_id)

    if not date_folders:
        raise FileNotFoundError("No dated session folders found under root folder.")

    latest_folder = date_folders[0]
    session_folder_id = latest_folder["id"]
    session_folder_name = latest_folder["name"]
    print(f"📁 Latest session folder detected: {session_folder_name}")

    os.makedirs("data/session", exist_ok=True)

    results = service.files().list(
        q=f"'{session_folder_id}' in parents",
        fields="files(id, name, mimeType, webViewLink)"
    ).execute()

    files = results.get("files", [])
    pdf_path, video_path, video_link = None, None, None

    for f in files:
        file_id, file_name, mime_type = f["id"], f["name"], f["mimeType"]
        print(f"🎞️ Detected file: {file_name} ({mime_type})")

        if "pdf" in mime_type or file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join("data/session", file_name)
            download_file(service, file_id, pdf_path)

        elif mime_type.startswith("video/") or "shortcut" in mime_type:
            video_link = f.get("webViewLink", None)
            if mime_type.startswith("video/"):
                video_path = os.path.join("data/session", file_name)
                try:
                    download_file(service, file_id, video_path)
                except Exception as e:
                    print(f"⚠️ Could not download video. Using Drive link only: {e}")

    if not pdf_path and not video_path:
        raise FileNotFoundError("No valid PDF or video found in the latest session folder.")

    return video_path, pdf_path, video_link, session_folder_name
