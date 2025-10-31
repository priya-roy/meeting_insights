# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- General App Config ---
ROOT_FOLDER_NAME = os.getenv("ROOT_FOLDER_NAME", "PythonKMSessionInsightsApp")
COMPETENCY = os.getenv("COMPETENCY", "PHP Drupal")
RECIPIENTS = [r.strip() for r in os.getenv("RECIPIENTS", "").split(",") if r.strip()]

# --- Google Drive ---
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")

# --- SMTP ---
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_SENDER_EMAIL = os.getenv("SMTP_SENDER_EMAIL", SMTP_USER)

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Directories ---
BASE_DIR = "data"
LOCAL_SESSION_DIR = os.path.join(BASE_DIR, "session")
