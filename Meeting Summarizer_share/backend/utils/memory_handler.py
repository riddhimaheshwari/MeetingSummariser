import os
import json

CHAT_DIR = "db/chat_history"

def ensure_chat_dir():
    os.makedirs(CHAT_DIR, exist_ok=True)

def get_chat_file(meeting_id):
    ensure_chat_dir()
    return os.path.join(CHAT_DIR, f"{meeting_id}.json")

def load_chat_history(meeting_id):
    file_path = get_chat_file(meeting_id)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

def save_chat_history(meeting_id, history):
    file_path = get_chat_file(meeting_id)
    with open(file_path, "w") as f:
        json.dump(history, f, indent=2)
