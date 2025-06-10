import os
import requests

# Configuration
SERVER_URL = "http://148.251.195.218:7005/transcribe/"
AUDIO_PATH = "audios/am.wav"  # Change this to your audio file

# Ensure the file exists
if not os.path.isfile(AUDIO_PATH):
    print(f"❌ File not found: {AUDIO_PATH}")
    exit(1)

# Send file to server
with open(AUDIO_PATH, "rb") as f:
    filename = os.path.basename(AUDIO_PATH)  # Just the name, not full path
    files = {"file": (filename, f, "audio/wav")}

    try:
        response = requests.post(SERVER_URL, files=files)

        if response.status_code == 200:
            data = response.json()
            print("✅ Transcript:", data.get("transcript", "[EMPTY]"))

        elif response.status_code == 503:
            print("❌ Server unavailable: ASR model not loaded")
            print(response.json())

        elif response.status_code == 500:
            print("❌ Server error (500):")
            print(response.json().get("traceback"))

        else:
            print(f"❌ Unexpected response: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
