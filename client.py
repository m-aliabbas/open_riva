import requests

# Configuration
SERVER_URL = "http://localhost:8000/transcribe/"
AUDIO_PATH = "audios/2086-149220-0033.wav"  # change to your file path

# Make request
with open(AUDIO_PATH, "rb") as f:
    files = {"file": (AUDIO_PATH, f, "audio/wav")}
    response = requests.post(SERVER_URL, files=files)

# Show response
if response.status_code == 200:
    data = response.json()
    print("📝 Transcript:", data.get("transcript", "[EMPTY]"))
else:
    print("❌ Error:", response.status_code, response.text)
