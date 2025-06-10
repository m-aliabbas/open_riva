## 🧠 **OpenRiva ASR Server**

An automatic speech recognition (ASR) server using NVIDIA NeMo's `parakeet-tdt-0.6b-v2` model, with hallucination filtering based on segment duration ("short burst" detection). Includes a FastAPI-based server and a Python client for sending audio and receiving transcriptions.

---

### 🔧 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 📁 Project Structure

```
open_riva/
├── main.py           # FastAPI server
├── client.py         # Request client for testing
├── audios/           # Audio directory (e.g., hello.wav, am.wav)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

### 🚀 Running the Server

Start the FastAPI server with:

```bash
python main.py
```

Or using uvicorn manually:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

### 📡 API Endpoints

#### 🔊 `/transcribe/` (POST)

Transcribes an uploaded `.wav` file.

**Request:**

* `multipart/form-data` with key: `file`

**Response:**

```json
{
  "transcript": "Hello? How are you?"
}
```

#### ❤️ `/live` (GET)

Health check endpoint. It tries to transcribe `./audios/hello.wav`.

* **200 OK** → Model and audio are functional
* **502** → Audio missing or transcription failed
* **503** → Model not loaded

---

### 🧪 Running the Client

To test the server with a local audio file:

```bash
python client.py
```

Make sure to edit `client.py` and point to your actual `.wav` file and the correct server IP/port.

---

### 🧠 Hallucination Filtering

This server uses a **short burst filter** to remove hallucinated segments (e.g., "yeah", "we") that are ≤ `0.1s` in duration — a common artifact in silent or noisy inputs.

---

### 👨‍💻 Developer Info

* **Developer:** Muhammad Ali Abbas
* **Role:** Senior ML Engineer, Idrak Ai
* **Contribution:**

  * Built client/server architecture
  * Integrated NeMo ASR with FastAPI
  * Designed and implemented hallucination detection via short-burst filtering

---

### ✅ Example Output On Server

```bash
✅ Transcript: Hello? How are you?
```

```bash
🚫 Short burst removed: 'Yeah' (0.08s)
```

