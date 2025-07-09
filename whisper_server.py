import os
import shutil
import traceback

from fastapi import FastAPI, UploadFile, File, status
from fastapi.responses import JSONResponse
import uvicorn

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import pipeline

app = FastAPI(title="ASR Server using HuggingFace Whisper")

# Load ASR model at startup
asr_pipe = None

@app.on_event("startup")
def load_model():
    global asr_pipe
    try:
        print("🔄 Loading HuggingFace Whisper model...")
        asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",device=device)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        asr_pipe = None

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    global asr_pipe

    if asr_pipe is None:
        return JSONResponse(status_code=503, content={"error": "ASR model not available"})

    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = asr_pipe(temp_path)
        os.remove(temp_path)

        return {"transcript": result["text"], "asr_name": "hf_whisper"}
    
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"traceback": traceback.format_exc()}
        )

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    global asr_pipe

    test_audio = "./audios/hello.wav"
    if not os.path.exists(test_audio):
        return JSONResponse(status_code=502, content={"error": "Test audio file not found"})

    if asr_pipe is None:
        return JSONResponse(status_code=503, content={"error": "ASR model not loaded"})

    try:
        result = asr_pipe(test_audio)
        transcript = result.get("text", "").strip()

        if not transcript:
            return JSONResponse(status_code=502, content={"error": "Empty transcript"})

        return JSONResponse(status_code=200, content={"transcript": transcript, "status": "ok"})

    except Exception:
        return JSONResponse(status_code=502, content={"traceback": traceback.format_exc()})

if __name__ == "__main__":
    uvicorn.run("whisper_server:app", host="0.0.0.0", port=8000, reload=True)
