import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import uvicorn
import traceback
from fastapi import status

from dotenv import load_dotenv
load_dotenv()

WHISPER_AT_ADDRESS = os.getenv("WHISPER_AT_ADDRESS","http://127.0.0.1:9007")
os.environ["NEMO_DISABLE_TDT_CUDA_GRAPHS"] = "1"
from whisper_at_client import WhisperATClient

whisper_at_client = WhisperATClient(WHISPER_AT_ADDRESS)

import nemo.collections.asr as nemo_asr

app = FastAPI(title="ASR Server with Short Burst Filtering")

SHORT_BURST_THRESHOLD = 0.1  # seconds
asr_model = None  # Global reference

# Load model at startup
@app.on_event("startup")
def load_model():
    global asr_model
    try:
        print("🔄 Loading ASR model...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )
    except Exception as e:
        print(f"❌ Failed to load ASR model: {e}")
        asr_model = None


def transcribe_with_burst_filter(filepath: str,helping_asr=False) -> str:
    global asr_model
    asr_status = 'p'
    if asr_model is None:
        raise RuntimeError("ASR model is not loaded")

    output = asr_model.transcribe([filepath], timestamps=True)
    hyp = output[0]
    # print(hyp)
    segments = hyp.timestamp.get("segment", [])
    cleaned_segments = []

    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration > SHORT_BURST_THRESHOLD:
            cleaned_segments.append(seg["segment"])
        else:
            print(f"🚫 Short burst removed: '{seg['segment']}' ({duration:.2f}s)")
    text = " ".join(cleaned_segments).strip()
    
    if helping_asr and len(text) < 1:
        response = whisper_at_client.transcribe(filepath)
        # print('AT Response',response)
        text = response.get("final_text","")
        asr_status = 'wat'
    return text,asr_status


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    global asr_model

    if asr_model is None:
        return JSONResponse(status_code=503, content={"error": "ASR model not available"})

    try:
        # Save uploaded file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        transcript,asr_status = transcribe_with_burst_filter(temp_path,helping_asr=True)
        os.remove(temp_path)

        return {"transcript": transcript,'asr_name':asr_status}
    
    except Exception:
        # Return full traceback
        return JSONResponse(
            status_code=500,
            content={"traceback": traceback.format_exc()}
        )


@app.get("/live", status_code=status.HTTP_200_OK)
def health_check():
    test_audio = "./audios/hello.wav"
    global asr_model

    if not os.path.exists(test_audio):
        return JSONResponse(status_code=502, content={"error": "Test audio file not found"})

    if asr_model is None:
        return JSONResponse(status_code=503, content={"error": "ASR model not loaded"})

    try:
        transcript,asr_status = transcribe_with_burst_filter(test_audio)

        if transcript.strip() == "":
            return JSONResponse(status_code=502, content={"error": "Empty transcript"})

        elif asr_status == 'wat':
            return JSONResponse(status_code=502, content={"error": "Empty transcript"}) 

        return JSONResponse(status_code=200, content={"transcript": transcript, "status": "ok"})

    except Exception:
        return JSONResponse(status_code=502, content={"traceback": traceback.format_exc()})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
