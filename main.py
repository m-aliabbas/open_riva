import argparse
import uvicorn
from config import asr_type

if asr_type == "whisper":
    model_app = "whisper_server:app"
elif asr_type == "nemo":
    model_app = "nemo_server:app"
else:
    raise ValueError(f"❌ Invalid ASR type specified: {asr_type}. Supported types are 'whisper' and 'nemo'.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run ASR Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--workers", type=int, default=1, help="Number of Uvicorn workers")
    return parser.parse_args()

def main():
    args = parse_args()
    
    uvicorn.run(
        model_app,  # reference to your app module
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False  # Set to True if you want auto-reload during development
    )

if __name__ == "__main__":
    main()
