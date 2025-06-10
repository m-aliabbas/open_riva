import os
import glob

os.environ["NEMO_DISABLE_TDT_CUDA_GRAPHS"] = "1"

import nemo.collections.asr as nemo_asr

# Load ASR model once
print("🔄 Loading ASR model...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

# Config
AUDIO_DIR = "audios"
SHORT_BURST_THRESHOLD = 0.1  # seconds

# Helper function to process one audio file
def transcribe_with_burst_filter(filepath):
    try:
        output = asr_model.transcribe([filepath],timestamps=True)
        hyp = output[0]
        # print(hyp.text)
        segments = hyp.timestamp["segment"]
        # print(hyp)
        cleaned_segments = []

        for seg in segments:
            # print(seg)
            duration = seg["end"] - seg["start"]
            if duration > SHORT_BURST_THRESHOLD:
                cleaned_segments.append(seg["segment"])
            else:
                print(f"🚫 Short burst removed in {os.path.basename(filepath)}: '{seg['segment']}' ({duration:.2f}s)")

        return " ".join(cleaned_segments).strip()

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return ""


# Loop over all .wav files in the directory
audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

print(f"🎧 Found {len(audio_files)} audio files in '{AUDIO_DIR}/'.")

# Process each file
for wav_path in audio_files:
    print(f"\n🔊 Processing: {os.path.basename(wav_path)}")
    transcript = transcribe_with_burst_filter(wav_path)
    print(f"📝 Transcript: {transcript if transcript else '[EMPTY]'}")
