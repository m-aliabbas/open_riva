from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

class SpeechClassifier:
    def __init__(self, min_speech_duration_ms=100, min_silence_duration_ms=500):
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.model = load_silero_vad()
        print("✅ Silero VAD model loaded successfully.")
        
    def get_speech_timestamps(self, audio_path, threshold):
        wav = read_audio(audio_path)
        return get_speech_timestamps(
            wav,
            self.model,
            threshold=threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True  # Return speech timestamps in samples
        )
    
    def total_speech_duration(self, speech_timestamps):
        if not speech_timestamps:
            return 0
        return speech_timestamps[-1]['end'] - speech_timestamps[0]['start']

    def has_speech(self, audio_path, threshold=0.35, verbose=False):
        # Get speech timestamps for the provided audio and threshold
        speech_timestamps = self.get_speech_timestamps(audio_path, threshold)
        
        # Calculate total speech duration
        total_duration = self.total_speech_duration(speech_timestamps)
        
        if verbose:
            print(f"Speech Timestamps: {speech_timestamps}")
            print(f"Total Speech Duration (in samples): {total_duration}")
        
        # Return True if there is speech, False otherwise
        has_speech = len(speech_timestamps) >= 1 and total_duration > 0.1 # 0.1s * 16000 samples/sec = 1600 samples
        if verbose:
            print(f"Speech Detected: {has_speech}")
        
        return has_speech


if __name__ == "__main__":
    # Usage
    audio_path = 'audios/example_call.wav'
    classifier = SpeechClassifier()

    # Call has_speech with audio path and threshold as arguments
    result = classifier.has_speech(audio_path, threshold=0.49, verbose=True)
    print(result)