import requests

class WhisperATClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def transcribe(
        self,
        file_path: str,
        audio_tagging_time_resolution: int = 10,
        temperature: float = 0.01,
        no_speech_threshold: float = 0.4
    ) -> dict:
        """
        Sends an audio file to the transcription API and returns filtered transcript.
        """
        url = f"{self.base_url}/transcribe/"
        files = {
            'file': (file_path, open(file_path, 'rb'), 'audio/x-wav')
        }
        data = {
            'audio_tagging_time_resolution': str(audio_tagging_time_resolution),
            'temperature': str(temperature),
            'no_speech_threshold': str(no_speech_threshold),
        }

        headers = {
            'accept': 'application/json',
        }

        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()

            # Step 1: check if any audio tag class matches the special sounds
            if self._contains_special_audio_tags(result.get("audio_tags", [])):
                result["final_text"] = "(DIAL TONE BEEP)"
            else:
                # Step 2: Otherwise extract filtered speech
                result["final_text"] = self._extract_text_from_segments(
                    result, no_speech_threshold
                )

            return result

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return {"transcript": {}, "error": str(e), "final_text": ""}

        finally:
            files['file'][1].close()

    def _extract_text_from_segments(self, response, threshold: float) -> str:
        """
        Extracts and concatenates text from segments where no_speech_prob < threshold.
        """
        segments = response.get("segments", [])
        filtered_text = " ".join(
            segment["text"].strip()
            for segment in segments
            if segment.get("no_speech_prob", 1.0) < threshold
        )
        return filtered_text.strip()

    def _contains_special_audio_tags(self, audio_tags: list) -> bool:
        """
        Checks if audio tags contain special sound classes like Dial tone, Beep, etc.
        """
        target_classes = {
            "Telephone", "Telephone bell ringing", "Ringtone", "Telephone dialing, DTMF",
            "Dial tone", "Busy signal", "Alarm clock", "Siren", "Civil defense siren",
            "Buzzer", "Tearing", "Beep, bleep", "Ping", "Sine wave", "Echo",
            "Sidetone", "Sound effect", "Cowbell", "Vibraphone"
        }

        for tag_group in audio_tags:
            for tag in tag_group.get("audio tags", []):
                class_name = tag[0].strip()
                if class_name in target_classes:
                    return True
        return False


# Example usage:
if __name__ == "__main__":
    client = WhisperATClient("http://148.251.178.29:9007")
    result = client.transcribe("audios/417956011590176.wav")
    print("Final Text:", result["final_text"])
