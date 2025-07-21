import os
import subprocess
import whisper

class Transcriber:
    def __init__(self, model_name: str = 'base'):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio: str, language: str = 'en'):
        # Convert to wav if not already
        if not audio.lower().endswith('.wav'):
            wav_path = os.path.splitext(audio)[0] + '.wav'
            if not os.path.exists(wav_path):
                subprocess.run([
                    'ffmpeg', '-y', '-i', audio, '-ar', '16000', '-ac', '1', wav_path
                ], check=True)
            audio = wav_path
        return self.model.transcribe(audio=audio, language=language, verbose=True)

if __name__ == "__main__":
    transcriber = Transcriber()
    result = transcriber.transcribe(audio='./input/audio.wav')
    print('-'*50)
    print(result.get('text', ''))