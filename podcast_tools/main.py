import podcast_tools.transcriber as transcriber
import podcast_tools.summarizer as summarizer

class PodcastTools:
    def __init__(self):
        self.transcriber = transcriber.Transcriber()
        self.summarizer = summarizer.Summarizer()

    def summarize_text(self, text: str, max_length: int = 100, prompt: str = None):
        return self.summarizer.summarize(text=text, max_length=max_length, prompt=prompt)

    def transcribe_audio(self, audio_path: str, language: str = 'en'):
        return self.transcriber.transcribe(audio=audio_path, language=language)
    
    def summarize_audio(self, audio_path: str, max_length: int = 100, language: str = 'en', prompt: str = None, batch_char_limit: int = 4000):
        transcription = self.transcribe_audio(audio_path, language)
        text = transcription.get('text', '')
        if len(text) <= batch_char_limit:
            return self.summarize_text(text=text, max_length=max_length, prompt=prompt)
        # Split text into batches
        batches = []
        start = 0
        while start < len(text):
            end = min(start + batch_char_limit, len(text))
            # Try to split at a sentence boundary if possible
            if end < len(text):
                period = text.rfind('.', start, end)
                if period != -1 and period > start:
                    end = period + 1
            batches.append(text[start:end].strip())
            start = end
        # Summarize each batch
        batch_summaries = [self.summarize_text(text=batch, max_length=max_length, prompt=prompt) for batch in batches]
        # Summarize the summaries
        combined_summary = '\n'.join(batch_summaries)
        final_summary = self.summarize_text(text=combined_summary, max_length=max_length, prompt=prompt)
        return final_summary
    

# Example podcast prompts (customize as needed)
PODCAST_PROMPTS = {
    'tech': "Summarize this tech podcast episode in 5 bullet points, focusing on key innovations discussed.",
    'news': "Summarize this news podcast episode, highlighting the main stories and their implications.",
    'sports': "Summarize this sports podcast episode, listing the main games and outcomes discussed.",
    # Add more podcast types and prompts as needed
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe and summarize a podcast episode.")
    parser.add_argument('audio_path', type=str, help='Path to the podcast audio file (mp3, wav, etc)')
    parser.add_argument('--podcast_type', type=str, default=None, help='Type of podcast (e.g., tech, news, sports)')
    parser.add_argument('--max_length', type=int, default=300, help='Max tokens for summary')
    parser.add_argument('--language', type=str, default='en', help='Language for transcription')
    args = parser.parse_args()

    tools = PodcastTools()
    print('-'*50)
    print('Transcribing...')
    audio_result = tools.transcribe_audio(audio_path=args.audio_path, language=args.language)
    print(audio_result.get('text', ''))

    print('-'*50)
    print('Summarizing...')
    prompt = PODCAST_PROMPTS.get(args.podcast_type, None)
    audio_summary_result = tools.summarize_audio(audio_path=args.audio_path, max_length=args.max_length, language=args.language, prompt=prompt)
    print(audio_summary_result)