import unittest
from unittest.mock import patch, MagicMock
from podcast_tools.main import PodcastTools

class TestPodcastTools(unittest.TestCase):
    @patch('podcast_tools.main.summarizer')
    def test_summarize_text(self, mock_summarizer):
        mock_instance = MagicMock()
        mock_instance.summarize.return_value = 'short summary'
        mock_summarizer.Summarizer.return_value = mock_instance
        tools = PodcastTools()
        result = tools.summarize_text('long text', max_length=10)
        self.assertEqual(result, 'short summary')
        mock_instance.summarize.assert_called_once_with(text='long text', max_length=10)

    @patch('podcast_tools.main.transcriber')
    def test_transcribe_audio(self, mock_transcriber):
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = {'text': 'transcribed text'}
        mock_transcriber.Transcriber.return_value = mock_instance
        tools = PodcastTools()
        result = tools.transcribe_audio('audio.wav', language='en')
        self.assertEqual(result, {'text': 'transcribed text'})
        mock_instance.transcribe.assert_called_once_with(audio='audio.wav', language='en')

    @patch('podcast_tools.main.summarizer')
    @patch('podcast_tools.main.transcriber')
    def test_summarize_audio(self, mock_transcriber, mock_summarizer):
        mock_transcribe_instance = MagicMock()
        mock_transcribe_instance.transcribe.return_value = {'text': 'transcribed audio'}
        mock_transcriber.Transcriber.return_value = mock_transcribe_instance
        mock_summarize_instance = MagicMock()
        mock_summarize_instance.summarize.return_value = 'audio summary'
        mock_summarizer.Summarizer.return_value = mock_summarize_instance
        tools = PodcastTools()
        result = tools.summarize_audio('audio.wav', max_length=20, language='en')
        self.assertEqual(result, 'audio summary')
        mock_transcribe_instance.transcribe.assert_called_once_with(audio='audio.wav', language='en')
        mock_summarize_instance.summarize.assert_called_once_with(text='transcribed audio', max_length=20)

if __name__ == '__main__':
    unittest.main()
