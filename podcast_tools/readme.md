prompts used

```
i want to use this script to use whisper ai to transcribe an mp3 podcast episode and then send then transcription to ollama to be summarized with different prompts depending on what the podcast is
````

whisper can only accept wav files if the input file is not wav use ffmepg to convert```

Use

`python podcast_tools/main.py /path/to/episode.mp3 --podcast_type tech --max_length 300 --language en`