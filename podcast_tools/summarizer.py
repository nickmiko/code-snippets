from ollama import chat, ChatResponse

class Summarizer:
    def __init__(self, model_name: str = 'llama2'):
        self.model_name = model_name

    def summarize(self, text: str, max_length: int = 100, prompt: str = None):
        if prompt:
            full_prompt = f"{prompt}\n\n{text}"
        else:
            full_prompt = text
        response: ChatResponse = chat(model=self.model_name, messages=[
            {
                'role': 'user',
                'content': full_prompt,
            },
        ], max_tokens=max_length)
        return response.message.content
    
if __name__ == "__main__":
    summarizer = Summarizer(model_name='gemma3')
    result = summarizer.summarize(text='This is a sample text to summarize.')
    print('-'*50)
    print(result)
