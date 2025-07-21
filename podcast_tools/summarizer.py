import ollama
class Summarizer:
    def __init__(self, model_name: str = 'llama2'):
        self.model = ollama.Model(model_name)

    def summarize(self, text: str, max_length: int = 100, prompt: str = None):
        if prompt:
            full_prompt = f"{prompt}\n\n{text}"
        else:
            full_prompt = text
        response = self.model.chat(full_prompt, max_tokens=max_length)
        return response.get('message', {}).get('content', '')
    
if __name__ == "__main__":
    summarizer = Summarizer()
    result = summarizer.summarize(text='This is a sample text to summarize.')
    print('-'*50)
    print(result)
