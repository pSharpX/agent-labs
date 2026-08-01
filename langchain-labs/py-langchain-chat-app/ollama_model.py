from langchain_ollama import ChatOllama

class OllamaModel:
    def __init__(self, model="llama3.1", temperature=0.1, max_tokens=1000):
        self.model = ChatOllama(model=model, temperature=temperature, num_predict=max_tokens)

    def invoke(self, prompt):
        message = self.model.invoke(prompt)
        return message.content

    def stream(self, prompt):
        message = yield self.model.stream(prompt)
        #print(message)
        return message.content
