from langchain_openai.chat_models import ChatOpenAI

class OpenAIModel:
    def __init__(self, model="gpt-3.5-turbo", temperature=0.2, max_tokens=1000):
        self.model = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    def invoke(self, prompt):
        message = self.model.invoke(prompt)
        return message.content

    def stream(self, prompt):
        message = yield self.model.stream(prompt)
        #print(message)
        return message.content