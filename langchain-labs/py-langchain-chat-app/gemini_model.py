from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiModel:
    def __init__(self, model="gemini-3.1-pro-preview", temperature=0.2, max_tokens=1000):
        self.model = ChatGoogleGenerativeAI(model=model)

    def invoke(self, prompt):
        message = self.model.invoke(prompt)
        return message.content

    def stream(self, prompt):
        message = yield self.model.stream(prompt)
        #print(message)
        return message.content