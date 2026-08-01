from langchain_anthropic import ChatAnthropic

class AnthropicModel:
    def __init__(self, model="claude-sonnet-4-5-20250929", temperature=0.2, max_tokens=1000):
        self.model = ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens)

    def invoke(self, prompt):
        message = self.model.invoke(prompt)
        return message.content

    def stream(self, prompt):
        message = yield self.model.stream(prompt)
        #print(message)
        return message.content