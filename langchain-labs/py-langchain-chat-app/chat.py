from prompts import template
from settings import BaseModelSettings, ModelProvider
from openai_model import OpenAIModel
from anthropic_model import AnthropicModel
from ollama_model import OllamaModel
from gemini_model import GeminiModel

class ProviderResolver:
    def __init__(self, provider: ModelProvider):
        self.model_provider = provider

    def resolve(self, config: BaseModelSettings):
        if self.model_provider is ModelProvider.ANTHROPIC:
            return AnthropicModel(model=config.model_name, temperature=config.temperature,
                                     max_tokens=config.max_tokens)
        elif self.model_provider is ModelProvider.OLLAMA:
            return OllamaModel(model=config.model_name, temperature=config.temperature,
                                     max_tokens=config.max_tokens)
        elif self.model_provider is ModelProvider.GEMINI:
            return GeminiModel(model=config.model_name, temperature=config.temperature,
                                     max_tokens=config.max_tokens)
        return OpenAIModel(model=config.model_name, temperature=config.temperature,
                           max_tokens=config.max_tokens)

class CustomGenericChat:
    def __init__(self):
        self.settings = BaseModelSettings()
        self.resolver = ProviderResolver(ModelProvider(self.settings.provider))
        self.model = self.resolver.resolve(self.settings)

    def __ask_with_prompt_templates(self, question: str):
        prompt = template.invoke({
            "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
                The models outperform their smaller counterparts and have become invaluable for developer who are creating applications with NLP capabilities. 
                Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
            "question": question,
        })
        content = self.model.invoke(prompt)
        return content

    def initialize(self):
        print("Ask your question: ")
        question = input()
        answer = self.__ask_with_prompt_templates(question)
        print(answer)


main_chat = CustomGenericChat()

if __name__ == '__main__':
    main_chat.initialize()

