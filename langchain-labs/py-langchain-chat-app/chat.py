from langchain_core.messages import SystemMessage
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

settings = BaseModelSettings()
resolver = ProviderResolver(ModelProvider(settings.provider))
model = resolver.resolve(settings)

system_msg = SystemMessage(
    """Your are a helpful assistant that responds to questions with three exclamation marks."""
)

def ask_with_prompt_templates(question: str):
    prompt = template.invoke({
        "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
            The models outperform their smaller counterparts and have become invaluable for developer who are creating applications with NLP capabilities. 
            Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
        "question": question,
    })
    content = model.invoke(prompt)
    return content

def main():
    print("Ask your question: ")
    question = input()
    answer = ask_with_prompt_templates(question)
    print(answer)

if __name__ == '__main__':
    main()

