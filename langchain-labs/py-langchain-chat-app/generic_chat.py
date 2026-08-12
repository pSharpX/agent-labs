from langchain.chat_models import init_chat_model
from settings import BaseModelSettings
from prompts import template

class DefaultGenericChat:
    def __init__(self):
        self.settings = BaseModelSettings()
        self.model = init_chat_model(model=self.settings.model_name, model_provider=self.settings.provider, temperature=self.settings.temperature)

    def __ask_with_prompt_templates(self, question: str):
        prompt = template.invoke({
            "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
                The models outperform their smaller counterparts and have become invaluable for developer who are creating applications with NLP capabilities. 
                Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
            "question": question,
        })
        message = self.model.invoke(prompt)
        return message.content

    def initialize(self):
        print("Ask your question: ")
        question = input()
        answer = self.__ask_with_prompt_templates(question)
        print(answer)


main_chat = DefaultGenericChat()

if __name__ == '__main__':
    main_chat.initialize()

