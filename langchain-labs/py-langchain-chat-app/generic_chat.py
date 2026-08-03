from langchain.chat_models import init_chat_model
from settings import BaseModelSettings
from prompts import template

settings = BaseModelSettings()
model = init_chat_model(model=settings.model_name, model_provider=settings.provider, temperature=settings.temperature)

def ask_with_prompt_templates(question: str):
    prompt = template.invoke({
        "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
            The models outperform their smaller counterparts and have become invaluable for developer who are creating applications with NLP capabilities. 
            Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
        "question": question,
    })
    message = model.invoke(prompt)
    return message.content

def main():
    print("Ask your question: ")
    question = input()
    answer = ask_with_prompt_templates(question)
    print(answer)

if __name__ == '__main__':
    main()

