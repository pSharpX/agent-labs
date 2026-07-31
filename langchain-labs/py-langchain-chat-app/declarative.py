from langchain_openai.chat_models import ChatOpenAI
from prompts import template


model = ChatOpenAI(model="gpt-3.5-turbo")

# combine them with the | operator
chatbot = template | model

def ask(question: str):
    """Declarative composition example of langchain components"""
    return chatbot.invoke({
        "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
            The models outperform their smaller counterparts and have become invaluable for developer who are creating applications with NLP capabilities. 
            Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
        "question": question,
    })

def main():
    print("Ask your question: ")
    question = input()
    answer = ask(question)
    print(answer)


if __name__ == '__main__':
    main()
