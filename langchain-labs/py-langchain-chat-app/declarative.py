from langchain_openai.chat_models import ChatOpenAI
from prompts import template

class DeclarativeBasedChatOpenAI:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-3.5-turbo")

        # combine them with the | operator
        self.chatbot = template | self.model

    def __ask(self, question: str):
        """Declarative composition example of langchain components"""
        return self.chatbot.invoke({
            "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
                The models outperform their smaller counterparts and have become invaluable for developer who are creating applications with NLP capabilities. 
                Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
            "question": question,
        })

    def initialize(self):
        print("Ask your question: ")
        question = input()
        answer = self.__ask(question)
        print(answer)


declarative_chat = DeclarativeBasedChatOpenAI()

if __name__ == '__main__':
    declarative_chat.initialize()
