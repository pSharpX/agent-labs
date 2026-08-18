import asyncio
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import chain
from prompts import template

class ImperativeBasedChatOpenAI:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-3.5-turbo")


    def __chatbot(self):
        @chain
        async def func(question: str):
            """Imperative composition example of langchain components in async way"""
            prompt = await template.ainvoke({
                "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
                    The models outperform their smaller counterparts and have become invaluable for developer who are creating applications with NLP capabilities. 
                    Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
                "question": question,
            })
            return await self.model.ainvoke(prompt)

        return func

    async def initialize(self):
        print("Ask your question: ")
        question = input()
        answer = await self.__chatbot().ainvoke(question)
        print(answer)


imperative_chat = ImperativeBasedChatOpenAI()

if __name__ == '__main__':
    asyncio.run(imperative_chat.initialize())

