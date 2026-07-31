import asyncio
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import chain
from prompts import template


model = ChatOpenAI(model="gpt-3.5-turbo")

@chain
async def chatbot(question: str):
    """Imperative composition example of langchain components in async way"""
    prompt = await template.ainvoke({
        "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
            The models outperform their smaller counterparts and have become invaluable for developer who are creating applications with NLP capabilities. 
            Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
        "question": question,
    })
    return await model.ainvoke(prompt)

async def main():
    print("Ask your question: ")
    question = input()
    answer = await chatbot.ainvoke(question)
    print(answer)

if __name__ == '__main__':
    asyncio.run(main())

