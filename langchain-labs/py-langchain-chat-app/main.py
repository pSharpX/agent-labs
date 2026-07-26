from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-3.5-turbo")
system_msg = SystemMessage(
    """Your are a helpful assistant that responds to questions with three exclamation marks."""
)

def ask(question: str):
    prompt = [system_msg, HumanMessage(question)]
    answer = model.invoke(prompt)
    print(answer.content)

def main():
    print("Ask your question: ")
    question = input()
    ask(question)

if __name__ == '__main__':
    main()

