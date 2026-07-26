from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-3.5-turbo")

def ask(question: str):
    prompt = [HumanMessage(question)]
    answer = model.invoke(prompt)
    print(answer.content)

def main():
    print("Ask your question: ")
    question = input()
    ask(question)

if __name__ == '__main__':
    main()

