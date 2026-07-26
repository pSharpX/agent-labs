from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from prompts import template
from output_formats import AnswerWithJustification

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
system_msg = SystemMessage(
    """Your are a helpful assistant that responds to questions with three exclamation marks."""
)
structured_model = model.with_structured_output(AnswerWithJustification)

def ask(question: str):
    prompt = [system_msg, HumanMessage(question)]
    answer = model.invoke(prompt)
    return answer.content

def ask_with_prompt_templates(question: str):
    prompt = template.invoke({
        "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
            The models outperform their smaller counterparts and have become invaluable for developer who are creating applications with NLP capabilities. 
            Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
        "question": question,
    })
    print(prompt)
    answer = model.invoke(prompt)
    return answer.content

def ask_with_output_format(question: str):
    answer = structured_model.invoke(question)
    return answer


def main():
    print("Ask your question: ")
    question = input()
    answer = ask_with_output_format(question)
    print(answer)

if __name__ == '__main__':
    main()

