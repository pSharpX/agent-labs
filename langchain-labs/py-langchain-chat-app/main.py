from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage
from prompts import template
from output_formats import AnswerWithJustification

class MainChatOpenAI:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.structured_model = self.model.with_structured_output(AnswerWithJustification)

        self.system_msg = SystemMessage(
            """Your are a helpful assistant that responds to questions with three exclamation marks."""
        )

    def __ask_with_prompt_templates(self, question: str):
        prompt = template.invoke({
            "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
                The models outperform their smaller counterparts and have become invaluable for developer who are creating applications with NLP capabilities. 
                Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
            "question": question,
        })
        answer = self.model.invoke(prompt)
        return answer.content

    def __ask_with_output_format(self, question: str):
        answer = self.structured_model.invoke(question)
        return answer

    def initialize(self):
        print("Ask your question: ")
        question = input()
        answer = self.__ask_with_output_format(question)
        print(answer)


main_chat = MainChatOpenAI()

if __name__ == '__main__':
    main_chat.initialize()

