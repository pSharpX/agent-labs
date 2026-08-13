from functools import wraps
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from settings import BaseModelSettings


def log_token_consumption(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        if self.log_enabled:
            self._log_token_consumption(msgs=[args[0]])

        return result
    return wrapper

class MemoryPoweredChatOpenAI:
    def __init__(self, log_enabled: bool = False):
        self.settings = BaseModelSettings()
        self.model = init_chat_model(
            model=self.settings.model_name,
            model_provider=self.settings.provider,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Answer all questions to the best of your ability."""),
            ("placeholder", "{messages}"),
        ])
        self.messages = []
        self.log_enabled = log_enabled

    @log_token_consumption
    def __add_human_message(self, message: HumanMessage):
        self.messages.append(message)

    @log_token_consumption
    def __add_ai_message(self, message: AIMessage):
        self.messages.append(message)

    def _log_token_consumption(self, msgs: list[BaseMessage]):
        num_tokens = self.model.get_num_tokens_from_messages(messages=msgs)
        print(f"Tokens consumed: {num_tokens}")

    def __ask(self, question: str):
        self.__add_human_message(HumanMessage(content=question))
        prompt = self.prompt_template.invoke({
            "messages": self.messages,
        })
        answer_msg = self.model.invoke(prompt)
        self.__add_ai_message(answer_msg)

        return answer_msg.content

    def initialize(self):
        print("Welcome to ChatPrompt!")
        print("Start typing ('c' for exit) >> ")
        while True:
            question = input()
            if question == "c":
                break
            answer = self.__ask(question)
            print(answer)
            self._log_token_consumption(self.messages)


main_chat = MemoryPoweredChatOpenAI(log_enabled=True)

if __name__ == '__main__':
    main_chat.initialize()

