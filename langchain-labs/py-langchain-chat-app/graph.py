import uuid
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from settings import BaseModelSettings


class State(TypedDict):
    messages: Annotated[list, add_messages]

class MemorySaverSingleNodeGraphPoweredChat:
    def __init__(self):
        self.settings = BaseModelSettings()
        self.model = init_chat_model(
            model=self.settings.model_name,
            model_provider=self.settings.provider,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        self.builder = StateGraph(MessagesState)
        self.graph = self.__build()

    def __chatbot(self):
        def func(state: MessagesState):
            print(state["messages"])
            answer_msg = self.model.invoke(state["messages"])

            return {
                "messages": [answer_msg],
            }

        return func

    def __build(self):
        self.builder.add_node("chatbot", self.__chatbot())

        self.builder.add_edge(START, "chatbot")
        self.builder.add_edge("chatbot", END)

        return self.builder.compile(checkpointer=InMemorySaver())

    def initialize(self, config):
        print("Welcome to ChatPrompt!")
        print("Start typing ('c' for exit) >> ")
        while True:
            question = input()
            if question == "c":
                break
            state = self.graph.invoke(input={
                "messages": [HumanMessage(content=question)]
            }, config=config)
            print(state["messages"][-1].content)


main_chat = MemorySaverSingleNodeGraphPoweredChat()

if __name__ == '__main__':
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    main_chat.initialize(thread_config)