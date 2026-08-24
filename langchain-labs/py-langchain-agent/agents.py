import uuid
from contextlib import ExitStack
from typing import Annotated, TypedDict, Literal
import warnings

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

from prompts import AGENDA_HANDLER_SYSTEM_PROMPT, WEATHER_ASSISTANT_SYSTEM_PROMPT, CINE_FINDER_SYSTEM_PROMPT
from settings import BaseModelSettings, BaseToolSettings, DatabaseSettings
from config import langfuse_handler
from helpers import add, sub, mul, div, WeatherClient
from src.contacts.domain.contact import Contact
from src.contacts.services.contact_service import ContactService

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\..*",
)

tools_settings = BaseToolSettings()

weather_client = WeatherClient(tools_settings)
contact_service = ContactService()


class AgendaState(AgentState):
    user_id: str
    display_name: str
    contacts: list[dict]

class AgendaContext(BaseModel):
    user_id: str
    display_name: str

class Input(TypedDict):
    user_id: str
    display_name: str
    question: str


@tool
def search_contact(runtime: ToolRuntime, name: str) -> list[Contact]:
    """Find a contact by name.

    Use this tool when you need to look up an existing contact.
    Provide the contact's full name or a partial name. If multiple
    contacts match, all matching contacts are returned so you can
    identify the correct person.

    Args:
        name: Full or partial name of the contact to search for.

    Returns:
        A list of matching contact records.
    """
    user_id = runtime.state["user_id"]
    contacts = contact_service.search_contact(name)
    return contacts


@tool
def get_contact_info(runtime: ToolRuntime, contact_id: str) -> Contact | None:
    """Retrieve a contact's information by contact ID.

    Use this tool when you already have a contact's unique ID and need
    to retrieve their complete stored contact information.

    Args:
        contact_id: The unique identifier of the contact to retrieve.

    Returns:
        The matching Contact record if found, otherwise None.
    """
    user_id = runtime.state["user_id"]
    contact = contact_service.get_contact(contact_id=contact_id)
    return contact


@tool
def update_contact_info(runtime: ToolRuntime[AgendaContext, AgendaState], contact: Contact) -> Command:
    """Update an existing contact's information.

    Use this tool when the user requests changes to an existing contact.
    The provided Contact object must contain the contact's unique ID and
    the information to be updated.

    Args:
        contact: The contact record containing the contact ID and the
            updated information to persist.

    Returns:
        A Command confirming the result of the contact update operation.
    """
    user_id = runtime.context.user_id
    contact_service.update_contact(contact)
    return Command(update={
        "messages": [
            ToolMessage(
                "Contact info updated successfully !",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


@tool
def greet(runtime: ToolRuntime[AgendaContext, AgendaState]) -> str | Command:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
       return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello {user_name}!"

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city"""
    data = weather_client.get_weather(city)
    if data is None:
        return f"Unable to retrieve weather for {city}"

    description = data["current"]["condition"]["text"]
    temp = data["current"]["temp_c"]
    feels_like = data["current"]["feelslike_c"]
    humidity = data["current"]["humidity"]
    wind = data["current"]["wind_kph"]

    return (
        f"Weather in {city}:\n"
        f"- Temperature: {temp}°C\n"
        f"- Feels Like: {feels_like}°C\n"
        f"- Conditions: {description}\n"
        f"- Humidity: {humidity}%\n"
        f"- Wind Speed: {wind} k/h"
    )

tools_registry: dict = {
    "get_weather": get_weather,
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
}


@before_model
def log_before_call(state: AgendaState, runtime: Runtime) -> None:
    """Log messages before calling the model."""
    messages = state["messages"]
    print("================= START - BEFORE CALLING MODEL =================")
    print(messages)
    print("================= END - BEFORE CALLING MODEL =================")
    return None

@after_model
def log_after_call(state: AgendaState, runtime: Runtime) -> None:
    """Log messages after calling the model."""
    messages = state["messages"]
    print("================= START - AFTER CALLING MODEL =================")
    print(messages)
    print("================= END - AFTER CALLING MODEL =================")
    return None


class WeatherWiseAgent:
    def __init__(self):
        self.settings = BaseModelSettings()
        self.model = init_chat_model(
            model=self.settings.model_name,
            model_provider=self.settings.provider,
            temperature=self.settings.temperature,
        )
        self.system_prompt = SystemMessage(
            content=WEATHER_ASSISTANT_SYSTEM_PROMPT)
        self.agent = create_agent(
            model=self.model,
            tools=[get_weather],
            system_prompt=self.system_prompt,
            name="weather-wise-agent",
        )

    def start(self, config):
        print("Welcome to Weather Wise Agent!")
        print("Start typing ('c' for exit) >> ")
        while True:
            question = input()
            if question == "c":
                break
            elif question.strip() == "":
                continue
            state = self.agent.invoke(input={
                "messages": [HumanMessage(content=question)]
            }, config=config)
            print(state["messages"][-1].content)


class AgendaHandlerAgent:
    def __init__(self):
        self.model_settings = BaseModelSettings()
        self.db_settings = DatabaseSettings()
        self.model = init_chat_model(
            model=self.model_settings.model_name,
            model_provider=self.model_settings.provider,
            temperature=self.model_settings.temperature,
        )
        self.system_prompt = AGENDA_HANDLER_SYSTEM_PROMPT
        self.exit_stack = ExitStack()
        self.checkpointer = self.exit_stack.enter_context(
            PostgresSaver.from_conn_string(self.db_settings.raw_url)
        )
        self.checkpointer.setup()
        # noinspection bad-argument-type
        self.agent = create_agent(
            model=self.model,
            tools=[search_contact, get_contact_info, update_contact_info],
            system_prompt=self.system_prompt,
            middleware=[log_before_call, log_after_call],
            name="agenda-handler-agent",
            state_schema=AgendaState,
            context_schema=AgendaContext,
            checkpointer=self.checkpointer
        )

    def on_destroy(self):
        self.exit_stack.close()

    def start(self, input_obj: dict, session_id: str):
        print("Welcome to Agenda Handler Agent!")
        print("Start typing ('c' for exit) >> ")
        while True:
            question = input()
            if question == "c":
                break
            elif question.strip() == "":
                continue
            state = self.agent.invoke(
                input={
                    "user_id": input_obj["user_id"],
                    "display_name": input_obj["display_name"],
                    "messages": [HumanMessage(content=question)]
                },
                config={
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_user_id": input_obj["user_id"],
                        "langfuse_session_id": session_id,
                        "langfuse_tags": ["environment:dev", "framework:langchain", "application:py-langchain-agent"]
                    },
                    "configurable": {
                        "thread_id": session_id
                    }
                },
                context=input_obj,
            )
            print(state["messages"][-1].content)


def load_default_user_info(user_id: str = "6d95e39d-d5b0-4584-91b1-d1fc1efff25b") -> dict:
    contact: Contact | None = contact_service.get_contact(user_id)
    if contact is None:
        raise ValueError(f"Invalid user '{user_id}' provided")

    return {
        "user_id": str(contact.id),
        "display_name": contact.display_name,
    }


#main_agent = WeatherWiseAgent()
main_agent = AgendaHandlerAgent()

if __name__ == '__main__':
    contact_info = load_default_user_info()
    #main_agent.start(config=thread_config)
    main_agent.start(input_obj= {
        "user_id": contact_info["user_id"],
        "display_name": contact_info["display_name"],
    }, session_id=str(uuid.uuid4()))
    main_agent.on_destroy()