import uuid
from typing import Annotated, TypedDict, Literal

from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel

from prompts import AGENDA_HANDLER_SYSTEM_PROMPT
from settings import BaseModelSettings, BaseToolSettings
from config import langfuse_handler
from helpers import add, sub, mul, div, WeatherClient
from src.contacts.domain.contact import Contact
from src.contacts.services.contact_service import ContactService

tools_settings = BaseToolSettings()
model_settings = BaseModelSettings()

weather_client = WeatherClient(tools_settings)
contact_service = ContactService()


class AgendaState(AgentState):
    user_id: str
    display_name: str
    contacts: list[dict]

class AgendaContext(TypedDict):
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
def update_contact_info(runtime: ToolRuntime[AgendaContext, AgendaState], contact: Contact) -> str:
    """Update an existing contact's information.

    Use this tool when the user requests changes to an existing contact.
    The provided Contact object must contain the contact's unique ID and
    the information to be updated.

    Args:
        contact: The contact record containing the contact ID and the
            updated information to persist.

    Returns:
        A Command containing the result of the contact update operation.
    """
    user_id = runtime.context.user_id
    contact_service.update_contact(contact)
    return "Contact info updated successfully !"


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


class WeatherWiseAgent:
    def __init__(self):
        self.settings = BaseModelSettings()
        self.model = init_chat_model(
            model=self.settings.model_name,
            model_provider=self.settings.provider,
            temperature=self.settings.temperature,
        )
        self.system_prompt = SystemMessage(
            content="""
            # Goal:
            - You are a helpful weather assistant specialized exclusively in providing weather information. Provide accurate, concise and up-to-date weather information
            
            # Tools
            - **get_weather**: Get accurate weather information
            
            # Instructions
            - Always use the **get_weather** tool to retrieve weather information.
            - Never guess or fabricate weather data.
            - Clearly state the location and relevant weather details such as temperature, conditions, precipitation, and forecast when available.
            - If the location is unclear, ask the user to specify it.
            
            # Scope
            You can answer questions about:
            - Current weather
            - Temperature
            - Weather conditions
            - Rain or precipitation
            - Wind
            - Humidity
            - Weather forecasts
            - Other information directly related to weather conditions
            
            # Guardrails
            - Stay on topic: Only answer questions related to weather.
            - If the user asks an unrelated question, politely refuse and redirect them to weather-related questions.
            - Do not provide general knowledge, news, sports, entertainment, coding help, medical advice, or other unrelated information.
            - Do not fabricate weather information or tool results.
            - Do not use information from your own knowledge when the get_whether tool can provide the requested data.
            - Do not claim to have weather information that was not returned by the tool.
            - Do not infer a city when the user's intended location is ambiguous; ask for clarification.
            - Keep responses concise and focused on the user's weather request.
            
            # Off-Topic Response
            For unrelated requests, respond with:
            - I'm a weather assistant, so I can only help with weather-related questions. Please provide a city and I'll check the weather for you.
            """)
        self.agent = create_agent(
            model=self.model,
            tools=[get_weather],
            system_prompt=self.system_prompt,
            name="weather-wise-agent",
        )

    def initialize(self, config):
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
        self.settings = BaseModelSettings()
        self.model = init_chat_model(
            model=self.settings.model_name,
            model_provider=self.settings.provider,
            temperature=self.settings.temperature,
        )
        self.system_prompt = AGENDA_HANDLER_SYSTEM_PROMPT
        # noinspection bad-argument-type
        self.agent = create_agent(
            model=self.model,
            tools=[search_contact, get_contact_info, update_contact_info],
            system_prompt=self.system_prompt,
            name="agenda-handler-agent",
            state_schema=AgendaState,
            context_schema=AgendaContext
        )

    def initialize(self, input_obj: dict, config):
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
                config=config,
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
    thread_config = {
        "callbacks": [langfuse_handler],
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }
    contact_info = load_default_user_info()
    #main_agent.initialize(config=thread_config)
    main_agent.initialize(input_obj= {
        "user_id": contact_info["user_id"],
        "display_name": contact_info["display_name"],
    }, config=thread_config)