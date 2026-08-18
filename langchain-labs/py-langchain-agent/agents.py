import uuid
import operator
from typing import Annotated, TypedDict, Literal

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage

from settings import BaseModelSettings, BaseToolSettings
from helpers import add, sub, mul, div, WeatherClient

tools_settings = BaseToolSettings()
model_settings = BaseModelSettings()

weather_client = WeatherClient(tools_settings)

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
            print(state["messages"])
            print(state["messages"][-1].content)


main_agent = WeatherWiseAgent()

if __name__ == '__main__':
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    main_agent.initialize(thread_config)