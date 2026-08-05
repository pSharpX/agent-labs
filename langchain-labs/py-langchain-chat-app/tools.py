import requests
from langchain.tools import tool
from settings import BaseToolSettings, BaseModelSettings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

settings = BaseToolSettings()
base_settings = BaseModelSettings()

class WeatherClient:
    def __init__(self, config: BaseToolSettings):
        self.api_key = config.weather_apikey
        self.api_url = config.weather_url

    def get_weather(self, city: str) -> dict | None:
        response = requests.get(f"{self.api_url}/v1/current.json?key={self.api_key}&q={city}")
        if response.status_code == 200:
            return response.json()
        return None

weather_client = WeatherClient(settings)

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

tools_dict: dict = {
    "get_weather": get_weather
}

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, verbose=True).bind_tools([get_weather])
system_msg = SystemMessage(
    """Your are a helpful assistant"""
)

def ask(question: str):
    human_msg = HumanMessage(question)
    messages: list[AIMessage | SystemMessage | HumanMessage | ToolMessage] = [system_msg, human_msg]

    answer_msg = model.invoke(messages)
    messages.append(answer_msg)

    for tool_call in answer_msg.tool_calls:
        selected_tool = tools_dict[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    final_msg = model.invoke(messages)
    print(messages + [final_msg])
    return final_msg.content

def main():
    print("Ask your question: ")
    question = input()
    answer = ask(question)
    print(answer)

if __name__ == '__main__':
    main()
