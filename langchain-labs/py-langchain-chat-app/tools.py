import requests
from langchain.tools import tool
from langchain_community.utilities import ArxivAPIWrapper, SerpAPIWrapper
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai.chat_models import ChatOpenAI
from settings import BaseToolSettings, BaseModelSettings

class WeatherClient:
    def __init__(self, config: BaseToolSettings):
        self.api_key = config.weather_apikey
        self.api_url = config.weather_url

    def get_weather(self, city: str) -> dict | None:
        response = requests.get(f"{self.api_url}/v1/current.json?key={self.api_key}&q={city}")
        if response.status_code == 200:
            return response.json()
        return None


tools_settings = BaseToolSettings()
model_settings = BaseModelSettings()

weather_client = WeatherClient(tools_settings)
serpapi = SerpAPIWrapper()
arxiv = ArxivAPIWrapper()

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

@tool
def add(a: int | float, b: int | float) -> int | float:
    """Adds a and b

    Args:
        a: The first number
        b: The second number
    """
    return a + b

@tool
def sub(a: int | float, b: int | float) -> int | float:
    """Subtracts a and b
    Args:
        a: The first number
        b: The second number
    """
    return a - b

@tool
def mul(a: int | float, b: int | float) -> int | float:
    """Multiplies a and b

    Args:
        a: The first number
        b: The second number
    """
    return a * b

@tool
def div(a: int | float, b: int | float) -> int | float:
    """Divides a and b

    Args:
        a: The first number
        b: The second number
    """
    return a / b

@tool
def search_web(query: str) -> str:
    """Search the web for current information.

    Args:
        query: term to search for
    """
    return serpapi.run(query)

@tool
def get_papers(query: str) -> str:
    """Search on ArXiv for deep research and investigation.

    Args:
        query: term to search for
    """
    return arxiv.run(query)

tools_registry: dict = {
    "get_weather": get_weather,
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
    "search_web": search_web,
    "get_papers": get_papers,
}

MAX_ITERATIONS = 20

class ToolsPoweredChatOpenAI:
    def __init__(self, settings: BaseModelSettings):
        self.model_settings = settings
        self.model = (ChatOpenAI(model=self.model_settings.model_name, temperature=self.model_settings.temperature, verbose=True)
         .bind_tools([get_weather, add, sub, mul, div, search_web, get_papers]))

        # LangChain chat models can expose a dictionary of supported features and capabilities through a profile attribute:
        # print(self.model.profile)

        # Helpful assistant Prompt
        # self.system_msg = SystemMessage("""Your are a helpful assistant""")

        # Calculator User expect Prompt
        self.system_msg = SystemMessage("""
        # Goal:
        - You are bad at math but are an expert at using a calculator
        
        # Tools
        - **add**: Perform addition operation
        - **sub**: Perform subtraction operation
        - **mul**: Perform multiplication operation
        - **div**: Perform division operation
        
        # Instructions
        - You must use the tools to perform math operations
        - Then you must break down the task into multiple steps and explain intermediate result.
        - This is the structure of the result: (1) Steps, (2) Final Result.
        
        # Input
        - User query
        
        # Output
        - Steps and result in Markdown format
        """)

        # Researcher Expert Prompt
        # self.system_msg = SystemMessage("""
        # # Goal:
        # - Perform a deep research
        #
        # # Tools
        # - **search_web**: Got web pages on the Web
        # - **get_papers**: Got papers from ArXiv
        #
        # # Instructions
        # - You must follow the user topic in the Web and ArXiv
        # - Then you must produce short review including citations in IEEE format.
        # - This is the structure of the review: (1) Summary, (2) Background and concepts, (3) Trends and challenges, (4) Conclusions, (5) References.
        #
        # # Input
        # - User query
        #
        # # Output
        # - Review and conclusions in markdown
        # """)

    @staticmethod
    def __execute_tools(tool_calls) -> list:
        output = []
        for tool_call in tool_calls:
            selected_tool = tools_registry[tool_call["name"].lower()]
            tool_output = selected_tool.invoke(tool_call["args"])
            output.append(
                ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call["id"]
                )
            )
        return output

    def __ask(self, question: str):
        human_msg = HumanMessage(question)
        messages: list[AIMessage | SystemMessage | HumanMessage | ToolMessage] = [self.system_msg, human_msg]

        for _ in range(MAX_ITERATIONS):
            answer_msg = self.model.invoke(messages)
            messages.append(answer_msg)

            if not answer_msg.tool_calls:
                break

            messages.extend(ToolsPoweredChatOpenAI.__execute_tools(answer_msg.tool_calls))
        else:
            raise RuntimeError("Chat exceeded maximum tool interactions")

        print(messages)
        return answer_msg.content

    def initialize(self):
        print("Ask your question: ")
        question = input()
        answer = self.__ask(question)
        print(answer)


main_chat = ToolsPoweredChatOpenAI(model_settings)

if __name__ == '__main__':
    main_chat.initialize()
