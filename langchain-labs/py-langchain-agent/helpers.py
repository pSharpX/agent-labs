import requests
import serpapi
import os

from pathlib import Path
from urllib.parse import urlparse

from langchain.tools import tool
from langchain_community.utilities import ArxivAPIWrapper

from settings import BaseToolSettings

#client = serpapi.Client(api_key=os.environ["SERPAPI_API_KEY"])

arxiv = ArxivAPIWrapper()


class WebScrapper:
    def __init__(self, api_url: str, api_key: str):
        pass

    def retrieve(self):
        pass


class WeatherClient:
    def __init__(self, config: BaseToolSettings):
        self.api_key = config.weather_apikey
        self.api_url = config.weather_url

    def get_weather(self, city: str) -> dict | None:
        response = requests.get(f"{self.api_url}/v1/current.json?key={self.api_key}&q={city}")
        if response.status_code == 200:
            return response.json()
        return None

#def search_serpapi(query: str, k: int=3) -> list[str]:
#    """Search the web for information about a topic."""
#    res = client.search(q=query, engine="google", hl="en", gl="us")
#    results = res["organic_results"] if len(res["organic_results"]) <= k else res["organic_results"][:k]
#    return [ item["link"] for item in results]

def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "format": "json",
        },
        headers={
            "User-Agent": "LLMWikipediaSearch/1.0"
        },
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("query", {}).get("search", [])
    if not results:
        return f"No Wikipedia results found for: {query}"

    return "\n".join(
        f"- {result['title']}: {result['snippet']}"
        for result in results
    )

@tool
def get_papers(query: str) -> str:
    """Search on ArXiv for deep research and investigation.

    Args:
        query: term to search for
    """
    return arxiv.run(query)

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


def classify_resource(resource: str) -> str:
    """
    Classify a resource as 'web', 'pdf', or 'text'.

    Args:
        resource: URL or local file path.

    Returns:
        One of: 'url', 'pdf', 'text'

    Raises:
        ValueError: If the resource type cannot be determined.
    """
    resource = resource.strip()

    # Check if it is a URL
    parsed = urlparse(resource)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return "web"

    # Check local file extension
    path = Path(resource)
    extension = path.suffix.lower()

    # Common docs-based files
    docs_extensions = {
        ".doc",
        ".docx",
        ".docm",
        ".ppt",
        ".pps",
        ".pot",
        ".pptx",
        ".pptm",
        ".ppsx",
        ".ppsm",
        ".xls",
        ".xlsx",
        ".xlsm",
        ".xlsb",
        ".odt",
        ".ods",
        ".odp",
        ".csv",
        ".pdf",
    }
    if extension in docs_extensions:
        return extension[1:]

    # Common text-based files
    text_extensions = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".yaml",
        ".yml",
    }

    if extension in text_extensions:
        return "text"

    raise ValueError(f"Unsupported resource type: {resource}")