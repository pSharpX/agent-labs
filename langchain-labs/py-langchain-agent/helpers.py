import asyncio
from typing import Literal

import requests

from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, LLMContentFilter, LLMConfig, BM25ContentFilter, \
    PruningContentFilter

from pathlib import Path
from urllib.parse import urlparse

from langchain.tools import tool

from settings import BaseToolSettings

ContentFilter = Literal["llm", "bm25", "prune"]

def get_content_filter(content_filter: ContentFilter | None, settings: dict):
    if content_filter == "llm":
        if "provider" not in settings or "api_token" not in settings:
            raise TypeError("Invalid or missing argument: provider, api_token")

        return LLMContentFilter(
            llm_config = LLMConfig(
                provider=settings["provider"],
                api_token=settings["api_token"]
            ), #or use environment variable
            instruction="""
            Focus on extracting the core cinema and entertainment content.
            Include:
            - Movie and TV show information
            - Actors, directors, and other key people
            - Reviews, ratings, and recommendations
            - Release dates, trailers, and upcoming releases
            - Relevant entertainment news and announcements
            Exclude:
            - Navigation elements
            - Advertisements
            - Sidebars and unrelated recommendations
            - Footer content
            Format the output as clean markdown with proper code blocks and headers.
            """,
            chunk_token_threshold=4096,  # Adjust based on your needs
            verbose=True
        )
    elif content_filter == "bm25":
        return BM25ContentFilter(
            user_query="extract main content",
            bm25_threshold=1.2,
            language="english"
        )
    elif content_filter == "prune":
        return PruningContentFilter(
            threshold=0.5,
            threshold_type="fixed",  # or "dynamic"
            min_word_threshold=50
        )
    return None

class CrawlHelper:
    def __init__(self, content_filter: ContentFilter = None, settings: dict = None):
        self.__content_filter = get_content_filter(content_filter, settings)
        self.__md_generator = DefaultMarkdownGenerator(
            content_source="fit_html", # cleaned_html
            content_filter=self.__content_filter,
            options={
                "ignore_links": True,
                "escape_html": True,
                "body_width": 80
            }
        )

        self.__browser_config = BrowserConfig(headless=True, java_script_enabled=True)
        self.__run_config = CrawlerRunConfig(
            markdown_generator=self.__md_generator,
            wait_for="css:.app",
            delay_before_return_html=3.0,
            process_iframes=True,
            js_code="""
                window.scrollTo(0, document.body.scrollHeight);
            """,
            scan_full_page=True
        )

    def run(self, url: str) -> str:
        async def __run():
            async with AsyncWebCrawler(config=self.__browser_config) as crawler:
                result = await crawler.arun(url, config=self.__run_config)
                print("RESULTS HERE ***********")
                print(result)
                print("RESULTS HERE ***********")
                if result.success:
                    print("MARKDOWN HERE ***********")
                    print(result.markdown)
                    return result.markdown

                raise ValueError(f"Crawl failed: {result.error_message}")

        return asyncio.run(__run())

    async def arun(self, url: str) -> str:
        async with AsyncWebCrawler(config=self.__browser_config) as crawler:
            result = await crawler.arun(url, config=self.__run_config)
            if result.success:
                return result.markdown

            raise ValueError(f"Crawl failed: {result.error_message}")



class WeatherClient:
    def __init__(self, config: BaseToolSettings):
        self.api_key = config.weather_apikey
        self.api_url = config.weather_url

    def get_weather(self, city: str) -> dict | None:
        response = requests.get(f"{self.api_url}/v1/current.json?key={self.api_key}&q={city}")
        if response.status_code == 200:
            return response.json()
        return None


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