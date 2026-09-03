from datetime import datetime
from typing import Type, Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from helpers import CrawlHelper

supported_cinemas = {
    "cineplanet": lambda param = None: "https://www.cineplanet.com.pe/peliculas",
    "cinemark": lambda param=datetime.now().strftime("YYYY-MM-DD"): f"https://www.cinemark-peru.com/elegir-pelicula?date={param}",
}

def validate_cinema(cinema: str) -> None:
    if cinema not in supported_cinemas:
        raise ValueError(f"Cinema '{cinema}' is not supported.")


class GetCurrentDateInput(BaseModel):
    date_format: str = Field(description="Python datetime format string. Defaults to YYYY-MM-DD.")

class GetCurrentDateTool(BaseTool):
    name: str = "get_current_date"
    description: str = "Returns the current date formatted according to the requested format"
    args_schema: Type[BaseModel] = GetCurrentDateInput

    def _run(self, date_format: str = "YYYY-MM-DD") -> str:
        """Synchronous execution logic."""
        return datetime.now().strftime(date_format)

    async def _arun(self, date_format: str = "YYYY-MM-DD") -> str:
        """Asynchronous execution logic (optional)."""
        return datetime.now().strftime(date_format)


class SearchCinemaInfoInput(BaseModel):
    cinema: str = Field(description="Name of the cinema whose website should be crawled to retrieve movies, showtimes, locations, prices, and other relevant cinema information.")
    date: Optional[str] = Field(
        default=None,
        description="Optional date to search for cinema information, especially movie availability and showtimes. Format: YYYY-MM-DD (e.g., 2026-09-02)."
    )

class SearchCinemaInfoTool(BaseTool):
    name: str = "search_cinema_info"
    description: str = (
        "Use this tool to retrieve up-to-date information from a specific "
        "cinema's website. It can find currently playing movies, movie "
        "availability, showtimes, cinema locations, prices, and other "
        "cinema-related information. Provide a date when the user asks about "
        "movies or showtimes for a specific day. The date must use the "
        "YYYY-MM-DD format. Returns the relevant extracted content in clean "
        "Markdown format."
    )
    args_schema: Type[BaseModel] = SearchCinemaInfoInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.__crawler = CrawlHelper(content_filter="bm25")

    def _run(self, cinema: str, date: str) -> str:
        """Synchronous execution logic."""
        validate_cinema(cinema)

        url = supported_cinemas[cinema](date)
        return self.__crawler.run(url)

    async def _arun(self, cinema: str, date: str) -> str:
        """Asynchronous execution logic (optional)."""
        validate_cinema(cinema)

        url = supported_cinemas[cinema](date)
        return self.__crawler.arun(url)


class SearchFilmReviewInput(BaseModel):
    film_name: str = Field(
        description=(
            "Name of the film to search for. Use the film title provided "
            "by the user to find matching film results on Rotten Tomatoes."
        )
    )

class SearchFilmReviewTool(BaseTool):
    name: str = "search_film_review"
    description: str = (
        "Searches Rotten Tomatoes using a film name and returns matching "
        "film results and relevant information that can be used to identify "
        "the correct film. This tool is for film discovery and selection; "
        "it does not necessarily return the actual film review. After the "
        "correct film has been identified, use get_film_review to retrieve "
        "the specific film review."
    )
    args_schema: Type[BaseModel] = SearchFilmReviewInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.__crawler = CrawlHelper()
        self.__website = "https://www.rottentomatoes.com/search?search={film_name}"

    def _run(self, film_name: str) -> str:
        """Synchronous execution logic."""

        url = self.__website.replace("{film_name}", film_name)
        return self.__crawler.run(url)

    async def _arun(self, film_name: str) -> str:
        """Asynchronous execution logic (optional)."""

        url = self.__website.replace("{film_name}", film_name)
        return self.__crawler.arun(url)


class GetFilmReviewInput(BaseModel):
    film_url: str = Field(
        description=(
            "URL of the specific film page identified by the film search. "
            "Use the URL returned by the search_film_review tool."
        )
    )

class GetFilmReviewTool(BaseTool):
    name: str = "get_film_review"
    description: str = (
        "Retrieves the actual film review from Rotten Tomatoes for a "
        "specific film. Use this tool after search_film_review has identified "
        "the correct film. Provide the film page URL returned by the search "
        "tool. This tool returns the film's review content, including the "
        "review text and relevant critic information, rather than searching "
        "for candidate films."
    )
    args_schema: Type[BaseModel] = GetFilmReviewInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.__crawler = CrawlHelper()

    def _run(self, film_url: str) -> str:
        return self.__crawler.run(film_url)

    async def _arun(self, film_url: str) -> str:
        return await self.__crawler.arun(film_url)