import requests
import serpapi
import os

client = serpapi.Client(api_key=os.environ["SERPAPI_API_KEY"])

def search_serpapi(query: str, k: int=3) -> list[str]:
    """Search the web for information about a topic."""
    res = client.search(q=query, engine="google", hl="en", gl="us")
    results = res["organic_results"] if len(res["organic_results"]) <= k else res["organic_results"][:k]
    return [ item["link"] for item in results]

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