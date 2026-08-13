from pathlib import Path
from urllib.parse import urlparse


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

    if extension == ".pdf":
        return "pdf"

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