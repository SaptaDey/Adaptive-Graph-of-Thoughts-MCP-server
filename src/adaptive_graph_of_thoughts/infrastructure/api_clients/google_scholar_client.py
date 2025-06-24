from typing import Any

from loguru import logger
from pydantic import BaseModel

from ...config import GoogleScholarConfig, LegacyConfig
from .base_client import (
    APIHTTPError,
    APIRequestError,
    AsyncHTTPClient,
    BaseAPIClientError,
)


class GoogleScholarArticle(BaseModel):
    title: str = ""
    link: str = ""
    snippet: str = ""
    publication_info: str = ""
    authors: str = ""
    source: str = "Google Scholar"
    related_articles_link: str = ""
    versions_link: str = ""
    citation_link: str = ""
    cited_by_count: int = 0
    cited_by_link: str = ""
    raw_result: str = ""


class GoogleScholarClientError(BaseAPIClientError):
    """Base error for GoogleScholarClient."""

    pass


class UnexpectedResponseStructureError(GoogleScholarClientError):
    """Raised when the API response lacks the expected structure."""

    pass


class GoogleScholarClient:
    def __init__(self, settings: LegacyConfig):
        if not settings.google_scholar:
            logger.error("Google Scholar configuration is missing in settings.")
            raise GoogleScholarClientError(
                "Google Scholar configuration is not properly set."
            )

        self.config: GoogleScholarConfig = settings.google_scholar

        # Use a default base URL for Google Scholar scraping
        self.http_client = AsyncHTTPClient(base_url="https://scholar.google.com")

        logger.info(
            f"GoogleScholarClient initialized with max_results: {self.config.max_results}"
        )

    async def close(self):
        await self.http_client.close()

    async def __aenter__(self):
        # Propagate context management to the underlying httpx.AsyncClient
        await self.http_client.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Propagate context management to the underlying httpx.AsyncClient
        await self.http_client.client.__aexit__(exc_type, exc_val, exc_tb)

    def _parse_serpapi_response(
        self, response_json: dict[str, Any]
    ) -> list[GoogleScholarArticle]:
        articles: list[GoogleScholarArticle] = []
        if "organic_results" not in response_json:
            search_parameters = response_json.get("search_parameters", {})
            query = search_parameters.get("q", "N/A")
            engine = search_parameters.get("engine", "N/A")
            logger.warning(
                f"No 'organic_results' in SerpApi response for query '{query}' on engine '{engine}'. Full response keys: {list(response_json.keys())}"
            )
            if "error" in response_json:
                logger.error(f"SerpApi returned an error: {response_json['error']}")
            raise UnexpectedResponseStructureError(
                "SerpApi response missing 'organic_results' key"
            )

        for res in response_json.get("organic_results", []):
            title = res.get("title")
            if not title:
                logger.warning("Skipping a Google Scholar result due to missing title.")
                continue

            publication_info_summary = res.get("publication_info", {}).get("summary")

            authors_data = res.get("publication_info", {}).get("authors")
            authors_str = None
            if isinstance(authors_data, list):
                authors_str = ", ".join(
                    author.get("name") for author in authors_data if author.get("name")
                )
            elif isinstance(authors_data, str):  # Fallback if it's already a string
                authors_str = authors_data

            cited_by_data = res.get("inline_links", {}).get("cited_by", {})
            cited_by_count = cited_by_data.get("total")  # SerpApi returns this as int
            cited_by_link = cited_by_data.get("link")

            if cited_by_count is not None and not isinstance(cited_by_count, int):
                try:  # Should already be an int from SerpApi, but good to be safe.
                    cited_by_count = int(cited_by_count)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not parse cited_by_count '{cited_by_count}' as int for title '{title}'"
                    )
                    cited_by_count = None

            versions_link = res.get("inline_links", {}).get("versions", {}).get("link")
            # SerpApi uses "related_pages_link" for "Related articles"
            related_articles_link = res.get("inline_links", {}).get(
                "related_pages_link"
            )
            citation_link = res.get("inline_links", {}).get("serpapi_cite_link")

            article_data = {
                "title": title,
                "link": res.get("link"),
                "snippet": res.get("snippet"),
                "publication_info": publication_info_summary,
                "authors": authors_str,
                "related_articles_link": related_articles_link,
                "versions_link": versions_link,
                "citation_link": citation_link,
                "cited_by_count": cited_by_count,
                "cited_by_link": cited_by_link,
                "raw_result": res,
            }
            try:
                articles.append(GoogleScholarArticle(**article_data))
            except Exception as e:  # Catch Pydantic validation errors or other issues
                logger.error(
                    f"Error creating GoogleScholarArticle for title '{title}': {e}. Data: {article_data}"
                )

        return articles

    async def search(
        self, query: str, num_results: int = 10, lang: str = "en", region: str = "us"
    ) -> list[GoogleScholarArticle]:
        logger.debug(
            f"Searching Google Scholar (via SerpApi) for query: '{query}', num_results: {num_results}"
        )

        params = {
            "api_key": self.api_key,
            "engine": "google_scholar",
            "q": query,
            "hl": lang,
            "gl": region,
            "num": str(num_results),
        }

        # Determine the correct endpoint based on how base_url is configured.
        # If self.config.base_url is "https://serpapi.com", endpoint is "/search".
        # If self.config.base_url is "https://serpapi.com/search", endpoint is "".
        api_endpoint = "/search"  # Default if base_url is like "https://serpapi.com"
        if self.http_client.base_url.endswith("/search"):
            api_endpoint = ""  # Base_url already includes /search

        try:
            response = await self.http_client.get(api_endpoint, params=params)
            response_json = response.json()  # This can raise JSONDecodeError

            articles = self._parse_serpapi_response(response_json)
            logger.info(
                f"Successfully retrieved and parsed {len(articles)} articles from Google Scholar API for query: '{query}'."
            )
            return articles

        except UnexpectedResponseStructureError:
            raise
        except (APIHTTPError, APIRequestError) as e:
            logger.error(f"Google Scholar API request error for query '{query}': {e}")
            raise GoogleScholarClientError(
                f"Google Scholar API request failed: {e}"
            ) from e
        except ValueError as e:  # Catches JSONDecodeError
            logger.error(
                f"Failed to decode JSON from Google Scholar API response for query '{query}': {e}"
            )
            # Potentially log part of the response text if it's small and not sensitive
            # logger.debug(f"Non-JSON response text (first 100 chars): {getattr(response, 'text', '')[:100]}")
            raise GoogleScholarClientError(
                f"Google Scholar API JSON decode error: {e}"
            ) from e
        except Exception as e:  # Catch-all for other unexpected errors
            logger.error(
                f"Unexpected error during Google Scholar search for query '{query}': {e}",
                exc_info=True,
            )
            raise GoogleScholarClientError(
                f"Unexpected error in Google Scholar search: {e}"
            ) from e


# Example Usage (for testing)
async def main_google_scholar_test():
    from adaptive_graph_of_thoughts.config import GoogleScholarConfig, Settings

    try:
        settings = Settings()  # Load from settings.yaml or environment variables
        # Check if google_scholar config exists
        if not settings.google_scholar:
            logger.warning(
                "Google Scholar config not set in settings; using defaults for testing."
            )
            settings.google_scholar = GoogleScholarConfig(max_results=10)

    except Exception as e:  # Broad exception for issues loading settings
        logger.warning(
            f"Could not load global settings ({e}), using default mock Google Scholar config for testing."
        )
        settings = (
            Settings(  # Create a new Settings object with a default GoogleScholarConfig
                google_scholar=GoogleScholarConfig(max_results=10)
            )
        )

    # Test configuration
    gs_config = settings.google_scholar
    logger.info(
        f"Testing with Google Scholar config - max_results: {gs_config.max_results}"
    )

    async with GoogleScholarClient(settings=settings) as client:
        try:
            query = "quantum computing applications in drug discovery"
            articles = await client.search(
                query, num_results=2
            )  # Keep num_results low for testing
            if articles:
                logger.info(f"Found {len(articles)} articles for '{query}':")
                for i, article in enumerate(articles):
                    logger.info(f"  Article #{i + 1}:")
                    logger.info(f"    Title: {article.title}")
                    logger.info(f"    Link: {article.link}")
                    logger.info(f"    Authors: {article.authors}")
                    logger.info(f"    Publication: {article.publication_info}")
                    logger.info(f"    Cited by: {article.cited_by_count}")
                    logger.info(
                        f"    Snippet: {article.snippet[:100]}..."
                        if article.snippet
                        else "N/A"
                    )
            else:
                logger.info(f"No articles found for query: {query}")

        except GoogleScholarClientError as e:
            logger.error(f"Google Scholar client test error: {e}")
        except Exception as e:  # Catch any other unexpected errors
            logger.error(
                f"An unexpected error occurred during Google Scholar client test: {e}",
                exc_info=True,
            )


if __name__ == "__main__":
    # To run this test:
    # 1. Ensure you have a config/settings.yaml or relevant environment variables for google_scholar.
    #    Example minimal settings.yaml:
    #    google_scholar:
    #      max_results: 10
    #      rate_limit_delay: 1.0
    # 2. Uncomment the line below.
    # asyncio.run(main_google_scholar_test())
    pass
