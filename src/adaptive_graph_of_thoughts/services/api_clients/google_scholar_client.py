from typing import Optional, List, Dict, Any
from loguru import logger
from pydantic import BaseModel, Field

from adaptive_graph_of_thoughts.config import Settings, GoogleScholarConfig
from .base_client import AsyncHTTPClient, APIRequestError, APIHTTPError, BaseAPIClientError

class GoogleScholarArticle(BaseModel):
    title: str
    link: Optional[str] = None
    snippet: Optional[str] = None
    publication_info: Optional[str] = None # e.g., "Nature, 2021"
    authors: Optional[str] = None # SerpApi often returns this as a single string or list of dicts
    source: str = Field(default="Google Scholar")

    # Links often found in SerpApi results
    related_articles_link: Optional[str] = None
    versions_link: Optional[str] = None
    citation_link: Optional[str] = None # Link to generate citation (e.g., SerpApi's cite link)

    cited_by_count: Optional[int] = None
    cited_by_link: Optional[str] = None

    # Raw result for further processing if needed
    raw_result: Optional[Dict[str, Any]] = None


class GoogleScholarClientError(BaseAPIClientError):
    """Custom error for GoogleScholarClient."""
    pass

class GoogleScholarClient:
    def __init__(self, settings: Settings):
        """
        Initializes the GoogleScholarClient with the provided settings.
        
        Validates the presence of Google Scholar API configuration in the settings. Raises a GoogleScholarClientError if required configuration is missing. Sets up the asynchronous HTTP client with the configured base URL and stores the API key.
        """
        if not settings.google_scholar or \
           not settings.google_scholar.base_url or \
           not settings.google_scholar.api_key:
            logger.error("Google Scholar configuration (base_url, api_key) is missing in settings.")
            raise GoogleScholarClientError("Google Scholar configuration is not properly set.")

        self.config: GoogleScholarConfig = settings.google_scholar
        # Assuming base_url in config is the full base path, e.g., "https://serpapi.com/search"
        # or "https://serpapi.com" and the endpoint "/search" will be specified in GET requests.
        # For consistency with how base_client is used, base_url should be up to the domain or common path.
        # If config.base_url is "https://serpapi.com/search", AsyncHTTPClient will append to it.
        # It's generally better if base_url = "https://serpapi.com" and endpoint = "/search"
        # The provided code seems to assume base_url might or might not include "/search"

        self.http_client = AsyncHTTPClient(
            base_url=self.config.base_url.rstrip('/')
        )
        self.api_key = self.config.api_key
        logger.info(f"GoogleScholarClient initialized. Configured base URL: {self.config.base_url}")

    async def close(self):
        """
        Closes the underlying asynchronous HTTP client.
        
        This should be called to release network resources when the client is no longer needed.
        """
        await self.http_client.close()

    async def __aenter__(self):
        # Propagate context management to the underlying httpx.AsyncClient
        """
        Enters the asynchronous context for the GoogleScholarClient.
        
        Ensures the underlying HTTP client is properly initialized for use within an async context manager.
        """
        await self.http_client.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Propagate context management to the underlying httpx.AsyncClient
        """
        Exits the asynchronous context manager, ensuring the underlying HTTP client is properly closed.
        """
        await self.http_client.client.__aexit__(exc_type, exc_val, exc_tb)

    def _parse_serpapi_response(self, response_json: Dict[str, Any]) -> List[GoogleScholarArticle]:
        """
        Parses a SerpApi JSON response and extracts Google Scholar articles.
        
        Iterates through the "organic_results" in the response, extracting relevant fields such as title, link, snippet, publication info, authors, citation counts, and related links. Returns a list of validated `GoogleScholarArticle` objects. Entries missing a title or with invalid data are skipped.
         
        Returns:
            A list of `GoogleScholarArticle` instances parsed from the response.
        """
        articles: List[GoogleScholarArticle] = []
        if "organic_results" not in response_json:
            search_parameters = response_json.get("search_parameters", {})
            query = search_parameters.get("q", "N/A")
            engine = search_parameters.get("engine", "N/A")
            logger.warning(f"No 'organic_results' in SerpApi response for query '{query}' on engine '{engine}'. Full response keys: {list(response_json.keys())}")
            if "error" in response_json:
                 logger.error(f"SerpApi returned an error: {response_json['error']}")
            return articles

        for res in response_json.get("organic_results", []):
            title = res.get("title")
            if not title:
                logger.warning("Skipping a Google Scholar result due to missing title.")
                continue

            publication_info_summary = res.get("publication_info", {}).get("summary")

            authors_data = res.get("publication_info", {}).get("authors")
            authors_str = None
            if isinstance(authors_data, list):
                authors_str = ", ".join(author.get("name") for author in authors_data if author.get("name"))
            elif isinstance(authors_data, str): # Fallback if it's already a string
                authors_str = authors_data

            cited_by_data = res.get("inline_links", {}).get("cited_by", {})
            cited_by_count = cited_by_data.get("total") # SerpApi returns this as int
            cited_by_link = cited_by_data.get("link")

            if cited_by_count is not None and not isinstance(cited_by_count, int):
                try: # Should already be an int from SerpApi, but good to be safe.
                    cited_by_count = int(cited_by_count)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse cited_by_count '{cited_by_count}' as int for title '{title}'")
                    cited_by_count = None

            versions_link = res.get("inline_links", {}).get("versions", {}).get("link")
            # SerpApi uses "related_pages_link" for "Related articles"
            related_articles_link = res.get("inline_links", {}).get("related_pages_link")
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
                "raw_result": res
            }
            try:
                articles.append(GoogleScholarArticle(**article_data))
            except Exception as e: # Catch Pydantic validation errors or other issues
                logger.error(f"Error creating GoogleScholarArticle for title '{title}': {e}. Data: {article_data}")

        return articles

    async def search(self, query: str, num_results: int = 10, lang: str = "en", region: str = "us") -> List[GoogleScholarArticle]:
        """
        Performs an asynchronous search on Google Scholar via SerpApi and returns a list of parsed articles.
        
        Args:
            query: The search query string.
            num_results: The maximum number of articles to retrieve (default is 10).
            lang: The language code for the search results (default is "en").
            region: The region code for the search results (default is "us").
        
        Returns:
            A list of GoogleScholarArticle objects representing the search results.
        
        Raises:
            GoogleScholarClientError: If the API request fails, the response cannot be decoded as JSON, or an unexpected error occurs.
        """
        logger.debug(f"Searching Google Scholar (via SerpApi) for query: '{query}', num_results: {num_results}")

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
        api_endpoint = "/search" # Default if base_url is like "https://serpapi.com"
        if self.http_client.base_url.endswith("/search"):
            api_endpoint = "" # Base_url already includes /search

        try:
            response = await self.http_client.get(api_endpoint, params=params)
            response_json = response.json() # This can raise JSONDecodeError (ValueError)

            articles = self._parse_serpapi_response(response_json)
            logger.info(f"Successfully retrieved and parsed {len(articles)} articles from Google Scholar API for query: '{query}'.")
            return articles

        except (APIHTTPError, APIRequestError) as e:
            logger.error(f"Google Scholar API request error for query '{query}': {e}")
            raise GoogleScholarClientError(f"Google Scholar API request failed: {e}") from e
        except ValueError as e: # Catches JSONDecodeError
            logger.error(f"Failed to decode JSON from Google Scholar API response for query '{query}': {e}")
            # Potentially log part of the response text if it's small and not sensitive
            # logger.debug(f"Non-JSON response text (first 100 chars): {getattr(response, 'text', '')[:100]}")
            raise GoogleScholarClientError(f"Google Scholar API JSON decode error: {e}") from e
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"Unexpected error during Google Scholar search for query '{query}': {e}", exc_info=True)
            raise GoogleScholarClientError(f"Unexpected error in Google Scholar search: {e}") from e

# Example Usage (for testing)
async def main_google_scholar_test():
    """
    Runs an asynchronous test of the Google Scholar client using current configuration.
    
    Attempts to load Google Scholar API settings, falling back to defaults if necessary. Skips execution if a valid API key is not provided. Performs a sample search query and logs details of retrieved articles or any errors encountered.
    """
    from adaptive_graph_of_thoughts.config import Settings, GoogleScholarConfig

    try:
        settings = Settings() # Load from settings.yaml or environment variables
        # Check if google_scholar config exists and has essential fields
        if not settings.google_scholar or not settings.google_scholar.api_key or not settings.google_scholar.base_url:
            logger.warning("Google Scholar config (api_key, base_url) not fully set in settings; test may use placeholders or fail.")
            # Fallback to a default mock if critical parts are missing, for structural testing.
            # However, an API key is essential for any real test against SerpApi.
            if not settings.google_scholar:
                 settings.google_scholar = GoogleScholarConfig(api_key="test_api_key_placeholder", base_url="https://serpapi.com/search")
            elif not settings.google_scholar.api_key:
                 settings.google_scholar.api_key = "test_api_key_placeholder"
            if not settings.google_scholar.base_url: # Ensure base_url is also set
                 settings.google_scholar.base_url="https://serpapi.com/search"

    except Exception as e: # Broad exception for issues loading settings
        logger.warning(f"Could not load global settings ({e}), using default mock Google Scholar config for testing.")
        settings = Settings( # Create a new Settings object with a default GoogleScholarConfig
            google_scholar=GoogleScholarConfig(api_key="YOUR_SERPAPI_KEY_HERE", base_url="https://serpapi.com/search")
        )

    # Critical check: Actual API key is needed to run a meaningful test.
    gs_config = settings.google_scholar
    if not gs_config or not gs_config.api_key or \
       "YOUR_SERPAPI_KEY_HERE" in gs_config.api_key or \
       "placeholder" in gs_config.api_key:
        logger.error("An actual SerpApi API key for Google Scholar is required to run this test. Please configure it in settings.yaml or environment variables (GOOGLE_SCHOLAR__API_KEY). Skipping test.")
        return

    logger.info(f"Attempting test with Google Scholar API Key: ...{gs_config.api_key[-4:] if gs_config.api_key else 'N/A'}")

    async with GoogleScholarClient(settings=settings) as client:
        try:
            query = "quantum computing applications in drug discovery"
            articles = await client.search(query, num_results=2) # Keep num_results low for testing
            if articles:
                logger.info(f"Found {len(articles)} articles for '{query}':")
                for i, article in enumerate(articles):
                    logger.info(f"  Article #{i+1}:")
                    logger.info(f"    Title: {article.title}")
                    logger.info(f"    Link: {article.link}")
                    logger.info(f"    Authors: {article.authors}")
                    logger.info(f"    Publication: {article.publication_info}")
                    logger.info(f"    Cited by: {article.cited_by_count}")
                    logger.info(f"    Snippet: {article.snippet[:100]}..." if article.snippet else "N/A")
            else:
                logger.info(f"No articles found for query: {query}")

        except GoogleScholarClientError as e:
            logger.error(f"Google Scholar client test error: {e}")
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during Google Scholar client test: {e}", exc_info=True)

if __name__ == "__main__":
    import asyncio
    # To run this test:
    # 1. Ensure you have a config/settings.yaml or relevant environment variables for google_scholar.
    #    Example minimal settings.yaml:
    #    google_scholar:
    #      base_url: "https://serpapi.com/search" # Or "https://serpapi.com"
    #      api_key: "YOUR_SERPAPI_API_KEY"
    # 2. Uncomment the line below.
    # asyncio.run(main_google_scholar_test())
    pass
