from typing import Optional, List, Dict, Any
from loguru import logger
from pydantic import BaseModel, Field

from adaptive_graph_of_thoughts.config import Config, ExaSearchConfig # Changed Settings to Config
from .base_client import AsyncHTTPClient, APIRequestError, APIHTTPError, BaseAPIClientError

class ExaArticleResult(BaseModel):
    id: str # Exa's result ID
    url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[str] = Field(default=None, alias="publishedDate") # Note alias for 'publishedDate'
    score: Optional[float] = None # Relevance score from Exa
    highlights: List[str] = Field(default_factory=list) # Exa can return highlights
    raw_result: Optional[Dict[str, Any]] = None # Store the raw result

class ExaSearchClientError(BaseAPIClientError):
    """Custom error for ExaSearchClient."""
    pass

class ExaSearchClient:
    def __init__(self, main_config: Config): # Changed settings: Settings to main_config: Config
        if not main_config.exa_search or \
           not main_config.exa_search.base_url or \
           not main_config.exa_search.api_key:
            logger.error("Exa Search configuration (base_url, api_key) is missing in settings.")
            raise ExaSearchClientError("Exa Search configuration is not properly set.")

        self.config: ExaSearchConfig = main_config.exa_search
        self.api_key = self.config.api_key

        # Default headers for Exa API
        default_headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"AdaptiveGraphOfThoughtsClient/1.0 (ExaSearchClient)", # More specific User-Agent
        }

        self.http_client = AsyncHTTPClient(
            base_url=self.config.base_url.rstrip('/'),
            default_headers=default_headers # Pass default headers to the base client
        )
        logger.info(f"ExaSearchClient initialized for base URL: {self.config.base_url}")

    async def close(self):
        await self.http_client.close()

    async def __aenter__(self):
        # Ensure the underlying httpx.AsyncClient's context is entered
        await self.http_client.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Ensure the underlying httpx.AsyncClient's context is exited
        await self.http_client.client.__aexit__(exc_type, exc_val, exc_tb)

    def _parse_exa_response(self, response_json: Dict[str, Any]) -> List[ExaArticleResult]:
        articles: List[ExaArticleResult] = []
        if "results" not in response_json:
            logger.warning("No 'results' key found in Exa API response.")
            if "error" in response_json: # Exa might return an error message at the top level
                 logger.error(f"Exa API returned an error: {response_json['error']}")
            return articles

        for res in response_json.get("results", []):
            if not res.get("id"):
                logger.warning("Skipping Exa result due to missing ID.")
                continue

            article_data = {
                "id": res.get("id"),
                "url": res.get("url"),
                "title": res.get("title"),
                "author": res.get("author"),
                "publishedDate": res.get("publishedDate"),
                "score": res.get("score"),
                "highlights": res.get("highlights", []), # Default to empty list if not present
                "raw_result": res
            }
            try:
                articles.append(ExaArticleResult(**article_data))
            except Exception as e: # Catch Pydantic validation errors or other issues
                logger.error(f"Error creating ExaArticleResult for ID '{res.get('id')}': {e}. Data: {article_data}")
        return articles

    async def search(
        self,
        query: str,
        num_results: int = 10,
        type: str = "neural", # "neural" or "keyword"
        use_autoprompt: bool = False,
        category: Optional[str] = None, # e.g. "article"
        start_published_date: Optional[str] = None, # YYYY-MM-DD
        end_published_date: Optional[str] = None # YYYY-MM-DD
    ) -> List[ExaArticleResult]:
        logger.debug(f"Searching Exa for query: '{query}', type: {type}, num_results: {num_results}, autoprompt: {use_autoprompt}")

        stripped_query = query.strip() # Added strip
        payload: Dict[str, Any] = {
            "query": stripped_query, # Use stripped query
            "num_results": num_results,
            "type": type,
        }
        # Exa's API expects use_autoprompt only if true, or it might error if 'false' is explicitly sent
        # depending on their exact API spec version. Some APIs prefer omitting false booleans.
        # For Exa, it's safer to include it as per their examples.
        payload["use_autoprompt"] = use_autoprompt

        if category:
            payload["category"] = category
        if start_published_date:
            payload["startPublishedDate"] = start_published_date # API expects camelCase
        if end_published_date:
            payload["endPublishedDate"] = end_published_date   # API expects camelCase

        # The http_client already has default headers including x-api-key
        try:
            response = await self.http_client.post("/search", json_data=payload)
            response_json = response.json()
            articles = self._parse_exa_response(response_json)
            logger.info(f"Successfully retrieved {len(articles)} results from Exa search for query: '{query}'.")
            return articles
        except (APIHTTPError, APIRequestError) as e:
            logger.error(f"Exa API search error for query '{query}': {e}")
            # If e has response content, it might be logged by base_client or here
            if isinstance(e, APIHTTPError) and e.response_content:
                 logger.error(f"Exa error response content: {e.response_content}")
            raise ExaSearchClientError(f"Exa API search request failed: {e}") from e
        except ValueError as e: # JSONDecodeError
            logger.error(f"Failed to decode JSON from Exa API search for query '{query}': {e}")
            raise ExaSearchClientError(f"Exa API search JSON decode error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Exa search for query '{query}': {e}", exc_info=True)
            raise ExaSearchClientError(f"Unexpected error in Exa search: {e}") from e

    async def find_similar(
        self,
        url: str,
        num_results: int = 10,
        start_published_date: Optional[str] = None, # YYYY-MM-DD
        end_published_date: Optional[str] = None # YYYY-MM-DD
    ) -> List[ExaArticleResult]:
        logger.debug(f"Finding similar content on Exa for URL: '{url}', num_results: {num_results}")

        payload: Dict[str, Any] = {
            "url": url,
            "num_results": num_results,
        }
        if start_published_date:
            payload["startPublishedDate"] = start_published_date # API expects camelCase
        if end_published_date:
            payload["endPublishedDate"] = end_published_date   # API expects camelCase

        try:
            response = await self.http_client.post("/find_similar", json_data=payload)
            response_json = response.json()
            articles = self._parse_exa_response(response_json)
            logger.info(f"Successfully retrieved {len(articles)} similar results from Exa for URL: '{url}'.")
            return articles
        except (APIHTTPError, APIRequestError) as e:
            logger.error(f"Exa API find_similar error for URL '{url}': {e}")
            if isinstance(e, APIHTTPError) and e.response_content:
                 logger.error(f"Exa error response content: {e.response_content}")
            raise ExaSearchClientError(f"Exa API find_similar request failed: {e}") from e
        except ValueError as e: # JSONDecodeError
            logger.error(f"Failed to decode JSON from Exa API find_similar for URL '{url}': {e}")
            raise ExaSearchClientError(f"Exa API find_similar JSON decode error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Exa find_similar for URL '{url}': {e}", exc_info=True)
            raise ExaSearchClientError(f"Unexpected error in Exa find_similar: {e}") from e

# Example Usage (for testing)
async def main_exa_search_test():
    from adaptive_graph_of_thoughts.config import Config, ExaSearchConfig # Changed Settings to Config

    try:
        main_app_config = Config() # Changed settings = Settings() to main_app_config = Config()
        if not main_app_config.exa_search or not main_app_config.exa_search.api_key or not main_app_config.exa_search.base_url:
            logger.warning("Exa Search config (api_key, base_url) not fully set in settings; test may use placeholders or fail.")
            if not main_app_config.exa_search:
                main_app_config.exa_search = ExaSearchConfig(api_key="test_api_key_placeholder", base_url="https://api.exa.ai")
            elif not main_app_config.exa_search.api_key:
                main_app_config.exa_search.api_key = "test_api_key_placeholder"
            if not main_app_config.exa_search.base_url:
                 main_app_config.exa_search.base_url="https://api.exa.ai"


    except Exception as e:
        logger.warning(f"Could not load settings ({e}), using mock Exa Search config for testing.")
        main_app_config = Config( # Changed settings = Settings()
            exa_search=ExaSearchConfig(api_key="YOUR_EXA_API_KEY_HERE", base_url="https://api.exa.ai")
        )

    exa_config_section = main_app_config.exa_search # Renamed exa_config to exa_config_section
    if not exa_config_section or not exa_config_section.api_key or \
       "YOUR_EXA_API_KEY_HERE" in exa_config_section.api_key or \
       "placeholder" in exa_config_section.api_key:
        logger.error("Actual Exa API key is required to run this test. Please configure it in settings.yaml or environment variables (EXA_SEARCH__API_KEY). Skipping test.")
        return

    logger.info(f"Attempting Exa client test with API Key: ...{exa_config_section.api_key[-4:] if exa_config_section.api_key else 'N/A'}")

    async with ExaSearchClient(main_config=main_app_config) as client: # Changed settings=settings to main_config=main_app_config
        try:
            # Test Search
            search_query = "latest advancements in battery technology"
            logger.info(f"--- Testing Exa Search for query: '{search_query}' ---")
            search_results = await client.search(search_query, num_results=2, type="neural", use_autoprompt=True)

            if search_results:
                logger.info(f"Found {len(search_results)} Exa search results for '{search_query}':")
                for i, res in enumerate(search_results):
                    logger.info(f"  Result #{i+1}: ID: {res.id}, Title: {res.title}, URL: {res.url}, Score: {res.score}")
                    if res.highlights:
                        logger.info(f"    Highlight (first): {res.highlights[0][:100]}...")

                # Test Find Similar with the first result's URL if available
                first_url_for_similar = search_results[0].url
                if first_url_for_similar:
                    logger.info(f"--- Testing Exa Find Similar for URL: {first_url_for_similar} ---")
                    similar_results = await client.find_similar(first_url_for_similar, num_results=2)
                    if similar_results:
                        logger.info(f"Found {len(similar_results)} Exa similar results for '{first_url_for_similar}':")
                        for j, sim_res in enumerate(similar_results):
                            logger.info(f"  Similar Result #{j+1}: ID: {sim_res.id}, Title: {sim_res.title}, URL: {sim_res.url}, Score: {sim_res.score}")
                    else:
                        logger.info(f"No similar results found by Exa for URL: {first_url_for_similar}")
                else:
                    logger.info("Skipping Find Similar test as first search result had no URL.")
            else:
                logger.info(f"No Exa search results found for query: {search_query}")

        except ExaSearchClientError as e:
            logger.error(f"Exa Search client test error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during Exa Search client test: {e}", exc_info=True)

if __name__ == "__main__":
    import asyncio
    # To run this test:
    # 1. Ensure you have a config/settings.yaml or relevant environment variables for exa_search.
    #    Example minimal settings.yaml:
    #    exa_search:
    #      base_url: "https://api.exa.ai"
    #      api_key: "YOUR_EXA_API_KEY"
    # 2. Uncomment the line below.
    # asyncio.run(main_exa_search_test())
    pass
