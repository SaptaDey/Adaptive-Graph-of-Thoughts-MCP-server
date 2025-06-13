from typing import Optional, List, Dict, Any
from loguru import logger
from pydantic import BaseModel, Field
import xml.etree.ElementTree as ET

from adaptive_graph_of_thoughts.config import Settings, PubMedConfig
from .base_client import AsyncHTTPClient, APIRequestError, APIHTTPError, BaseAPIClientError

class PubMedArticle(BaseModel):
    pmid: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    publication_date: Optional[str] = None # Consider using datetime internally
    doi: Optional[str] = None
    url: str # e.g., https://pubmed.ncbi.nlm.nih.gov/PMID/

class PubMedClientError(BaseAPIClientError):
    """Custom error for PubMedClient."""
    pass

class PubMedClient:
    def __init__(self, settings: Settings):
        """
        Initializes the PubMedClient with the provided settings.
        
        Raises:
            PubMedClientError: If the PubMed configuration or base URL is missing in the settings.
        """
        if not settings.pubmed or not settings.pubmed.base_url:
            logger.error("PubMed configuration is missing in settings.")
            raise PubMedClientError("PubMed configuration (base_url) is not set.")

        self.config: PubMedConfig = settings.pubmed
        self.http_client = AsyncHTTPClient(
            base_url=self.config.base_url,
            # settings=settings # Pass settings if base_client needs it
        )
        self.email = self.config.email
        self.api_key = self.config.api_key
        logger.info(f"PubMedClient initialized for base URL: {self.config.base_url}")

    async def close(self):
        """
        Closes the underlying asynchronous HTTP client and releases network resources.
        """
        await self.http_client.close()

    async def __aenter__(self):
        """
        Enters the asynchronous context for the PubMedClient, initializing the underlying HTTP client.
        
        Returns:
            The PubMedClient instance for use within an async context manager.
        """
        await self.http_client.client.__aenter__() # Enter the httpx client context
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the asynchronous context manager, ensuring the underlying HTTP client is properly closed.
        """
        await self.http_client.client.__aexit__(exc_type, exc_val, exc_tb) # Exit the httpx client context

    def _construct_common_params(self) -> Dict[str, Any]:
        """
        Builds a dictionary of common query parameters for PubMed API requests.
        
        Includes the API key and email if they are configured.
        """
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

    def _parse_esummary_response(self, xml_text: str) -> List[PubMedArticle]:
        """
        Parses PubMed eSummary XML and extracts article metadata into PubMedArticle objects.
        
        Args:
            xml_text: XML string returned from the PubMed eSummary API.
        
        Returns:
            A list of PubMedArticle instances containing metadata such as PMID, title, authors, journal, publication date, DOI, and article URL.
        
        Raises:
            PubMedClientError: If the XML cannot be parsed.
        """
        articles: List[PubMedArticle] = []
        try:
            root = ET.fromstring(xml_text)
            for doc_sum in root.findall("DocSum"):
                pmid_node = doc_sum.find("Id")
                if pmid_node is None or not pmid_node.text:
                    continue
                pmid = pmid_node.text

                data: Dict[str, Any] = {"pmid": pmid, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}

                for item in doc_sum.findall("Item"):
                    name = item.get("Name")
                    # item_type = item.get("Type") # Not strictly needed for current parsing logic
                    if name == "Title" and item.text:
                        data["title"] = item.text
                    elif name == "AuthorList":
                        authors = [author.text for author in item.findall("Item[@Name='Author']") if author.text]
                        data["authors"] = authors
                    elif name == "Source" and item.text: # Journal name
                        data["journal"] = item.text
                    elif name == "PubDate" and item.text: # Publication date (e.g., "2023" or "2023 May")
                        data["publication_date"] = item.text
                    elif name == "DOI" and item.text:
                        data["doi"] = item.text
                    # Add more fields as needed, e.g., EPubDate, FullJournalName

                if "title" not in data: # Skip if essential info like title is missing
                    logger.warning(f"Skipping article PMID {pmid} due to missing title in eSummary.")
                    continue

                articles.append(PubMedArticle(**data))
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed eSummary XML: {e}")
            raise PubMedClientError(f"XML parsing error for eSummary: {e}") from e
        except Exception as e: # Catch any other unexpected errors during parsing
            # Declaring pmid outside the loop for reference in error message if parsing fails early
            current_pmid_parsing = "unknown"
            if 'pmid' in locals() and pmid:
                current_pmid_parsing = pmid
            logger.error(f"Unexpected error parsing PubMed eSummary for PMID {current_pmid_parsing}: {e}")
            # Optionally re-raise or continue with partial data
        return articles

    async def search_articles(self, query: str, max_results: int = 10) -> List[PubMedArticle]:
        """
        Searches PubMed for articles matching a query and retrieves detailed metadata.
        
        Performs an asynchronous search using the PubMed eSearch and eSummary APIs to find articles matching the provided query, up to the specified maximum number of results. For each article found, fetches and populates the abstract. Returns a list of fully populated `PubMedArticle` objects.
        
        Args:
            query: The search query string.
            max_results: The maximum number of articles to retrieve.
        
        Returns:
            A list of `PubMedArticle` instances containing metadata and abstracts for each found article.
        
        Raises:
            PubMedClientError: If an error occurs during the PubMed API requests or response parsing.
        """
        logger.debug(f"Searching PubMed for query: '{query}', max_results: {max_results}")

        # Step 1: ESearch to get PMIDs
        esearch_params = self._construct_common_params()
        esearch_params.update({
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "usehistory": "y", # Useful if we need to retrieve more results later
            "retmode": "json",
        })

        pmids: List[str] = []
        try:
            response_esearch = await self.http_client.get("esearch.fcgi", params=esearch_params)
            esearch_data = response_esearch.json()

            if "esearchresult" in esearch_data and "idlist" in esearch_data["esearchresult"]:
                pmids = esearch_data["esearchresult"]["idlist"]
                logger.debug(f"Found {len(pmids)} PMIDs from eSearch: {pmids}")
            else:
                logger.warning(f"No PMIDs found or unexpected eSearch response format for query: {query}")
                return []
            if not pmids: # Handles case where idlist is empty
                logger.info(f"eSearch returned no PMIDs for query: {query}")
                return []
        except (APIHTTPError, APIRequestError) as e:
            logger.error(f"PubMed eSearch API error for query '{query}': {e}")
            raise PubMedClientError(f"PubMed eSearch failed: {e}") from e
        except ValueError as e: # JSONDecodeError inherits from ValueError
            logger.error(f"Failed to decode JSON from PubMed eSearch for query '{query}': {e}")
            raise PubMedClientError(f"PubMed eSearch JSON decode error: {e}") from e

        # Step 2: ESummary to get article summaries
        esummary_params = self._construct_common_params()
        esummary_params.update({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml", # eSummary provides more details in XML
        })

        try:
            response_esummary = await self.http_client.get("esummary.fcgi", params=esummary_params)
            xml_text = response_esummary.text
            articles = self._parse_esummary_response(xml_text)

            # Fetch abstracts for each article found
            # This makes search_articles more comprehensive but potentially slower.
            # Consider if abstracts should always be fetched or if it's an optional step / separate call.
            for article in articles:
                if article.pmid: # Ensure pmid is valid before fetching abstract
                    try:
                        abstract = await self.fetch_abstract(article.pmid)
                        article.abstract = abstract # Populate abstract field
                    except PubMedClientError as e:
                        logger.warning(f"Could not fetch abstract for PMID {article.pmid}: {e}")
                        # Decide if this should halt processing or just log and continue
                        article.abstract = None

            logger.info(f"Successfully retrieved and parsed {len(articles)} articles from PubMed for query: '{query}'.")
            return articles
        except (APIHTTPError, APIRequestError) as e:
            logger.error(f"PubMed eSummary API error for PMIDs '{','.join(pmids)}': {e}")
            raise PubMedClientError(f"PubMed eSummary failed: {e}") from e
        # Errors from _parse_esummary_response are already logged and potentially re-raised
        return [] # Should not be reached if successful, but as a fallback

    async def fetch_abstract(self, pmid: str) -> Optional[str]:
        """
        Fetches the abstract text for a given PubMed article by PMID.
        
        Queries the PubMed eFetch API for the specified PMID, parses the XML response, and extracts the abstract text, including labeled or categorized sections if present. Returns the full abstract as a single string, or None if no abstract is found.
        
        Args:
            pmid: The PubMed ID of the article.
        
        Returns:
            The abstract text as a string, or None if no abstract is available.
        
        Raises:
            PubMedClientError: If there is an error with the API request or XML parsing.
        """
        logger.debug(f"Fetching abstract for PubMed PMID: {pmid}")
        efetch_params = self._construct_common_params()
        efetch_params.update({
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml",
        })

        try:
            response_efetch = await self.http_client.get("efetch.fcgi", params=efetch_params)
            xml_text = response_efetch.text

            root = ET.fromstring(xml_text)
            abstract_texts: List[str] = []

            # Iterate over Article/Abstract/AbstractText and potentially other locations
            # like OtherAbstract or even full text snippets if available and desired.
            for abstract_node in root.findall(".//AbstractText"): # Common path
                text_content = "".join(abstract_node.itertext()).strip()
                if text_content: # Ensure there's actual text
                    label = abstract_node.get("Label")
                    nlm_category = abstract_node.get("NlmCategory") # e.g., BACKGROUND, METHODS, RESULTS, CONCLUSIONS

                    prefix = ""
                    if label:
                        prefix = f"{label}: "
                    elif nlm_category: # Use NlmCategory as a fallback label if Label is not present
                        prefix = f"{nlm_category.capitalize()}: "

                    abstract_texts.append(f"{prefix}{text_content}")

            if abstract_texts:
                full_abstract = "\n".join(abstract_texts)
                logger.debug(f"Successfully fetched abstract for PMID: {pmid}")
                return full_abstract
            else: # Check for alternative abstract locations if primary is empty
                for other_abstract_node in root.findall(".//OtherAbstract/AbstractText"):
                    text_content = "".join(other_abstract_node.itertext()).strip()
                    if text_content:
                        abstract_texts.append(text_content)
                if abstract_texts:
                    full_abstract = "\n".join(abstract_texts)
                    logger.debug(f"Successfully fetched abstract (from OtherAbstract) for PMID: {pmid}")
                    return full_abstract

                logger.warning(f"No abstract found in eFetch response for PMID: {pmid}")
                return None
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed eFetch XML for PMID {pmid}: {e}")
            raise PubMedClientError(f"XML parsing error for eFetch (PMID {pmid}): {e}") from e
        except (APIHTTPError, APIRequestError) as e:
            logger.error(f"PubMed eFetch API error for PMID {pmid}: {e}")
            raise PubMedClientError(f"PubMed eFetch failed (PMID {pmid}): {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching abstract for PMID {pmid}: {e}")
            raise PubMedClientError(f"Unexpected error in fetch_abstract (PMID {pmid}): {e}") from e

# Example Usage (for testing)
async def main_pubmed_test():
    """
    Demonstrates usage of the PubMedClient by searching for articles and logging results.
    
    Attempts to load PubMed configuration, falling back to a default if necessary. Performs a sample search for "artificial intelligence in medicine," retrieves article metadata and abstracts, and logs the results. Handles and logs configuration, client, and unexpected errors.
    """
    from adaptive_graph_of_thoughts.config import Settings, PubMedConfig # Ensure PubMedConfig is imported

    # Attempt to load settings, fallback to a default mock if issues occur
    try:
        settings = Settings()
        if not settings.pubmed or not settings.pubmed.base_url:
            logger.warning("PubMed config not fully set in settings.yaml or env vars; using mock for testing.")
            # Provide a default email, as NCBI recommends it.
            settings.pubmed = PubMedConfig(base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/", email="test.user@example.com")
    except Exception as e:
        logger.warning(f"Could not load global settings ({e}), using mock PubMed config for testing.")
        # Manually create a Settings object with a default PubMedConfig
        settings = Settings(
            pubmed=PubMedConfig(base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/", email="test.user@example.com")
            # app, asr_got, etc., would use their defaults or be None if not specified
        )

    async with PubMedClient(settings=settings) as client:
        try:
            query = "artificial intelligence in medicine"
            articles = await client.search_articles(query, max_results=1) # Reduced for faster test
            if articles:
                logger.info(f"Found {len(articles)} articles for '{query}':")
                for article in articles:
                    logger.info(f"  PMID: {article.pmid}, Title: {article.title}, DOI: {article.doi}")
                    # Abstract is now fetched within search_articles, so it should be populated
                    if article.abstract:
                         logger.info(f"    Abstract (first 150 chars): {article.abstract[:150]}...")
                    else:
                        logger.info(f"    No abstract found or fetched for PMID {article.pmid}.")
            else:
                logger.info(f"No articles found for query: {query}")

        except PubMedClientError as e:
            logger.error(f"PubMed client test error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during PubMed client test: {e}", exc_info=True)

if __name__ == "__main__":
    import asyncio
    # To run this test:
    # 1. Ensure you have a config/settings.yaml or relevant environment variables for pubmed.
    #    Example minimal settings.yaml:
    #    pubmed:
    #      base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    #      email: "your.email@example.com" # NCBI strongly recommends providing an email
    #      # api_key: "YOUR_NCBI_API_KEY" # Optional, but good for higher rate limits
    # 2. Uncomment the line below.
    # asyncio.run(main_pubmed_test())
    pass
