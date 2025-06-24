"""External API clients used by the infrastructure layer."""

from .base_client import AsyncHTTPClient, APIHTTPError, APIRequestError, BaseAPIClientError
from .exa_search_client import ExaArticleResult, ExaSearchClient, ExaSearchClientError
from .google_scholar_client import (
    GoogleScholarArticle,
    GoogleScholarClient,
    GoogleScholarClientError,
    UnexpectedResponseStructureError,
)
from .pubmed_client import (
    PubMedArticle,
    PubMedClient,
    PubMedClientError,
)

__all__ = [
    "AsyncHTTPClient",
    "APIHTTPError",
    "APIRequestError",
    "BaseAPIClientError",
    "ExaArticleResult",
    "ExaSearchClient",
    "ExaSearchClientError",
    "GoogleScholarArticle",
    "GoogleScholarClient",
    "GoogleScholarClientError",
    "UnexpectedResponseStructureError",
    "PubMedArticle",
    "PubMedClient",
    "PubMedClientError",
]
