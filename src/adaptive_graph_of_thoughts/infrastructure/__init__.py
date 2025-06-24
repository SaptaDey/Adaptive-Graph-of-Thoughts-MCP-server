"""Infrastructure implementations of domain interfaces and external clients."""

from .neo4j_repository import Neo4jGraphRepository
from .neo4j_utils import (
    execute_query,
    get_neo4j_driver,
    close_neo4j_driver,
    bulk_create_nodes_optimized,
)
from .api_clients import (
    AsyncHTTPClient,
    APIHTTPError,
    APIRequestError,
    BaseAPIClientError,
    ExaArticleResult,
    ExaSearchClient,
    ExaSearchClientError,
    GoogleScholarArticle,
    GoogleScholarClient,
    GoogleScholarClientError,
    UnexpectedResponseStructureError,
    PubMedArticle,
    PubMedClient,
    PubMedClientError,
)

__all__ = [
    "Neo4jGraphRepository",
    "execute_query",
    "get_neo4j_driver",
    "close_neo4j_driver",
    "bulk_create_nodes_optimized",
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
