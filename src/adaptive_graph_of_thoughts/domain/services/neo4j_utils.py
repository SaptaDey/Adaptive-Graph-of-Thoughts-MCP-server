import asyncio
import atexit
import threading
from typing import Any, Optional

import re

from loguru import logger
from dataclasses import dataclass
from neo4j import (
    Driver,
    GraphDatabase,
    Record,
    Result,
    Session,
    Transaction,
    unit_of_work,
)
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.adaptive_graph_of_thoughts.config import runtime_settings


# --- Simple Configuration ---
class Neo4jSettings:
    def __init__(self):
        """
        Initialize Neo4j connection settings from the application's runtime configuration.
        """
        self.uri: str = runtime_settings.neo4j.uri
        self.user: str = runtime_settings.neo4j.user
        self.password: str = runtime_settings.neo4j.password
        self.database: str = runtime_settings.neo4j.database


# --- Global Configuration ---
class GlobalSettings:
    def __init__(self):
        self.neo4j = Neo4jSettings()


_neo4j_settings: Optional[GlobalSettings] = None
_driver: Optional[Driver] = None  # Backwards compatibility for tests


class Neo4jDriverManager:
    """Thread-safe manager for the Neo4j driver."""

    def __init__(self) -> None:
        self._driver: Optional[Driver] = None
        self._lock = threading.Lock()
        atexit.register(self.cleanup)

    def _create_driver(self) -> Driver:
        settings = get_neo4j_settings().neo4j
        return create_neo4j_driver(settings)

    def get_driver(self) -> Driver:
        with self._lock:
            if self._driver is None or self._driver.closed:
                self._driver = self._create_driver()
if self._driver:
    global _driver
    _driver = self._driver
            return self._driver

    def cleanup(self) -> None:
        if self._driver and not self._driver.closed:
            logger.info("Closing Neo4j driver.")
            self._driver.close()
            self._driver = None
if self._driver is not None:
    global _driver
    _driver = None


driver_manager = Neo4jDriverManager()

# Allowed labels for node creation to mitigate injection attacks
ALLOWED_LABELS = {"User", "Document", "Hypothesis", "Evidence"}


def sanitize_cypher_input(value: str) -> str:
    """Remove potentially dangerous characters from a Cypher identifier."""
    # Allow alphanumeric, underscore, hyphen, and dot (for namespaces)
    return re.sub(r"[^\w.-]", "", value)

def mask_uri(uri: str) -> str:
    """Mask sensitive parts of a URI for logging."""
    import re

    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", uri)


def mask_username(username: str) -> str:
    """Mask a username for logging."""
    if len(username) <= 2:
        return "***"
    return username[:2] + "*" * (len(username) - 2)


@dataclass
class Neo4jConnection:
    """Lightweight Neo4j connection wrapper used in unit tests."""

    uri: str
    user: str
    password: str
    database: str = "neo4j"

    def __post_init__(self) -> None:
        self._driver: Driver | None = None

    def connect(self) -> None:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None


def create_neo4j_driver(settings: Neo4jSettings) -> Driver:
    """Create and verify a new Neo4j driver using the provided settings."""
    if not settings.uri or not settings.user or not settings.password:
        raise ServiceUnavailable("Neo4j connection details are incomplete in settings.")

    logger.info(f"Initializing Neo4j driver for URI: {settings.uri}")
    driver = GraphDatabase.driver(
        settings.uri,
        auth=(settings.user, settings.password),
        max_connection_lifetime=3600,
        max_connection_pool_size=50,
        connection_acquisition_timeout=60,
    )
    driver.verify_connectivity()
    logger.info("Neo4j driver initialized and connectivity verified.")
    return driver


def get_neo4j_settings() -> GlobalSettings:
    """Returns the Neo4j settings, initializing them if necessary."""
    global _neo4j_settings
    if _neo4j_settings is None:
        logger.info("Initializing Neo4j settings.")
        _neo4j_settings = GlobalSettings()
        logger.debug(
            "Neo4j Settings loaded: URI='%s', User='%s', Default DB='%s'",
            mask_uri(_neo4j_settings.neo4j.uri),
            mask_username(_neo4j_settings.neo4j.user),
            _neo4j_settings.neo4j.database,
        )
    return _neo4j_settings


# --- Driver Management ---
def get_neo4j_driver() -> Driver:
    """Return a singleton Neo4j driver instance."""

    try:
        return driver_manager.get_driver()
    except ServiceUnavailable:
        raise
    except Exception as e:
        logger.error(f"Unexpected error obtaining Neo4j driver: {e}")
        raise


def close_neo4j_driver() -> None:
    """Close the Neo4j driver if it is open."""
    driver_manager.cleanup()


# --- Query Execution ---
async def execute_query(
    query: str,
    parameters: Optional[dict[str, Any]] = None,
    database: Optional[str] = None,
    tx_type: str = "read",  # 'read' or 'write'
    *,
    driver: Optional[Driver] = None,
) -> list[Record]:
    """
    Executes a Cypher query asynchronously against the Neo4j database.

    Runs the specified query as either a read or write transaction, using the provided parameters and database name. If no database is specified, falls back to the configured default or "neo4j". Returns the list of records resulting from the query.

    Args:
        query: The Cypher query string to execute.
        parameters: Optional dictionary of parameters to pass to the query.
        database: Optional database name; uses the configured default or "neo4j" if not provided.
        tx_type: Transaction type, either "read" or "write". Defaults to "read".
        driver: Optional Neo4j ``Driver`` instance. If not provided, the module's
            global driver will be used.

    Returns:
        List of records returned by the query.

    Raises:
        ServiceUnavailable: If the Neo4j service is unavailable.
        Neo4jError: If an error occurs during query execution.
        ValueError: If an invalid transaction type is specified.
    """
    driver = driver or get_neo4j_driver()
    if not driver:  # Should not happen if get_neo4j_driver raises on failure
        logger.error("Neo4j driver not available. Cannot execute query.")
        raise ServiceUnavailable("Neo4j driver not initialized or connection failed.")

    settings = get_neo4j_settings()
    db_name = database if database else settings.database

    records: list[Record] = []

    def _execute_sync_query() -> list[Record]:
        with driver.session(database=db_name) as session:
            logger.debug(
                f"Executing query on database '{db_name}' with type '{tx_type}': {query[:100]}..."
            )

            @unit_of_work(timeout=30)  # Example timeout, adjust as needed
            def _transaction_work(tx: Transaction) -> list[Record]:
                result: Result = tx.run(query, parameters)
                return list(result)

            if tx_type == "read":
                sync_records = session.execute_read(_transaction_work)
            elif tx_type == "write":
                sync_records = session.execute_write(_transaction_work)
            else:
                logger.error(
                    f"Invalid transaction type: {tx_type}. Must be 'read' or 'write'."
                )
                raise ValueError(
                    f"Invalid transaction type: {tx_type}. Must be 'read' or 'write'."
                )

            logger.info(
                f"Query executed successfully on database '{db_name}'. Fetched {len(sync_records)} records."
            )
            return sync_records

    try:
        # Run blocking I/O in a thread to avoid blocking the event loop
        records = await asyncio.to_thread(_execute_sync_query)
    except Neo4jError as e:
        logger.error(f"Neo4j error executing Cypher query on database '{db_name}': {e}")
        logger.error(f"Query: {query}, Parameters: {parameters}")
        raise  # Re-raise the specific Neo4jError
    except ServiceUnavailable:
        logger.error(
            f"Neo4j service became unavailable while attempting to execute query on '{db_name}'."
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error executing Cypher query on database '{db_name}': {e}"
        )
        logger.error(f"Query: {query}, Parameters: {parameters}")
        raise  # Re-raise any other unexpected exception

    return records


async def create_node(label: str, properties: dict[str, Any]) -> list[Record]:

    # Validate label to prevent injection
    if not label.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            f"Invalid label: {label}. Labels must be alphanumeric with underscores/hyphens only."
        )

    query = (
        f"CREATE (n:{label} {{"
        + ", ".join(f"{k}: ${k}" for k in properties)
        + "}) RETURN n"
    )
    return await execute_query(query, properties, tx_type="write")


    query = f"CREATE (n:{clean_label}) SET n = $props RETURN n"
    return await execute_query(query, {"props": properties}, tx_type="write")


async def update_node(node_id: str, updates: dict[str, Any]) -> list[Record]:
    # Validate property names to prevent injection
    for key in updates:
        if not key.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Invalid property name: {key}. Property names must be alphanumeric with underscores/hyphens only."
            )


async def update_node(node_id: str, updates: dict[str, Any]) -> list[Record]:
    try:
        node_id_int = int(node_id)
    except ValueError:
        raise ValueError(f"Invalid node_id: {node_id}. Must be a valid integer.")

    query = "MATCH (n) WHERE id(n) = $id SET n += $props RETURN n"
    params = {"id": node_id_int, "props": updates}
    return await execute_query(query, params, tx_type="write")


async def delete_node(node_id: str) -> list[Record]:
    try:
        node_id_int = int(node_id)
    except ValueError:
        raise ValueError(f"Invalid node_id: {node_id}. Must be a valid integer.")

    query = "MATCH (n) WHERE id(n) = $id DETACH DELETE n RETURN count(n)"
    return await execute_query(query, {"id": node_id_int}, tx_type="write")


async def find_nodes(label: str, filters: dict[str, Any]) -> list[Record]:

    # Validate label to prevent injection
    if not label.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            f"Invalid label: {label}. Labels must be alphanumeric with underscores/hyphens only."
        )

    # Validate property names to prevent injection
    for key in filters:
        if not key.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Invalid property name: {key}. Property names must be alphanumeric with underscores/hyphens only."
            )

    where = " AND ".join(f"n.{k} = ${k}" for k in filters)
    query = f"MATCH (n:{label}) WHERE {where} RETURN n"
    return await execute_query(query, filters)



async def create_relationship(
    from_id: str, to_id: str, rel_type: str, properties: dict[str, Any]
) -> list[Record]:

    # Validate relationship type to prevent injection
    if not rel_type.replace("_", "").replace("-", "").isalnum():

        raise ValueError(
            f"Invalid relationship type: {rel_type}. "
            "Must be alphanumeric with underscores/hyphens only."
        )


    # Validate property names to prevent injection
    for key in properties:
        if not key.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Invalid property name: {key}. "
                "Property names must be alphanumeric with underscores/hyphens only."
            )


    # Validate node IDs
    try:
        from_id_int = int(from_id)
        to_id_int = int(to_id)
    except ValueError:
        raise ValueError(
            "Invalid node IDs. Both from_id and to_id must be valid integers."
        )

    query = (
        f"MATCH (a),(b) WHERE id(a)=$from AND id(b)=$to "
        f"CREATE (a)-[r:{clean_rel_type}]->(b) SET r = $props RETURN r"
    )
    params = {"from": from_id_int, "to": to_id_int, "props": properties}
    return await execute_query(query, params, tx_type="write")


async def get_database_info() -> list[Record]:
    query = "CALL db.info()"
    return await execute_query(query)


async def validate_connection() -> bool:
    try:
        await get_database_info()
        return True
    except Exception:
        return False


async def bulk_create_nodes(label: str, nodes: list[dict[str, Any]]) -> list[Record]:

    # Validate label to prevent injection
    if not label.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            f"Invalid label: {label}. Labels must be alphanumeric with underscores/hyphens only."
        )

    # Use UNWIND for efficient bulk creation
    query = f"UNWIND $nodes AS nodeData CREATE (n:{label}) SET n = nodeData RETURN n"

    return await execute_query(query, {"nodes": nodes}, tx_type="write")


async def execute_cypher_file(path: str) -> list[Record]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            query = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Cypher file not found: {path}")
    except IOError as e:
        raise IOError(f"Error reading Cypher file {path}: {e}")

    if not query.strip():
        raise ValueError(f"Cypher file {path} is empty or contains only whitespace")

    return await execute_query(query, tx_type="write")


# Example of how to use (optional, for testing or demonstration)
if __name__ == "__main__":
    logger.add("neo4j_utils.log", rotation="500 MB")  # For local testing

    # Ensure NEO4J_PASSWORD is set as an environment variable if different from default
    # For example: export NEO4J_PASSWORD="your_actual_password"


async def main():
    try:
        # Example Read Query
        logger.info("Attempting to execute a sample READ query...")
        read_query = "MATCH (n) RETURN count(n) AS node_count"
        await execute_query(read_query, tx_type="read")
        # ... rest of example code with await ...
    # ... exception handling ...
    finally:
        close_neo4j_driver()
        logger.info("Neo4j utils example finished.")


if __name__ == "__main__":
    asyncio.run(main())
    """
    try:
        # Example Read Query
        logger.info("Attempting to execute a sample READ query...")
        read_query = "MATCH (n) RETURN count(n) AS node_count"
        read_results = await execute_query(read_query, tx_type="read")
        if read_results:
            logger.info(f"Read query results: Found {read_results[0]['node_count']} nodes in database.")
        else:
            logger.info("Read query returned no results or failed.")

        # Example Write Query (use with caution on your database)
        # logger.info("Attempting to execute a sample WRITE query...")
        # write_query = "CREATE (a:Greeting {message: $msg})"
        # write_params = {"msg": "Hello from neo4j_utils"}
        # await execute_query(write_query, parameters=write_params, tx_type="write")
        # logger.info("Write query executed (if no errors).")

        # logger.info("Attempting to read the written data...")
        # verify_query = "MATCH (g:Greeting) WHERE g.message = $msg RETURN g.message AS message"
        # verify_results = await execute_query(verify_query, parameters={"msg": "Hello from neo4j_utils"}, tx_type="read")
        # if verify_results:
        #     logger.info(f"Verification query results: Found message '{verify_results[0]['message']}'")
        # else:
        #     logger.warning("Verification query did not find the written data or failed.")
    except ServiceUnavailable:
        logger.error("Could not connect to Neo4j. Ensure Neo4j is running and accessible.")
    except Exception as e:
        logger.error(f"An error occurred during the example usage: {e}")
    finally:
        close_neo4j_driver()
        logger.info("Neo4j utils example finished.")
    """
