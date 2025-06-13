from neo4j import GraphDatabase, Driver, Record, Result, Transaction, unit_of_work
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from typing import Optional, Any, List, Dict
import asyncio
from loguru import logger

from adaptive_graph_of_thoughts.config import settings # Import global settings

# --- Global Driver Instance ---
_driver: Optional[Driver] = None

# --- Driver Management ---
def get_neo4j_driver() -> Driver:
    """
    Initializes and returns a Neo4j driver instance using a singleton pattern.
    Handles authentication using configured credentials from global settings.
    """
    global _driver

    if _driver is None or _driver.closed:
        if settings.neo4j is None:
            logger.error("Neo4j configuration is missing in global settings.")
            raise ServiceUnavailable("Neo4j configuration is not available.")

        uri = settings.neo4j.uri
        username = settings.neo4j.username
        password = settings.neo4j.password

        if not uri or not username or not password:
            logger.error("Neo4j URI, username, or password missing in configuration.")
            raise ServiceUnavailable("Neo4j URI, username, or password missing in configuration.")

        logger.info(f"Initializing Neo4j driver for URI: {uri}")
        try:
            _driver = GraphDatabase.driver(uri, auth=(username, password))
            _driver.verify_connectivity()
            logger.info("Neo4j driver initialized and connectivity verified.")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j at {uri}: {e}")
            _driver = None  # Ensure driver is None if connection failed
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while initializing Neo4j driver: {e}")
            _driver = None  # Ensure driver is None on other errors
            raise
    return _driver

def close_neo4j_driver() -> None:
    """Closes the Neo4j driver instance if it's open."""
    global _driver
    if _driver is not None and not _driver.closed:
        logger.info("Closing Neo4j driver.")
        _driver.close()
        _driver = None
    else:
        logger.info("Neo4j driver is already closed or not initialized.")

# --- Query Execution ---
async def execute_query(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    database: Optional[str] = None,
    tx_type: str = "read"  # 'read' or 'write'
) -> List[Record]:
    """
    Executes a Cypher query using a session from the driver.

    Args:
        query: The Cypher query string.
        parameters: Optional dictionary of parameters for the query.
        database: Optional name of the database to use. If None, uses default from settings.
        tx_type: Type of transaction ('read' or 'write'). Defaults to 'read'.

    Returns:
        A list of records resulting from the query.

    Raises:
        ServiceUnavailable: If the driver cannot connect to Neo4j.
        Neo4jError: For errors during query execution.
        ValueError: If an invalid tx_type is provided.
    """
    driver = get_neo4j_driver()  # Ensures driver is initialized

    # Determine the database name
    db_name: Optional[str] = database  # Use function argument first
    if db_name is None and settings.neo4j and settings.neo4j.database:
        db_name = settings.neo4j.database  # Fallback to global settings
    if db_name is None:
        db_name = "neo4j"  # Default if not specified anywhere

    records: List[Record] = []

    def _execute_sync_query() -> List[Record]:
        with driver.session(database=db_name) as session:
            logger.debug(f"Executing query on database '{db_name}' with type '{tx_type}': {query[:100]}...")

            @unit_of_work(timeout=30)  # Example timeout, adjust as needed
            def _transaction_work(tx: Transaction) -> List[Record]:
                result: Result = tx.run(query, parameters)
                return [record for record in result]

            if tx_type == "read":
                sync_records = session.execute_read(_transaction_work)
            elif tx_type == "write":
                sync_records = session.execute_write(_transaction_work)
            else:
                logger.error(f"Invalid transaction type: {tx_type}. Must be 'read' or 'write'.")
                raise ValueError(f"Invalid transaction type: {tx_type}. Must be 'read' or 'write'.")

            logger.info(f"Query executed successfully on database '{db_name}'. Fetched {len(sync_records)} records.")
            return sync_records

    try:
        # Run blocking I/O in a thread to avoid blocking the event loop
        records = await asyncio.to_thread(_execute_sync_query)
    except Neo4jError as e:
        logger.error(f"Neo4j error executing Cypher query on database '{db_name}': {e}")
        logger.error(f"Query: {query}, Parameters: {parameters}")
        raise  # Re-raise the specific Neo4jError
    except ServiceUnavailable:
        logger.error(f"Neo4j service became unavailable while attempting to execute query on '{db_name}'.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error executing Cypher query on database '{db_name}': {e}")
        logger.error(f"Query: {query}, Parameters: {parameters}")
        raise  # Re-raise any other unexpected exception

    return records

# Example of how to use (optional, for testing or demonstration)
if __name__ == "__main__":
    logger.add("neo4j_utils.log", rotation="500 MB") # For local testing

    # Ensure NEO4J_PASSWORD is set as an environment variable if different from default
    # For example: export NEO4J_PASSWORD="your_actual_password"

async def main():
    try:
        # Example Read Query
        logger.info("Attempting to execute a sample READ query...")
        read_query = "MATCH (n) RETURN count(n) AS node_count"
        read_results = await execute_query(read_query, tx_type="read")
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
