import os
from pathlib import Path

import typer
from dotenv import set_key
from neo4j import GraphDatabase

app = typer.Typer(add_completion=False)


def _test_connection(uri: str, user: str, password: str, database: str) -> bool:
    """
    Attempts to connect to a Neo4j database using the provided URI, username, password, and database name.
    
    Returns:
        bool: True if the connection is successful; False otherwise.
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


@app.command()
def run() -> None:
    """
    Launches an interactive command-line wizard to configure and securely store Neo4j database credentials for the Adaptive GoT application.
    
    Prompts the user for connection details, tests the connection, and saves the credentials to a `.env` file with restricted permissions. Exits with an error if the connection cannot be established.
    """
    uri = typer.prompt(
        "Neo4j URI", default=os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    )
    user = typer.prompt("Neo4j User", default=os.getenv("NEO4J_USER", "neo4j"))
    password = typer.prompt(
        "Neo4j Password",
        default=os.getenv("NEO4J_PASSWORD", ""),
        hide_input=True,
    )
    database = typer.prompt(
        "Neo4j Database", default=os.getenv("NEO4J_DATABASE", "neo4j")
    )

    typer.echo("Testing Neo4j connection...")
    if not _test_connection(uri, user, password, database):
        typer.secho("Failed to connect to Neo4j with provided details", fg="red")
        raise typer.Exit(1)
    typer.secho("Connected successfully!", fg="green")

    env_path = Path(".env")
    env_path.touch(mode=0o600, exist_ok=True)
    set_key(str(env_path), "NEO4J_URI", uri)
    set_key(str(env_path), "NEO4J_USER", user)
    set_key(str(env_path), "NEO4J_PASSWORD", password)
    set_key(str(env_path), "NEO4J_DATABASE", database)
    env_path.chmod(0o600)
    typer.secho(f"Credentials saved to {env_path}", fg="green")


if __name__ == "__main__":
    app()
