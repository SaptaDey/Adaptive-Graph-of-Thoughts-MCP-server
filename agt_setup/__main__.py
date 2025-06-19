import importlib
import os
from pathlib import Path

import typer
from dotenv import load_dotenv, set_key
from neo4j import GraphDatabase

app = typer.Typer(add_completion=False)


from neo4j.exceptions import AuthError, ServiceUnavailable

def _test_connection(uri: str, user: str, password: str, database: str) -> bool:
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            session.run("MATCH (n) RETURN count(n) LIMIT 1")
        driver.close()
        return True
    except Exception as e:
        # Log the unexpected error
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        return False


@app.command()
def run() -> None:
    """Interactive setup wizard for Adaptive GoT."""
    if Path(".env").exists():
        if typer.confirm("Import existing .env values?", default=True):
            load_dotenv(".env")
            typer.echo("Loaded values from .env")
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

    missing: list[str] = []
    for pkg in ["openai", "anthropic"]:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)

    if missing:
        joined = ", ".join(missing)
        typer.secho(f"Optional packages missing: {joined}", fg="yellow")
        typer.secho("Install them with 'poetry add ' or 'pip install' to enable LLM features.", fg="yellow")


if __name__ == "__main__":
    app()
