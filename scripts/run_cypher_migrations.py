#!/usr/bin/env python
"""Apply versioned Cypher migrations to the Neo4j database."""
import argparse
import asyncio
from pathlib import Path
from adaptive_graph_of_thoughts.infrastructure.neo4j_utils import execute_cypher_file

async def run_migrations(directory: str) -> None:
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Migration directory '{directory}' does not exist")

    files = sorted(p for p in path.glob("*.cypher") if p.is_file())
    if not files:
        print(f"No migration files found in '{directory}'.")
        return

    for cypher_file in files:
        print(f"Applying {cypher_file.name}...")
        try:
            await execute_cypher_file(str(cypher_file))
            print(f"✓ Applied {cypher_file.name}")
            # TODO: Track applied migrations to prevent re-running
        except Exception as e:
            print(f"✗ Failed to apply {cypher_file.name}: {e}")
            raise  # Re-raise to stop further migrations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cypher migrations")
    parser.add_argument(
        "--dir",
        default="database_migrations",
        help="Directory containing .cypher migration files",
    )
    args = parser.parse_args()
    asyncio.run(run_migrations(args.dir))
