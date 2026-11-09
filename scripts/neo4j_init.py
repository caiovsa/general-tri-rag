#!/usr/bin/env python3
"""Create the Neo4j databases this repo expects (idempotent).

Run: uv run python -m scripts.neo4j_init
Needs a multi-database server (Neo4j Desktop / Enterprise). On Community the
`CREATE DATABASE` command is unavailable; use the isolated `neo4j-metaqa`
service instead (see docs/metaqa-track.md).
"""

from __future__ import annotations

from neo4j import GraphDatabase

from shared.config import settings


def databases() -> list[str]:
    return [
        settings.NEO4J_DATABASE,
        settings.NEO4J_METAQA_DATABASE,
        settings.NEO4J_METAQA_MINI_DATABASE,
        settings.NEO4J_METAQA_MINI30_DATABASE,
        settings.NEO4J_PHASE3_DATABASE,
        settings.NEO4J_METAQA_PHASE3_DATABASE,
        settings.NEO4J_METAQA_MINI_PHASE3_DATABASE,
        settings.NEO4J_METAQA_MINI30_PHASE3_DATABASE,
        settings.LOCAL_NEO4J_DATABASE,
        settings.LOCAL_NEO4J_PHASE3_DATABASE,
    ]


def main() -> None:
    driver = GraphDatabase.driver(settings.NEO4J_URI,
                                  auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    created, existing, failed = [], [], []
    try:
        with driver.session(database="system") as session:
            existing = [row["name"] for row in session.run("SHOW DATABASES YIELD name")]
            print(f"Neo4j {settings.NEO4J_URI}: {len(existing)} database(s)")
            for name in dict.fromkeys(databases()):
                if name in existing:
                    continue
                try:
                    session.run(f"CREATE DATABASE `{name}` IF NOT EXISTS").consume()
                    created.append(name)
                except Exception as error:  # noqa: BLE001 - reported below
                    failed.append((name, str(error).splitlines()[0][:120]))
    finally:
        driver.close()

    print("created: " + (", ".join(created) or "none"))
    if failed:
        print("could not create:")
        for name, reason in failed:
            print(f"  {name}: {reason}")
        print("Hint: Neo4j Community supports one user database. Use Neo4j Desktop/Enterprise "
              "or the isolated `neo4j-metaqa` service (docker compose up -d neo4j-metaqa).")


if __name__ == "__main__":
    main()
