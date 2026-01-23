import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("neo4j-node-fuzzy-search")


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str


def load_config(env_path: str = ".env") -> Neo4jConfig:
    load_dotenv(env_path)
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    missing = [
        k
        for k, v in {
            "NEO4J_URI": uri,
            "NEO4J_USER": user,
            "NEO4J_PASSWORD": password,
        }.items()
        if not v
    ]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {missing}")

    return Neo4jConfig(uri=uri, user=user, password=password)


def run_query(
    driver: Driver, cypher: str, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    params = params or {}
    with driver.session() as session:
        result = session.run(cypher, params)
        return [r.data() for r in result]


def fuzzy_search_nodes(
    driver: Driver, user_text: str, limit: int = 10, factor: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Fulltext fuzzy search on node.search_text.
    """
    cypher = """
    CALL db.index.fulltext.queryNodes('kg_nodes_fulltext', $q)
    YIELD node, score
    RETURN labels(node) AS labels,
           score AS score,
           node.search_text AS search_text
    ORDER BY score DESC
    LIMIT $limit
    """
    q = f"{user_text}~{factor}"
    return run_query(driver, cypher, {"q": q, "limit": limit})


def main():
    config = load_config(".env")
    driver: Optional[Driver] = None

    try:
        driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))

        user_text = input("Enter fuzzy search text: ").strip()
        if not user_text:
            print("Empty input.")
            return

        results = fuzzy_search_nodes(driver, user_text, limit=10, factor=0.8)

        print("\nTop matches:\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. labels={r['labels']} score={r['score']:.3f}")
            print(r["search_text"][:300])
            print("-" * 80)

    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()
