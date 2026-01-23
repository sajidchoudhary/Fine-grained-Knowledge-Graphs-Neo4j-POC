import os
import json
import math
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver


# ---------------------------
# Logging Setup (OPS Standard)
# ---------------------------
def setup_logging(
    name: str = "neo4j-ingest",
    log_file: str = "neo4j_ingest.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Creates logger with both Console + File handlers.
    Ensures logs are printed and file is created.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers (prevents duplicates if re-run)
    if logger.handlers:
        logger.handlers.clear()

    # Save log file in same directory as script
    log_path = Path(__file__).parent / log_file

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Logging initialized. Log file: %s", log_path.resolve())
    return logger


# Change to logging.DEBUG if you want more detailed logs
logger = setup_logging(name="neo4j-ingest", log_file="neo4j_ingest.log", level=logging.INFO)


# ---------------------------
# Config
# ---------------------------
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
        k for k, v in {"NEO4J_URI": uri, "NEO4J_USER": user, "NEO4J_PASSWORD": password}.items() if not v
    ]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {missing}")

    logger.info("Loaded Neo4j config from %s", env_path)
    return Neo4jConfig(uri=uri, user=user, password=password)


# ---------------------------
# Helpers
# ---------------------------
def clean_props(props: Any) -> Any:
    """Replace NaN with None so Neo4j can store it as null. Works recursively for dict/list."""
    if isinstance(props, dict):
        return {k: clean_props(v) for k, v in props.items()}

    if isinstance(props, list):
        return [clean_props(x) for x in props]

    if isinstance(props, float) and math.isnan(props):
        return None

    return props


def get_unique_key(props: Dict[str, Any]) -> Tuple[str, Any]:
    """Pick best unique key for node merge."""
    if not props:
        raise ValueError("Node properties are empty. Cannot determine unique key.")

    lookup_key = props.get("lookupKey")
    if isinstance(lookup_key, str) and lookup_key in props and props.get(lookup_key) is not None:
        return lookup_key, props[lookup_key]

    for k in (
        "purchase_order_number",
        "purchase_requisition_id",
        "invoice_reconciliation_id",
        "contract_number",
        "id",
        "name",
    ):
        if k in props and props.get(k) is not None:
            return k, props[k]

    first_key = next(iter(props.keys()))
    return first_key, props[first_key]


def validate_row(row: Dict[str, Any]) -> None:
    required = {"src_label", "tgt_label", "edge_label", "src_props", "tgt_props"}
    missing = required - set(row.keys())
    if missing:
        raise ValueError(f"Invalid triplet row. Missing keys: {missing}")

    if not isinstance(row["src_props"], dict) or not isinstance(row["tgt_props"], dict):
        raise ValueError("src_props and tgt_props must be dict objects.")


# ---------------------------
# Neo4j Ingestion (Batch)
# ---------------------------
def ingest_triplets_batch(tx, rows: List[Dict[str, Any]]) -> None:
    """
    Batch ingestion using UNWIND for performance.
    Note: Labels and relationship types cannot be parameterized in Cypher,
    so we group by (src_label, tgt_label, rel_type).
    """
    if not rows:
        return

    row0 = rows[0]
    src_label = row0["src_label"]
    tgt_label = row0["tgt_label"]
    rel_type = row0["edge_label"]

    query = f"""
    UNWIND $rows AS row

    MERGE (s:{src_label} {{ _merge_key: row.src_merge_key }})
    SET s += row.src_props

    MERGE (t:{tgt_label} {{ _merge_key: row.tgt_merge_key }})
    SET t += row.tgt_props

    MERGE (s)-[r:{rel_type}]->(t)
    SET r += row.rel_props
    """

    tx.run(query, rows=rows)


def group_triplets(triplets: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """Group by (src_label, tgt_label, edge_label) because labels/types can't be parameterized."""
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for t in triplets:
        key = (t["src_label"], t["tgt_label"], t["edge_label"])
        grouped.setdefault(key, []).append(t)
    return grouped


def prepare_triplet(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize + clean a single row for batch ingestion."""
    validate_row(row)

    src_props = clean_props(row["src_props"])
    tgt_props = clean_props(row["tgt_props"])
    rel_props = clean_props(row.get("edge_props", {})) or {}

    src_key, src_val = get_unique_key(src_props)
    tgt_key, tgt_val = get_unique_key(tgt_props)

    # store original key info also (helpful for debugging)
    src_props["_unique_key"] = src_key
    tgt_props["_unique_key"] = tgt_key

    return {
        "src_label": row["src_label"],
        "tgt_label": row["tgt_label"],
        "edge_label": row["edge_label"],
        "src_merge_key": str(src_val),
        "tgt_merge_key": str(tgt_val),
        "src_props": src_props,
        "tgt_props": tgt_props,
        "rel_props": rel_props,
    }


def chunked(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------
# Main
# ---------------------------
def load_triplets(json_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Triplet file not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("ontology.json must contain a JSON list of triplets.")

    return data


def main(
    ontology_file: str = "Data/ontology.json",
    batch_size: int = 200,
) -> None:
    logger.info("Starting ingestion job...")
    logger.info("Ontology file=%s | batch_size=%d", ontology_file, batch_size)

    config = load_config(".env")

    raw_triplets = load_triplets(ontology_file)
    logger.info("Loaded %d triplets from %s", len(raw_triplets), ontology_file)

    # prepare + clean
    prepared = []
    failed = 0

    for idx, row in enumerate(raw_triplets, start=1):
        try:
            prepared.append(prepare_triplet(row))
        except Exception:
            failed += 1
            logger.exception("Failed to prepare triplet at index=%d", idx)

    logger.info("Prepared triplets=%d | Failed=%d", len(prepared), failed)

    grouped = group_triplets(prepared)
    logger.info("Total groups=%d", len(grouped))

    driver: Optional[Driver] = None
    try:
        logger.info("Connecting to Neo4j: %s", config.uri)
        driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))

        total_inserted = 0

        with driver.session() as session:
            for (src_label, tgt_label, rel_type), rows in grouped.items():
                logger.info(
                    "Ingesting group: (%s)-[%s]->(%s) | rows=%d",
                    src_label,
                    rel_type,
                    tgt_label,
                    len(rows),
                )

                for batch in chunked(rows, batch_size):
                    try:
                        session.execute_write(ingest_triplets_batch, batch)
                        total_inserted += len(batch)

                        if total_inserted % 500 == 0:
                            logger.info("Inserted %d triplets so far...", total_inserted)

                    except Exception:
                        logger.exception(
                            "Failed batch insert for group (%s)-[%s]->(%s)",
                            src_label,
                            rel_type,
                            tgt_label,
                        )

        logger.info("Done. Inserted %d triplets into Neo4j.", total_inserted)

    except Exception:
        logger.exception("Ingestion job failed")
        raise

    finally:
        if driver:
            driver.close()
            logger.info("Neo4j driver closed")


if __name__ == "__main__":
    main()
