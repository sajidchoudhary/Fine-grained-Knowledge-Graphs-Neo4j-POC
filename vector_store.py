import os
import json
import time
import math
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from openai import OpenAI


# one time run to create vector index in neo4j
"""
CREATE VECTOR INDEX kgdoc_embedding_index
FOR (d:KGDocument) ON (d.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
};


"""


# ============================
# Logging (OPS Standard)
# ============================
def setup_logging(
    name: str = "neo4j-vector-ingest",
    log_file: str = "vector_ingest.log",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    log_path = Path(__file__).parent / log_file

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    logger.info("Logging initialized. Log file: %s", log_path.resolve())
    return logger


logger = setup_logging(level=logging.INFO)


# ============================
# Config
# ============================
@dataclass(frozen=True)
class AppConfig:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    openai_api_key: str
    data_file: str = "data.json"
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072


def load_config(env_path: str = ".env") -> AppConfig:
    load_dotenv(env_path)

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    missing = [
        k
        for k, v in {
            "NEO4J_URI": neo4j_uri,
            "NEO4J_USER": neo4j_user,
            "NEO4J_PASSWORD": neo4j_password,
            "OPENAI_API_KEY": openai_api_key,
        }.items()
        if not v
    ]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {missing}")

    return AppConfig(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key,
    )


# ============================
# Utilities
# ============================
def clean_props(value: Any) -> Any:
    """Replace NaN with None so Neo4j can store null."""
    if isinstance(value, dict):
        return {k: clean_props(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_props(x) for x in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def stable_id(label: str, key: str, value: Any) -> str:
    """Create stable id like PurchaseOrder:purchase_order_number:PO-INF-001"""
    return f"{label}:{key}:{value}"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_unique_key(props: Dict[str, Any]) -> Tuple[str, Any]:
    """Get stable unique key for node identification."""
    if not props:
        raise ValueError("Empty properties. Cannot determine unique key.")

    if "lookupKey" in props and props["lookupKey"] in props:
        k = props["lookupKey"]
        return k, props[k]

    for k in [
        "purchase_order_number",
        "purchase_requisition_id",
        "invoice_reconciliation_id",
        "contract_number",
    ]:
        if k in props and props.get(k) is not None:
            return k, props[k]

    first_key = next(iter(props.keys()))
    return first_key, props[first_key]


def build_text(label: str, props: Dict[str, Any]) -> str:
    """Build meaningful text for embeddings."""
    parts = [f"Label: {label}"]

    preferred = [
        "purchase_order_title",
        "purchase_requisition_title",
        "agreement_name",
        "agreement_description",
        "line_item_description",
        "invoice_purpose",
    ]
    for k in preferred:
        v = props.get(k)
        if v:
            parts.append(f"{k}: {v}")

    context_keys = [
        "purchase_order_number",
        "purchase_requisition_id",
        "invoice_number",
        "contract_number",
        "supplier_id",
        "currency",
        "payment_terms",
    ]
    for k in context_keys:
        v = props.get(k)
        if v:
            parts.append(f"{k}: {v}")

    return "\n".join(parts)


def load_triplets(data_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("data.json must contain a JSON list of triplets.")

    return data


# ============================
# Embedding Service
# ============================
class Embedder:
    def __init__(
        self, api_key: str, model: str, max_retries: int = 3, backoff_sec: float = 2.0
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec

    def embed(self, text: str) -> List[float]:
        """Embed text with retry."""
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.embeddings.create(model=self.model, input=text)
                return resp.data[0].embedding
            except Exception as e:
                last_err = e
                logger.warning(
                    "Embedding failed (attempt %d/%d): %s", attempt, self.max_retries, e
                )
                time.sleep(self.backoff_sec * attempt)

        raise RuntimeError(
            f"Embedding failed after {self.max_retries} attempts"
        ) from last_err


# ============================
# Neo4j Storage
# ============================
def upsert_documents_batch(tx, docs):
    query = """
    UNWIND $docs AS doc
    MERGE (d:KGDocument {doc_id: doc.doc_id})
    SET d.source_label = doc.source_label,
        d.source_key = doc.source_key,
        d.source_value = doc.source_value,
        d.text = doc.text,
        d.text_hash = doc.text_hash,
        d.embedding = doc.embedding,
        d.props_json = doc.props_json
    """
    tx.run(query, docs=docs)


def fetch_existing_hashes(tx, doc_ids: List[str]) -> Dict[str, str]:
    """
    Fetch existing text_hash to skip re-embedding if unchanged.
    """
    query = """
    UNWIND $doc_ids AS id
    MATCH (d:KGDocument {doc_id: id})
    RETURN d.doc_id AS doc_id, d.text_hash AS text_hash
    """
    result = tx.run(query, doc_ids=doc_ids)
    return {r["doc_id"]: r["text_hash"] for r in result}


def chunked(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ============================
# Pipeline
# ============================
def build_documents(triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract unique nodes from src + tgt and convert into KGDocument payloads.
    """
    docs: List[Dict[str, Any]] = []
    seen = set()

    for row in triplets:
        for side in ("src", "tgt"):
            label = row.get(f"{side}_label")
            props = row.get(f"{side}_props")

            if not label or not isinstance(props, dict):
                continue

            props = clean_props(props)
            key, value = get_unique_key(props)
            doc_id = stable_id(label, key, value)

            if doc_id in seen:
                continue
            seen.add(doc_id)

            text = build_text(label, props)
            docs.append(
                {
                    "doc_id": doc_id,
                    "source_label": label,
                    "source_key": key,
                    "source_value": str(value),
                    "text": text,
                    "text_hash": hash_text(text),
                    "props_json": json.dumps(props, ensure_ascii=False),
                }
            )
    return docs


def main(data_file: str = "data.json", batch_size: int = 20) -> None:
    logger.info("Starting vector ingestion job...")
    config = load_config(".env")

    triplets = load_triplets(data_file)
    logger.info("Loaded %d triplets from %s", len(triplets), data_file)

    docs = build_documents(triplets)
    logger.info("Extracted %d unique nodes for embedding", len(docs))

    embedder = Embedder(api_key=config.openai_api_key, model=config.embedding_model)

    driver: Optional[Driver] = None
    try:
        logger.info("Connecting to Neo4j: %s", config.neo4j_uri)
        driver = GraphDatabase.driver(
            config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
        )

        total_written = 0
        total_skipped = 0

        with driver.session() as session:
            # Pre-check existing hashes in Neo4j (skip unchanged)
            doc_ids = [d["doc_id"] for d in docs]
            existing_hashes = session.execute_read(fetch_existing_hashes, doc_ids)
            logger.info("Found %d existing KGDocument nodes", len(existing_hashes))

            to_process: List[Dict[str, Any]] = []
            for d in docs:
                existing_hash = existing_hashes.get(d["doc_id"])
                if existing_hash and existing_hash == d["text_hash"]:
                    total_skipped += 1
                    continue
                to_process.append(d)

            logger.info(
                "To embed=%d | skipped (unchanged)=%d", len(to_process), total_skipped
            )

            for batch in chunked(to_process, batch_size):
                for doc in batch:
                    doc["embedding"] = embedder.embed(doc["text"])

                session.execute_write(upsert_documents_batch, batch)
                total_written += len(batch)

                logger.info("Upserted %d/%d documents", total_written, len(to_process))

        logger.info("Done. Upserted=%d | Skipped=%d", total_written, total_skipped)

    except Exception:
        logger.exception("Vector ingestion failed")
        raise

    finally:
        if driver:
            driver.close()
            logger.info("Neo4j driver closed")


if __name__ == "__main__":
    main(data_file="Data/ontology.json", batch_size=10)
