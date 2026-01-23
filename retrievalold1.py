import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

from prompts import (
    NEO4J_TEMPLATE_BASED_CYPHER_PROMPT,
    CYPHER_TEMPLATES,
    CYPHER_QA_PROMPT,
)

SCHEMA_PATH = "graph_schema.json"


# ---------------------------
# Logging Setup (OPS Standard)
# ---------------------------
def setup_logging(
    name: str = "neo4j-retrieval",
    log_file: str = "retrieval.log",
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

    # Log file path (always in current project folder)
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


# Change level to DEBUG if you want more logs
logger = setup_logging(name="neo4j-retrieval", log_file="retrieval.log", level=logging.INFO)


# ---------------------------
# Config
# ---------------------------
@dataclass(frozen=True)
class AppConfig:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    openai_api_key: str


def load_config(env_path: str = ".env") -> AppConfig:
    """Load required configuration from a .env file."""
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
        raise EnvironmentError(f"Missing required environment variables: {missing}")

    logger.info("Loaded config from %s", env_path)

    return AppConfig(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key,
    )


def load_graph_schema(schema_path: str) -> Dict[str, Any]:
    """Load the graph schema JSON."""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    logger.info("Graph schema loaded from %s", schema_path)
    return schema


# ---------------------------
# Retrieval Class
# ---------------------------
class Retrieval:
    """
    Retrieval over Neo4j using LLM-generated Cypher and answer synthesis.
    """

    def __init__(
        self,
        driver: Driver,
        graph_schema: Dict[str, Any],
        model_name: str = "gpt-5.1",
        openai_api_key: Optional[str] = None,
        top_k: int = 5,
    ) -> None:
        if not isinstance(graph_schema, dict):
            raise TypeError("graph_schema must be a dict")

        self.driver = driver
        self.graph_schema = graph_schema
        self.model_name = model_name
        self.top_k = int(top_k)

        logger.info("Initializing Retrieval with model=%s top_k=%d", self.model_name, self.top_k)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=openai_api_key,
        )

        # Build chains
        self.nl_to_cypher_chain = RunnableSequence(
            NEO4J_TEMPLATE_BASED_CYPHER_PROMPT | self.llm
        )
        self.cypher_to_answer_chain = RunnableSequence(CYPHER_QA_PROMPT | self.llm)

        logger.info("LangChain pipelines initialized successfully")

    def close(self) -> None:
        """Close the Neo4j driver."""
        try:
            self.driver.close()
            logger.info("Neo4j driver closed successfully")
        except Exception:
            logger.exception("Failed to close Neo4j driver")

    def generate_cypher(self, question: str) -> str:
        """Generate a Cypher query from a natural language question."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        logger.info("Generating Cypher for question: %s", question)

        response = self.nl_to_cypher_chain.invoke(
            {
                "graph_schema": self.graph_schema,
                "natural_language_request": question,
                "cypher_templates": CYPHER_TEMPLATES,
                "top_k": self.top_k,
            }
        )

        cypher_query = (getattr(response, "content", "") or "").strip()

        if not cypher_query:
            raise ValueError("Generated Cypher query is empty.")

        logger.info("Cypher query generated successfully")
        logger.debug("Generated Cypher:\n%s", cypher_query)

        return cypher_query

    def run_cypher_query(
        self,
        cypher_query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query against Neo4j."""
        params = params or {}

        if not cypher_query.strip():
            raise ValueError("Cypher query cannot be empty.")

        logger.info("Running Cypher query...")
        logger.debug("Cypher Query:\n%s", cypher_query)
        logger.debug("Params: %s", params)

        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, params)
                rows = [record.data() for record in result]

            logger.info("Neo4j query executed successfully. Rows returned=%d", len(rows))
            logger.debug("Rows: %s", rows)
            return rows

        except Exception:
            logger.exception("Neo4j query failed")
            raise

    def generate_final_answer(
        self,
        question: str,
        cypher_result: Union[List[Dict[str, Any]], Dict[str, Any], str],
    ) -> str:
        """Generate final natural language answer from query results."""
        logger.info("Generating final answer from query results...")

        response = self.cypher_to_answer_chain.invoke(
            {
                "natural_language_request": question,
                "context": cypher_result,
            }
        )

        final_answer = (getattr(response, "content", "") or "").strip()

        if not final_answer:
            raise ValueError("Final response from LLM is empty.")

        logger.info("Final answer generated successfully")
        logger.debug("Final Answer:\n%s", final_answer)

        return final_answer

    def ask(self, question: str) -> Dict[str, Any]:
        """Full pipeline: NL → Cypher → Neo4j → Final Answer."""
        logger.info("Pipeline started for question: %s", question)

        cypher_query = self.generate_cypher(question)
        results = self.run_cypher_query(cypher_query)
        final_answer = self.generate_final_answer(question, results)

        logger.info("Pipeline completed successfully")
        return {
            "question": question,
            "cypher_query": cypher_query,
            "results": results,
            "final_answer": final_answer,
        }

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        schema_path: str = SCHEMA_PATH,
        model_name: str = "gpt-5.1",
        top_k: int = 5,
    ) -> "Retrieval":
        """Factory to build Retrieval from AppConfig."""
        logger.info("Building Retrieval from config...")

        graph_schema = load_graph_schema(schema_path)

        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )

        logger.info("Neo4j driver created successfully for URI=%s", config.neo4j_uri)

        return cls(
            driver=driver,
            graph_schema=graph_schema,
            model_name=model_name,
            openai_api_key=config.openai_api_key,
            top_k=top_k,
        )


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    config = load_config(".env")
    retriever = Retrieval.from_config(config, SCHEMA_PATH, model_name="gpt-5.1", top_k=5)

    try:
        question = "What are the payment terms for PO-INF-001?"
        output = retriever.ask(question)

        print("\nFinal Response:\n", output["final_answer"])

    finally:
        retriever.close()


if __name__ == "__main__":
    main()
