import os
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

from openai import OpenAI

from prompts import (
    NEO4J_TEMPLATE_BASED_CYPHER_PROMPT,
    CYPHER_TEMPLATES,
    CYPHER_QA_PROMPT,
)

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("neo4j-retrieval")

SCHEMA_PATH = "graph_schema.json"


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


def load_graph_schema(schema_path: str) -> Dict[str, Any]:
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# Retrieval Class
# ---------------------------
class Retrieval:
    """
    Hybrid Retrieval:
    1) Typo handling using Neo4j Fulltext fuzzy search on nodes (kg_nodes_fulltext)
    2) Graph Cypher query generation (LLM)
    3) Vector search from KGDocument (Neo4j vector index)
    4) Final answer generation using BOTH contexts
    """

    def __init__(
        self,
        driver: Driver,
        graph_schema: Dict[str, Any],
        model_name: str = "gpt-5.1",
        openai_api_key: Optional[str] = None,
        top_k_templates: int = 5,
        vector_top_k: int = 5,
        fuzzy_limit: int = 5,
        fuzzy_factor: float = 0.8,
    ) -> None:
        self.driver = driver
        self.graph_schema = graph_schema
        self.model_name = model_name

        self.top_k_templates = int(top_k_templates)
        self.vector_top_k = int(vector_top_k)

        self.fuzzy_limit = int(fuzzy_limit)
        self.fuzzy_factor = float(fuzzy_factor)

        # LLM for Cypher + QA
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=openai_api_key,
        )

        # OpenAI client for embeddings
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Chains
        self.nl_to_cypher_chain = RunnableSequence(
            NEO4J_TEMPLATE_BASED_CYPHER_PROMPT | self.llm
        )
        self.cypher_to_answer_chain = RunnableSequence(CYPHER_QA_PROMPT | self.llm)

    def close(self) -> None:
        try:
            self.driver.close()
            logger.info("Neo4j driver closed")
        except Exception as e:
            logger.warning("Failed to close Neo4j driver: %s", e)

    # ---------------------------
    # Neo4j helpers
    # ---------------------------
    def run_query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        params = params or {}
        with self.driver.session() as session:
            result = session.run(cypher, params)
            return [r.data() for r in result]

    # ---------------------------
    # 1) Fulltext fuzzy search (typo handling)
    # ---------------------------
    def fuzzy_node_search(self, user_text: str) -> List[Dict[str, Any]]:
        """
        Search nodes using fulltext index kg_nodes_fulltext.
        Returns best candidate nodes that match question text (typo tolerant).
        """
        if not user_text.strip():
            return []

        q = f"{user_text}~{self.fuzzy_factor}"

        cypher = """
        CALL db.index.fulltext.queryNodes("kg_nodes_fulltext", $q)
        YIELD node, score
        RETURN labels(node) AS labels,
               score AS score,
               node.search_text AS search_text
        ORDER BY score DESC
        LIMIT $limit
        """

        results = self.run_query(cypher, {"q": q, "limit": self.fuzzy_limit})
        logger.info("Fuzzy search returned %d candidates", len(results))
        return results

    # ---------------------------
    # 2) NL -> Cypher generation (LLM)
    # ---------------------------
    def generate_cypher(
        self, question: str, fuzzy_context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate Cypher query. We add fuzzy_context into schema prompt
        so LLM can use corrected entities.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("Question cannot be empty")

        logger.info("Generating Cypher for question: %s", question)

        response = self.nl_to_cypher_chain.invoke(
            {
                "graph_schema": self.graph_schema,
                "natural_language_request": question,
                "cypher_templates": CYPHER_TEMPLATES,
                "top_k": self.top_k_templates,
                # Add fuzzy candidates to prompt (very useful)
                "fuzzy_candidates": fuzzy_context or [],
            }
        )

        cypher_query = (getattr(response, "content", "") or "").strip()
        if not cypher_query:
            raise ValueError("Generated Cypher query is empty")

        logger.info("Generated Cypher:\n%s", cypher_query)
        return cypher_query

    # ---------------------------
    # 3) Execute Cypher on graph
    # ---------------------------
    def run_cypher_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        if not cypher_query.strip():
            raise ValueError("Cypher query cannot be empty")

        logger.info("Executing Cypher query...")
        try:
            rows = self.run_query(cypher_query)
            logger.info("Graph query returned %d rows", len(rows))
            return rows
        except Exception as e:
            logger.error("Cypher execution failed: %s", e)
            raise

    # ---------------------------
    # 4) Vector search on KGDocument
    # ---------------------------
    def embed_question(self, question: str) -> List[float]:
        resp = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=question,
        )
        return resp.data[0].embedding

    def vector_search(self, question: str) -> List[Dict[str, Any]]:
        """
        Vector similarity search in Neo4j against :KGDocument nodes.
        Requires vector index: kgdoc_embedding_index
        """
        logger.info("Running vector search for additional context...")
        q_embedding = self.embed_question(question)

        cypher = """
        CALL db.index.vector.queryNodes("kgdoc_embedding_index", $k, $embedding)
        YIELD node, score
        RETURN node.doc_id AS doc_id,
               node.text AS text,
               node.source_label AS source_label,
               node.source_key AS source_key,
               node.source_value AS source_value,
               score AS score
        ORDER BY score DESC
        LIMIT $k
        """

        results = self.run_query(
            cypher,
            {"k": self.vector_top_k, "embedding": q_embedding},
        )

        logger.info("Vector search returned %d docs", len(results))
        return results

    # ---------------------------
    # 5) Final answer generation
    # ---------------------------
    def generate_final_answer(
        self,
        question: str,
        graph_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
    ) -> str:
        """
        Final answer uses BOTH:
        - graph_results (structured truth)
        - vector_results (extra semantic context)
        """
        logger.info("Generating final answer...")

        context = {
            "graph_results": graph_results,
            "vector_results": vector_results,
        }

        response = self.cypher_to_answer_chain.invoke(
            {
                "natural_language_request": question,
                "context": context,
            }
        )

        final_answer = (getattr(response, "content", "") or "").strip()
        if not final_answer:
            raise ValueError("Final response from LLM is empty")

        return final_answer

    # ---------------------------
    # Full pipeline
    # ---------------------------
    def ask(self, question: str) -> Dict[str, Any]:
        logger.info("----- NEW QUESTION -----")
        logger.info("User Question: %s", question)

        # Step 1: fuzzy candidates (typo help)
        fuzzy_candidates = self.fuzzy_node_search(question)

        # Step 2: Cypher generation with fuzzy hints
        cypher_query = self.generate_cypher(question, fuzzy_candidates)

        # Step 3: Graph execution
        graph_results = self.run_cypher_query(cypher_query)

        # Step 4: Vector search (always)
        vector_results = self.vector_search(question)

        # Step 5: Final answer
        final_answer = self.generate_final_answer(
            question, graph_results, vector_results
        )

        return {
            "question": question,
            "fuzzy_candidates": fuzzy_candidates,
            "cypher_query": cypher_query,
            "graph_results": graph_results,
            "vector_results": vector_results,
            "final_answer": final_answer,
        }

    # ---------------------------
    # Factory
    # ---------------------------
    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        schema_path: str = SCHEMA_PATH,
        model_name: str = "gpt-5.1",
        top_k_templates: int = 5,
        vector_top_k: int = 5,
        fuzzy_limit: int = 5,
        fuzzy_factor: float = 0.8,
    ) -> "Retrieval":
        graph_schema = load_graph_schema(schema_path)
        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        return cls(
            driver=driver,
            graph_schema=graph_schema,
            model_name=model_name,
            openai_api_key=config.openai_api_key,
            top_k_templates=top_k_templates,
            vector_top_k=vector_top_k,
            fuzzy_limit=fuzzy_limit,
            fuzzy_factor=fuzzy_factor,
        )


# ---------------------------
# CLI Test
# ---------------------------
def main() -> None:
    config = load_config(".env")

    retriever = Retrieval.from_config(
        config=config,
        schema_path=SCHEMA_PATH,
        model_name="gpt-5.1",
        top_k_templates=5,
        vector_top_k=5,
        fuzzy_limit=5,
        fuzzy_factor=0.8,
    )

    try:
        while True:
            q = input("\nAsk a question (or type exit): ").strip()
            if not q or q.lower() == "exit":
                break

            out = retriever.ask(q)

            print("\n--- Generated Cypher ---")
            print(out["cypher_query"])

            print("\n--- Graph Results ---")
            print(out["graph_results"])

            print("\n--- Vector Results (Top) ---")
            for d in out["vector_results"][:3]:
                print(f"- {d['doc_id']} score={d['score']:.3f}")

            print("\n--- Final Answer ---")
            print(out["final_answer"])

    finally:
        retriever.close()


if __name__ == "__main__":
    main()
