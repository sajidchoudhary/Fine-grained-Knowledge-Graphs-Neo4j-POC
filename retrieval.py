import os
import json
import logging
import re
import httpx
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

    LUCENE_SPECIAL = r'[\+\-\!\(\)\{\}\[\]\^"~\*\?:\\/]|&&|\|\|'

    def __init__(
        self,
        driver: Driver,
        graph_schema: Dict[str, Any],
        model_name: str = "gpt-5.1",
        openai_api_key: Optional[str] = None,
        top_k_templates: int = 5,
        top_k_hops: int = 2,
        vector_top_k: int = 5,
        fuzzy_limit: int = 5,
        fuzzy_factor: float = 0.8,
    ) -> None:

        self.driver = driver
        self.graph_schema = graph_schema
        self.model_name = model_name

        self.top_k_templates = int(top_k_templates)
        self.top_k_hops = int(top_k_hops)
        self.vector_top_k = int(vector_top_k)

        self.fuzzy_limit = int(fuzzy_limit)
        self.fuzzy_factor = float(fuzzy_factor)

        # 🔹 Shared HTTP client (SSL disabled)
        self.http_client = httpx.Client(verify=False)

        # 🔹 LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=openai_api_key,
            http_client=self.http_client,
        )

        # 🔹 Embedding client
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            http_client=self.http_client,
        )

        # Chains
        self.nl_to_cypher_chain = RunnableSequence(
            NEO4J_TEMPLATE_BASED_CYPHER_PROMPT | self.llm
        )

        self.cypher_to_answer_chain = RunnableSequence(CYPHER_QA_PROMPT | self.llm)

    # ---------------------------
    # Utility
    # ---------------------------
    @staticmethod
    def _sanitize_token(token: str) -> str:
        token = re.sub(Retrieval.LUCENE_SPECIAL, " ", token)
        token = re.sub(r"[^a-zA-Z0-9]", "", token)
        return token.strip()

    def _build_fuzzy_query(self, text: str) -> str:
        raw_tokens = text.lower().split()
        tokens = []
        for t in raw_tokens:
            clean = self._sanitize_token(t)
            if len(clean) >= 3:
                tokens.append(f"{clean}~{self.fuzzy_factor}")
        return " ".join(tokens) if tokens else text

    # ---------------------------
    # Neo4j Helper
    # ---------------------------
    def run_query(self, cypher: str, params=None):
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [r.data() for r in result]

    # ---------------------------
    # Fuzzy Search
    # ---------------------------
    def fuzzy_node_search(self, user_text: str):
        if not user_text.strip():
            return []

        fuzzy_q = self._build_fuzzy_query(user_text)
        logger.info("Fuzzy query: %s", fuzzy_q)

        cypher = """
        CALL db.index.fulltext.queryNodes("kg_nodes_fulltext", $q)
        YIELD node, score
        RETURN labels(node) AS labels,
               score,
               node.search_text AS search_text
        ORDER BY score DESC
        LIMIT $limit
        """

        return self.run_query(
            cypher,
            {"q": fuzzy_q, "limit": self.fuzzy_limit},
        )

    # ---------------------------
    # Cypher Generation
    # ---------------------------
    def generate_cypher(self, question, fuzzy_context):
        response = self.nl_to_cypher_chain.invoke(
            {
                "graph_schema": self.graph_schema,
                "natural_language_request": question,
                "cypher_templates": CYPHER_TEMPLATES,
                "top_k": self.top_k_templates,
                "max_hops": self.top_k_hops,
                "fuzzy_candidates": fuzzy_context or [],
            }
        )

        cypher = (getattr(response, "content", "") or "").strip()

        if not cypher or cypher.lower() == "none":
            raise ValueError(f"Invalid Cypher generated: {cypher}")

        logger.info("Generated Cypher:\n%s", cypher)
        return cypher

    # ---------------------------
    # Execute Cypher
    # ---------------------------
    def run_cypher_query(self, cypher_query: str):
        logger.info("Executing Cypher...")
        return self.run_query(cypher_query)

    # ---------------------------
    # Vector Search
    # ---------------------------
    def embed_question(self, question: str):
        resp = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=question,
        )
        return resp.data[0].embedding

    def vector_search(self, question: str):
        logger.info("Running vector search...")
        emb = self.embed_question(question)

        cypher = """
        CALL db.index.vector.queryNodes("kgdoc_embedding_index", $k, $embedding)
        YIELD node, score
        RETURN node.doc_id AS doc_id,
               node.text AS text,
               score
        ORDER BY score DESC
        LIMIT $k
        """

        return self.run_query(
            cypher,
            {"k": self.vector_top_k, "embedding": emb},
        )

    # ---------------------------
    # Final Answer
    # ---------------------------
    def generate_final_answer(self, question, graph, vector):
        response = self.cypher_to_answer_chain.invoke(
            {
                "natural_language_request": question,
                "context": {
                    "graph_results": graph,
                    "vector_results": vector,
                },
            }
        )
        return (getattr(response, "content", "") or "").strip()

    # ---------------------------
    # Full Pipeline
    # ---------------------------
    def ask(self, question: str):

        logger.info("New question: %s", question)

        fuzzy = self.fuzzy_node_search(question)
        cypher = self.generate_cypher(question, fuzzy)
        graph = self.run_cypher_query(cypher)
        vector = self.vector_search(question)
        answer = self.generate_final_answer(question, graph, vector)

        return {
            "question": question,
            "fuzzy_candidates": fuzzy,
            "cypher_query": cypher,
            "graph_results": graph,
            "vector_results": vector,
            "final_answer": answer,
        }

    def close(self):
        self.driver.close()
        self.http_client.close()
        logger.info("Neo4j driver and HTTP client closed")


# ---------------------------
# CLI Mode
# ---------------------------
def main():
    config = load_config(".env")

    retriever = Retrieval(
        driver=GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        ),
        graph_schema=load_graph_schema(SCHEMA_PATH),
        openai_api_key=config.openai_api_key,
    )

    try:
        while True:
            q = input("\nAsk a question (exit to quit): ").strip()
            if q.lower() == "exit":
                break

            out = retriever.ask(q)

            print("\n--- Cypher ---")
            print(out["cypher_query"])

            print("\n--- Final Answer ---")
            print(out["final_answer"])

    finally:
        retriever.close()


if __name__ == "__main__":
    main()
