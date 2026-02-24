import streamlit as st
from neo4j import GraphDatabase

from retrieval import (
    Retrieval,
    load_config,
    load_graph_schema,
    SCHEMA_PATH,
)

st.image("assets/impetus_logo.png", width=200)


# ---------------------------------
# Cached retriever
# ---------------------------------
@st.cache_resource
def get_retriever(
    model_name: str = "gpt-5.1",
    top_k_templates: int = 5,
    top_k_hops: int = 2,
) -> Retrieval:
    config = load_config(".env")
    schema = load_graph_schema(SCHEMA_PATH)

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )

    return Retrieval(
        driver=driver,
        graph_schema=schema,
        model_name=model_name,
        openai_api_key=config.openai_api_key,
        top_k_templates=top_k_templates,  # template selection
        top_k_hops=top_k_hops,  # graph traversal depth
        vector_top_k=5,
        fuzzy_limit=5,
        fuzzy_factor=0.8,
    )


# ---------------------------------
# Streamlit UI
# ---------------------------------
def main() -> None:
    st.set_page_config(page_title="GraphQuery AI", layout="wide")

    st.title("🤖 GraphRAG Explorer")
    st.caption(
        "Natural language → Cypher → Graph traversal → Vector context → Final answer"
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        model_name = st.text_input("Model", value="gpt-5.1")

        top_k_templates = st.slider(
            "Top-K Cypher Templates",
            min_value=1,
            max_value=20,
            value=5,
        )

        top_k_hops = st.slider(
            "Top-K Graph Hops",
            min_value=1,
            max_value=5,
            value=2,
        )

        st.divider()

        show_cypher = st.checkbox("Show generated Cypher", value=True)
        show_fuzzy = st.checkbox("Show fuzzy matches", value=False)
        show_graph_results = st.checkbox("Show raw graph results", value=False)
        show_vector_results = st.checkbox("Show vector results", value=False)

    st.divider()

    question = st.text_area(
        "Enter your question",
        value="",
        placeholder="Ask anything about the knowledge graph...",
        height=100,
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        ask_btn = st.button("Ask", type="primary", use_container_width=True)

    if ask_btn:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        try:
            retriever = get_retriever(
                model_name=model_name,
                top_k_templates=int(top_k_templates),
                top_k_hops=int(top_k_hops),
            )

            with st.spinner("Running hybrid retrieval..."):
                output = retriever.ask(question)

            st.subheader("Final Answer")
            st.write(output["final_answer"])

            if show_cypher:
                st.subheader("🕸️ Generated Cypher")
                st.code(output["cypher_query"], language="cypher")

            if show_fuzzy:
                st.subheader("📝 Fuzzy Candidates")
                st.json(output["fuzzy_candidates"])

            if show_graph_results:
                st.subheader("📦 Graph Results")
                st.json(output["graph_results"])

            if show_vector_results:
                st.subheader("📚 Vector Results")
                st.json(output["vector_results"])

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
