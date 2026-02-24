\# 🤖 GraphRAG Explorer – Fine-Grained Knowledge Graph + Hybrid RAG



A Hybrid Graph + Vector Retrieval Augmented Generation (GraphRAG) Proof of Concept built using:



\- 🐍 Python 3.11.5  

\- 🧠 OpenAI (LLM + Embeddings)  

\- 🗄 Neo4j (Graph + Fulltext + Native Vector Index)  

\- 🎨 Streamlit (UI Layer)  



This system combines:



\- Structured Knowledge Graph reasoning  

\- Fine-Grained node modeling  

\- Lucene-based fuzzy search  

\- Vector semantic retrieval  

\- Multi-hop graph traversal  

\- LLM-based context fusion  



---



\# 🚀 Key Features



\- Natural Language → Cypher generation  

\- Multi-hop graph traversal (Top-K Hops configurable)  

\- Fulltext typo-tolerant search (Lucene fuzzy)  

\- Native Neo4j vector similarity search  

\- Hybrid Graph + Vector context fusion  

\- Fine-Grained Knowledge Graph modeling  

\- Enterprise logging \& configuration  

\- Streamlit interactive UI  



---



\# 🧠 Fine-Grained Knowledge Graph Approach



This project uses a \*\*Fine-Grained Knowledge Graph (FG-KG)\*\* design instead of a document-centric approach.



\## What is Fine-Grained KG?



Fine-Grained Knowledge Graph means:



\- Each business entity is modeled as a separate node  

\- Relationships represent real-world semantic connections  

\- Properties are stored at entity-level granularity  

\- Graph structure reflects actual business logic  



\## Why Fine-Grained KG is Powerful



| Coarse Document Graph | Fine-Grained Knowledge Graph |

|------------------------|------------------------------|

| Document-level nodes | Entity-level nodes |

| Limited relationship reasoning | Strong semantic reasoning |

| Hard aggregation queries | Native aggregation support |

| Mostly vector-driven | Graph-native + vector hybrid |

| Higher hallucination risk | Deterministic structured queries |



---



\## Benefits of Fine-Grained Approach



\- Precise multi-hop traversal  

\- Clean aggregations (SUM, AVG, COUNT)  

\- Better explainability  

\- Reduced hallucination risk  

\- Strong schema control  

\- Easier debugging  

\- Natural graph expansion  



---



\# 🏗 System Architecture



User Question  

&nbsp;       ↓  

Fuzzy Fulltext Search (Lucene)  

&nbsp;       ↓  

LLM → Cypher Generation  

&nbsp;       ↓  

Neo4j Graph Execution  

&nbsp;       ↓  

Vector Similarity Search (KGDocument)  

&nbsp;       ↓  

LLM Context Fusion  

&nbsp;       ↓  

Final Answer  



---



\# 🔎 Hybrid Retrieval Strategy



This system uses two complementary retrieval mechanisms:



\## 1️⃣ Structured Graph Retrieval

\- LLM-generated Cypher

\- Multi-hop traversal

\- Schema-aware reasoning

\- Deterministic results



\## 2️⃣ Semantic Vector Retrieval

\- `text-embedding-3-large`

\- 3072 dimension embeddings

\- Cosine similarity search

\- Context enrichment layer



The final answer is generated using both graph results and semantic vector context.



---



\# 🐍 Python Version



This project is built and tested on:



```

Python 3.11.5

```



Check your version:



```

python --version

```



---



\# 🔧 Installation



\## 1️⃣ Clone Repository



```

git clone https://github.com/your-org/neo4j-graph-rag-poc.git

cd neo4j-graph-rag-poc

```



---



\## 2️⃣ Create Virtual Environment



Windows:



```

python -m venv venv

venv\\Scripts\\activate

```



Mac/Linux:



```

python -m venv venv

source venv/bin/activate

```



---



\## 3️⃣ Install Dependencies



```

pip install -r requirements.txt

```



Example requirements.txt:



```

streamlit

neo4j

openai

langchain

langchain-openai

python-dotenv

httpx

```



---



\# 🔐 Environment Configuration



Create a `.env` file:



```

NEO4J\_URI=bolt://localhost:7687

NEO4J\_USER=neo4j

NEO4J\_PASSWORD=your\_password

OPENAI\_API\_KEY=your\_openai\_key

```



---



\# 🗄 Neo4j Setup



\## Create Fulltext Index



```

CREATE FULLTEXT INDEX kg\_nodes\_fulltext

FOR (n:PurchaseOrder|PurchaseRequest|Invoice|Contract)

ON EACH \[n.search\_text];

```



---



\## Create Vector Index



```

CREATE VECTOR INDEX kgdoc\_embedding\_index

FOR (d:KGDocument)

ON d.embedding

OPTIONS {

&nbsp; indexConfig: {

&nbsp;   `vector.dimensions`: 3072,

&nbsp;   `vector.similarity\_function`: 'cosine'

&nbsp; }

};

```



---



\## Verify Indexes



```

SHOW INDEXES;

```



Both indexes must show state: ONLINE



---



\# ▶ Running the Application



```

streamlit run retrieval\_UI.py

```



Open in browser:



```

http://localhost:8504

```



---



\# ⚙️ Configurable Controls (UI)



\- Top-K Templates  

\- Top-K Hops  

\- Vector Top-K  

\- Show Generated Cypher  

\- Show Fuzzy Candidates  

\- Show Graph Results  

\- Show Vector Results  



---



\# 🔐 SSL Handling (Corporate Networks)



This project uses:



```

httpx.Client(verify=False)

```



This disables SSL verification for internal enterprise environments.



⚠ Not recommended for production use.



For production environments use:



```

httpx.Client(verify="path/to/cert.pem")

```



---



\# 📂 Project Structure



```

.

├── retrieval.py

├── retrieval\_UI.py

├── prompts.py

├── graph\_schema.json

├── Data/

├── assets/

│   └── logo.png

├── requirements.txt

└── README.md

```



---



\# 🧪 Example Questions



\- What are the payment terms for PO-INF-002?

\- What is the total OPEX spend?

\- Which contract talks about disaster recovery?

\- What percentage of PurchaseOrders are high risk?

\- If OPEX increases by 7%, what is the projected spend?



---



\# 🛡 Production Considerations



\- Replace verify=False with proper SSL certificate

\- Add retry and timeout tuning

\- Enable caching

\- Add monitoring

\- Secure API keys properly

\- Add authentication to UI



---



\# 👤 Author



Sajid Choudhary  

Knowledge Graph + Generative AI Engineering  

