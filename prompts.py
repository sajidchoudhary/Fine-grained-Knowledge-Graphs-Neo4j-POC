from langchain_core.prompts import PromptTemplate

CYPHER_TEMPLATES = [
    # 1) Simple node fetch (return full node)
    "MATCH (n:{{node_label}}) RETURN n",
    # 2) Node fetch with filter (works for numeric/date if filter_value is formatted correctly)
    "MATCH (n:{{node_label}}) WHERE n.{{filter_property}} {{filter_operator}} {{filter_value}} RETURN n",
    # 3) Relationship traversal (return full nodes + relationship)
    "MATCH (a:{{from_node_label}})-[r:{{relationship_type}}]->(b:{{to_node_label}}) RETURN a, r, b",
    # 4) Relationship traversal with filter on source node
    "MATCH (a:{{from_node_label}})-[r:{{relationship_type}}]->(b:{{to_node_label}}) WHERE a.{{filter_property}} {{filter_operator}} {{filter_value}} RETURN a, r, b",
    # 5) Relationship traversal with filter on target node
    "MATCH (a:{{from_node_label}})-[r:{{relationship_type}}]->(b:{{to_node_label}}) WHERE b.{{filter_property}} {{filter_operator}} {{filter_value}} RETURN a, r, b",
    # 6) Count relationships per node (Neo4j-valid ORDER BY)
    "MATCH (a:{{from_node_label}})-[r:{{relationship_type}}]->(b:{{to_node_label}}) RETURN a, COUNT(b) AS {{return_property_alias}} ORDER BY 2 DESC",
    # 7) Undirected relationship match
    "MATCH (a:{{from_node_label}})-[r:{{relationship_type}}]-(b:{{to_node_label}}) RETURN a, r, b",
    # 8) Variable-length traversal (fixed syntax)
    "MATCH (a:{{start_label}})-[r*1..{{hops}}]-(n) WHERE a.{{filter_property}} {{filter_operator}} {{filter_value}} RETURN a, r, n",
    # 9) Filter on both ends (kept same property placeholder on both sides)
    "MATCH (a:{{from_node_label}})-[r:{{relationship_type}}]->(b:{{to_node_label}}) WHERE a.{{filter_property}} {{filter_operator}} {{filter_value}} AND b.{{filter_property}} {{filter_operator}} {{filter_value}} RETURN a, r, b",
    # 10) Two-hop traversal
    "MATCH (a:{{start_label}})-[r1:{{relationship_type}}]->(b:{{middle_label}})-[r2:{{relationship_type}}]->(c:{{end_label}}) RETURN a, r1, b, r2, c",
    # 11) Case-insensitive partial match (Neo4j-valid)
    "MATCH (n:{{node_label}}) WHERE toLower(n.{{filter_property}}) CONTAINS toLower({{filter_value}}) RETURN n",
]


NEO4J_TEMPLATE_BASED_CYPHER_PROMPT_TEMPLATE = """
You are an expert Cypher query generator for Neo4j.

### IMPORTANT RULES (must follow strictly):
1. You MUST generate Cypher by selecting ONE of the provided Cypher Templates and filling its placeholders.
2. You MUST output ONLY the final Cypher query text (no JSON, no explanation, no markdown).
3. You MUST generate ONLY read-only queries:
   Allowed: MATCH, OPTIONAL MATCH, WHERE, RETURN, ORDER BY, LIMIT, COUNT
   Not allowed: CREATE, MERGE, DELETE, SET, REMOVE, CALL, LOAD CSV, apoc.*
4. Never use RETURN *.
5. In RETURN clause, always return complete nodes (e.g., RETURN po, inv) not single properties.
6. Use ONLY labels, relationship types, and properties that exist in the Graph Schema.
7. If the filter value is a string, you MUST use the case-insensitive partial match template:
   toLower(n.{{filter_property}}) CONTAINS toLower({{filter_value}})
8. Always typecast properties used in filters to match the value type:
   - int:   toInteger(n.prop)
   - float: toFloat(n.prop)
   - date:  date(n.prop) and date("YYYY-MM-DD")
   - bool:  n.prop = true/false
9. If the question can be answered using a single node, do NOT include relationships.
10. If multiple queries are needed, return them separated by a newline. Do NOT use UNION.
11. LIMIT must always be {top_k}.
12. If you cannot answer using the schema + templates, return exactly: None

### Graph Schema:
{graph_schema}

### Entity ID Rules:
- If request contains a PO number like PO-INF-001, filter using PurchaseOrder.purchase_order_number
- If request contains a PR number like PR-INF-001, filter using PurchaseRequest.purchase_requisition_id
- If request contains an Invoice ID like IR-INF-001, filter using Invoice.invoice_reconciliation_id
- If request contains a Contract number like C-INF-001, filter using Contract.contract_number

### Graph traversal rule:
- You may traverse at most {max_hops} hops
- Never exceed this limit
- Prefer explicit relationship paths
- Avoid variable-length traversal unless necessary
- If using variable hops, use:
  [*1..{max_hops}]


### User Question:
{natural_language_request}

### Cypher Templates (choose the best one):
{cypher_templates}

### Output:
Return ONLY the final Neo4j Cypher query.
"""


NEO4J_TEMPLATE_BASED_CYPHER_PROMPT = PromptTemplate(
    input_variables=[
        "graph_schema",
        "natural_language_request",
        "cypher_templates",
        "top_k",
        "max_hops",
    ],
    template=NEO4J_TEMPLATE_BASED_CYPHER_PROMPT_TEMPLATE,
)


CYPHER_QA_TEMPLATE = """
You are an assistant that generates clear, concise, and human-understandable answers based solely on the provided
Question and Information.

Instructions:
- Carefully read and fully understand the user's question.
- Use only the relevant information from the provided context to answer. Do not include unnecessary or unrelated details.
- Treat the provided context as the authoritative source. Do not question, supplement, or correct it using your own knowledge.
- The context is a list of dictionaries (output of a Cypher query). Use this data directly to answer the question.
- If the context contains a list of items, assume they are already filtered as per the question and present them as the answer.
- Never use your internal knowledge to modify, filter, or answer any part of the question.
- Do not expose any data from the context that is not required to answer the question.
- If the question has multiple parts but the context only answers some, answer only those parts and clearly state the answer for the rest.
- Write your answer as a direct response to the question. Do not mention that your answer is based on the provided information.
- You may use markdown formatting (bullet points, numbered lists, etc.) to improve clarity and readability.
- If the question is a greeting, you may greet the user.

Example:

Question: Give me all purchase orders with invoice amounts greater than 100000 dollars.
Context: [{{'purch_ord_id': '322901889', 'purch_ord_no': 'PO1311532'}}, {{'purch_ord_id': '322897475', 'purch_ord_no': 'PO1311530'}}]
Helpful Answer: The purchase orders with invoice amounts greater than 100000 dollars are: PO1311532, PO1311530.

Follow this example when generating answers.
If the provided information is empty or you cannot answer the question using the given context, simply say that you don't know the
answer.

Information:
{context}

Question: {natural_language_request}
Helpful Answer:
"""
CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "natural_language_request"], template=CYPHER_QA_TEMPLATE
)
