from fastmcp import FastMCP
from qdrant_client import QdrantClient
from items_mcp_server.utils import retrieve_items_data, rerank_data, process_context

mcp = FastMCP("items_mcp_server")

@mcp.tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """ 
    Search available products and return the top k inventory items

    Expand the customer's question into 1-5 concise search statements and issue them in parallel in a single turn.
    Each statement covers one distinct product or attribute; no two way express the same intent. Use natural product-description language.
    If no brand or model is specified, search broadly rather than refusing.

    "Earphones for me and a waterproof speaker" -> "Peronal earphones" | "Waterproof speaker"
    "A warm winter jacket for hiking" -> "Insulated Winter jacket" | "Hiking outerwear for cold weather"

    Args:
        query: The query to get the context for
        top_k: The number of context to return
        
    Returns:
        A string of the top k context with IDs and average ratings for each chunk.
    """

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    retrieved_context = retrieve_items_data(query, qdrant_client, k=20)

    retrieved_context = rerank_data(query, retrieved_context, top_k=top_k)

    formatted_context = process_context(retrieved_context)

    return formatted_context

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)