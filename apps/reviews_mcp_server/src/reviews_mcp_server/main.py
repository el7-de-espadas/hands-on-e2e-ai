from fastmcp import FastMCP
from qdrant_client import QdrantClient
from reviews_mcp_server.utils import retrieve_prefiltered_reviews_data, process_reviews_context

mcp = FastMCP("reviews_mcp_server")

@mcp.tool
def get_formatted_reviews_context(query: str,parent_asins: list[str], top_k: int = 5) -> str:
    """ 
    Get the top k reviews matching a query for a list of prefiltered items

    Args:
        query: The query to get the context for
        parent_asins: The list of parent ASINs to filter the context for
        top_k: The number of context to return

    Returns:
        A string of the top k context chunks with IDs prepending for each chun, each representing a review for a given inventory item.
    """
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    retrieved_context = retrieve_prefiltered_reviews_data(query, parent_asins, qdrant_client, top_k)

    formatted_context = process_reviews_context(retrieved_context)

    return formatted_context

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)