from google import genai
from google.genai import types
import cohere
import instructor
from langsmith import traceable
import os
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http.models import   Document, Prefetch

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

### Items metadata retrieval tool

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={
        "ls_provider": "google",
        "ls_model_name": "gemini-embedding-001",
    }
)
def get_embeddings(text, task_type="SEMANTIC_SIMILARITY", model="gemini-embedding-001"):
    result = gemini_client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return result.embeddings[0].values

@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_data(query, qdrant_client, k=5, hybrid_search=True):
    query_embedding = get_embeddings(query)
    if hybrid_search:
        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using="gemini-embedding-001",
                    limit=20
                ),
                Prefetch(
                    query=Document(
                        text=query,
                        model="qdrant/bm25"
                    ),
                    using="bm25",
                    limit=20
                )
            ],
            query=models.RrfQuery(rrf=models.Rrf(weights=[3,1])),
            limit=k
        )
    else:
        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            query=query_embedding,
            using="gemini-embedding-001",
            limit=k
        )
    retrieved_context_ids=[]
    retrieved_context=[]
    similarity_scores=[]
    retrieved_context_ratings=[]

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["preprocessed_description"])
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }

@traceable(
    name="rerank_data",
    run_type="tool"
)
def rerank_data(query, context, top_k=5):
    cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    response = cohere_client.rerank(
        model="rerank-v4.0-pro",
        query=query,
        documents=context["retrieved_context"],
        top_n=top_k
    )
    order = [result.index for result in response.results]
    
    return {
        "retrieved_context_ids": [context["retrieved_context_ids"][i] for i in order],
        "retrieved_context": [context["retrieved_context"][i] for i in order],
        "similarity_scores": [context["similarity_scores"][i] for i in order],
        "retrieved_context_ratings": [context["retrieved_context_ratings"][i] for i in order]
    }

### Format retrieved context function
@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context

@tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """ Get the top k context, each representing an inventory item for a given query 

    Args:
        query: The query to get the context for
        top_k: The number of context to return
        
    Returns:
        A string of the top k context with IDs and average ratings for each chunk.
    """

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    retrieved_context = retrieve_data(query, qdrant_client, k=20)

    retrieved_context = rerank_data(query, retrieved_context, top_k=top_k)

    formatted_context = process_context(retrieved_context)

    return formatted_context