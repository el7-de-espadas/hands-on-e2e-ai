from google import genai
from google.genai import types
import os
from qdrant_client.http.models import Prefetch, Document 
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from qdrant_client.http.models import FusionQuery

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embeddings(text, task_type="SEMANTIC_SIMILARITY", model="gemini-embedding-001"):
    result = gemini_client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return result.embeddings[0].values

def retrieve_prefiltered_reviews_data(query, parent_asins, qdrant_client, k=5):
    query_embedding = get_embeddings(query)
    results = qdrant_client.query_points(
            collection_name="Amazon-reviews-collection-01",
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using="gemini-embedding-001",
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="parent_asin",
                                match=MatchAny(
                                    any=parent_asins
                                )
                            )
                        ]
                    ),
                    limit=20
                )
            ],
            query=FusionQuery(fusion='rrf'),
            limit=k
    )
    retrieved_context_ids=[]
    retrieved_context=[]
    similarity_scores=[]

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["preprocessed_data"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }

def process_reviews_context(context):
    formatted_context = ""
    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- ID: {id}, description: {chunk}\n"
    return formatted_context