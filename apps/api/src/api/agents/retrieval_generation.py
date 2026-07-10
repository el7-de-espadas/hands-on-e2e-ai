from google import genai
from google.genai import types
import os
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

### Other task_type = SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, RETRIEVAL_DOCUMENT, CODE_RETRIEVAL_QUERY, QUESTION_ANSWERING, FACT_VERIFICATION
@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={
        "ls_provider": "google",
        "ls_model_name": "gemini-embedding-001"
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
def retrieve_data(query, qdrant_client, k=5):
    query_embedding = get_embeddings(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01",
        query=query_embedding,
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

@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(query, preprocessed_context):
    prompt = f"""
    You are a helpful assistant that can answer questions about the products in stock.
    You are given a question and a list of context.
    You need to answer the question based on the product descriptions.

    Instructions:
    - Answer the question based on the provided context only.
    - Never use the word "context" in your answer, refer it as the available products.
    - Do not use markdown formatting.

    Context:
    {preprocessed_context}

    Question: 
    {query}
    """
    return prompt

@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={
        "ls_provider": "google",
        "ls_model_name": "gemini-2.5-flash"
    }
)
def generate_answer(prompt):
    response = gemini_client.models.generate_content(
    contents=[prompt],
    model="gemini-2.5-flash",
)
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
        }
    return response.text

@traceable(
    name="rag_pipeline"
)
def rag_pipeline(query, qdrant_client, topk_k=5):
    retrieved_context = retrieve_data(query, qdrant_client, topk_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(query, preprocessed_context)
    answer = generate_answer(prompt)

    final_answer = {
        "answer": answer,
        "question": query,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"]
    }
    return final_answer


