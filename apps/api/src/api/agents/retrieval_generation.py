from google import genai
from google.genai import types
import os
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
import instructor
from pydantic import BaseModel, Field
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Prefetch, Document
from qdrant_client.http import models
from api.agents.utils.prompt_management import prompt_template_config


gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class RAGUsedContext(BaseModel):
    id: str = Field(description="   id of the item used to answer the question")
    description: str = Field(description="   description of the item used to answer the question")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="   answer to the question")
    references: list[RAGUsedContext] = Field(description="   list of ritems used to answer the question")

### Other task_type = SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, RETRIEVAL_DOCUMENT, CODE_RETRIEVAL_QUERY, QUESTION_ANSWERING, FACT_VERIFICATION
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
def retrieve_data(query, qdrant_client, k=5):
    query_embedding = get_embeddings(query)
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
    template = prompt_template_config("api/agents/prompts/retrieval_generation.yaml", "retrieval-generation")  
    prompt = template.render(
        preprocessed_context=preprocessed_context,
        query=query
    )
    
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
    client = instructor.from_genai(
    gemini_client,
    model="gemini-3.1-flash-lite"
)
    response, raw_response = client.create_with_completion(
    messages=[
        {"role": "user", "content": prompt}
    ],
    response_model=RAGGenerationResponse,
)
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage_metadata.prompt_token_count,
            "output_tokens": raw_response.usage_metadata.candidates_token_count,
            "total_tokens": raw_response.usage_metadata.total_token_count,
        }
    return response

@traceable(
    name="rag_pipeline"
)
def rag_pipeline(query, qdrant_client, topk_k=5):
    retrieved_context = retrieve_data(query, qdrant_client, topk_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(query, preprocessed_context)
    answer = generate_answer(prompt)

    final_answer = {
        "answer": answer.answer,
        "references": answer.references,
        "question": query,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"]
    }
    return final_answer

def rag_pipeline_wrapper(query, topk_k=5):

    qdrant_client = QdrantClient(url="http://qdrant:6333")
    result = rag_pipeline(query, qdrant_client, topk_k)
    used_context = []
    for item in result.get("references",[]):
        payload = qdrant_client.scroll(
            collection_name="Amazon-items-collection-01-hybrid-search",
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                        )
                    ]
                )
            )[0][0].payload
        image_url = payload.get("image_url", "")
        price = payload.get("price")
        if image_url:
            used_context.append(
                {
                    "image_url": image_url,
                    "price": price,
                    "description": item.description
                }
            )

    return {
        "answer": result["answer"],
        "used_context": used_context
    }
