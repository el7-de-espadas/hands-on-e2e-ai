from api.agents.retrieval_generation import rag_pipeline

import os
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from langsmith import Client
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
evaluator_llm = llm_factory(
    "gemini-3.1-flash-lite",   # use a real model id
    provider="openai",    # OpenAI-compatible endpoint
    client=async_client,
    max_tokens=4000,
)

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")

vertex_embeddings = GoogleEmbeddings(
    model="gemini-embedding-001",
    client=gemini_client,         
)

# 2. Wrap them for the Ragas runtime
evaluator_embeddings = vertex_embeddings

def context_precision_id_based(run, example):

    retrieved_context_ids = {str(id) for id in run.outputs["retrieved_context_ids"]}
    reference_context_ids = {str(id) for id in example.outputs["reference_context_ids"]}

    score = len(retrieved_context_ids & reference_context_ids) / len(retrieved_context_ids) if retrieved_context_ids else 0.0
    
    return score

def context_recall_id_based(run, example):

    retrieved_context_ids = {str(id) for id in run.outputs["retrieved_context_ids"]}
    reference_context_ids = {str(id) for id in example.outputs["reference_context_ids"]}

    score = score = len(retrieved_context_ids & reference_context_ids) / len(reference_context_ids) if reference_context_ids else 0.0
    return score

def ragas_faithfulness(run, example):

    scorer = Faithfulness(llm=evaluator_llm)

    result = scorer.score(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    )
    return result.value

def ragas_relevancy(run, example):

    scorer = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

    result = scorer.score(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
    )

    return result.value

print("Evaluating plain retriever")

results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client, topk_k=10, hybrid_search=False, rerank=False),
    data="RAG-Eval-Dataset-extended",
    evaluators=[context_precision_id_based, context_recall_id_based, ragas_faithfulness, ragas_relevancy],
    experiment_prefix="plain"
)

print("Evaluating hybrid retriever")

results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client, topk_k=10, hybrid_search=True, rerank=False),
    data="RAG-Eval-Dataset-extended",
    evaluators=[context_precision_id_based, context_recall_id_based, ragas_faithfulness, ragas_relevancy],
    experiment_prefix="hybrid"
)

print("Evaluating reranked retriever")

results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client, topk_k=10, hybrid_search=True, rerank=True),
    data="RAG-Eval-Dataset-extended",
    evaluators=[context_precision_id_based, context_recall_id_based, ragas_faithfulness, ragas_relevancy],
    experiment_prefix="rerank"
)
