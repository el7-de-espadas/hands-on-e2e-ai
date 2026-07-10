from api.agents.retrieval_generation import rag_pipeline

import os
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from langsmith import Client
from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")
vertex_chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vertex_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",           
    google_api_key=os.getenv("GOOGLE_API_KEY")                        
)

# 2. Wrap them for the Ragas runtime
evaluator_llm = LangchainLLMWrapper(vertex_chat)
evaluator_embeddings = LangchainEmbeddingsWrapper(vertex_embeddings)

def ragas_context_precision_id_based(run, example):

    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"]
    )
    scorer = IDBasedContextPrecision()
    return scorer.single_turn_score(sample)

def ragas_context_recall_id_based(run, example):

    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"]
    )
    scorer = IDBasedContextRecall()
    return scorer.single_turn_score(sample)

def ragas_faithfulness(run, example):

    sample = SingleTurnSample(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    )
    scorer = Faithfulness(llm=evaluator_llm)
    return scorer.single_turn_score(sample)

def ragas_relevancy(run, example):

    sample = SingleTurnSample(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    )
    scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    return scorer.single_turn_score(sample)

results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="RAG-Eval-Dataset",
    evaluators=[ragas_context_precision_id_based, ragas_context_recall_id_based, ragas_faithfulness, ragas_relevancy],
    experiment_prefix="retriever"
)



