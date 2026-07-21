from google import genai
import os
from google.genai import types
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from operator import add
from jinja2 import Template
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_openai_messages, AIMessage
import instructor
from langsmith import traceable
from langsmith import get_current_run_tree
from langgraph.types import Send
from pydantic import Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from api.core.config import config
from api.agents.utils.prompt_management import prompt_template_config
from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

### QnA Response Model
class RAGUsedContext(BaseModel):
    id: str = Field(description="   id of the item used to answer the question")
    description: str = Field(description="   description of the item used to answer the question")

class FinalResponse(BaseModel):


    """ Call this tool when the final answer is possible using available context"""

    answer:str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")


###Intent Router Response Model
class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str = Field(description="Answer to the question")


### Q&A Agent Node

@traceable(
    name="agent_node",
    run_type="llm",
    metadata={
        "ls_provider": "google",
        "ls_model_name": "gemini-3.1-flash-lite"
    }
)
def agent_node(state) -> dict:

    template = prompt_template_config("api/agents/prompts/qna_agent.yaml", "qna_agent") 
    prompt = template.render()

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", client=gemini_client)
    llm_with_tools = llm.bind_tools([get_formatted_item_context, get_formatted_reviews_context, FinalResponse], tool_choice="required")

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt),
            *state.messages,
        ]
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage_metadata["input_tokens"],
            "output_tokens": response.usage_metadata["output_tokens"],
            "total_tokens": response.usage_metadata["total_tokens"]
        }

    final_answer = False
    answer = ""
    references = []

    def sanitise_response(response):

        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalResponse":
                answer = tool_call.get("args").get("answer")
        
        return AIMessage(content=answer)

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalResponse":
                final_answer = True
                answer = tool_call.get("args").get("answer")
                references.extend(tool_call.get("args").get("references"))
                response = sanitise_response(response)


    return {
        "messages": [response],
        "iteration": state.iteration + 1,
        "final_answer": final_answer,
        "answer": answer,
        "references": references
    }

### Intent Router Node

@traceable(
    name="route_intent",
    run_type="llm",
    metadata={
        "ls_provider": "google",
        "ls_model_name": "gemini-3.1-flash-lite"
    }
)
def intent_router_node(state) -> dict:

    template = prompt_template_config("api/agents/prompts/intent_router_agent.yaml", "intent_router_agent") 
    prompt = template.render()

    messages = state.messages
    conversation = []
    conversation.append(convert_to_openai_messages(messages[-1]))
    client = instructor.from_genai(
    gemini_client,
    model="gemini-3.1-flash-lite"
    )

    response, raw_response = client.create_with_completion(
    messages=[
        {"role": "user", "content": prompt},
        *conversation
    ],
    response_model=IntentRouterResponse,
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage_metadata.prompt_token_count,
            "output_tokens": raw_response.usage_metadata.candidates_token_count,
            "total_tokens": raw_response.usage_metadata.total_token_count,
        }
        trace_id = str(current_run.trace_id)

    else:
        trace_id = ""


    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
        "trace_id": trace_id
    }