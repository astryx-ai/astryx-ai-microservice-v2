
from typing import Dict, Any
from .super_agent import run_super_agent
from .azure_openai import chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from .tools.exa import exa_search, exa_find_similar, fetch_url, exa_live_search


def build_agent():
    llm = chat_model(temperature=0.2)
    tools = [exa_search, exa_find_similar, exa_live_search, fetch_url]
    return create_react_agent(llm, tools)

def agent_answer(question: str, memory: Dict[str, Any] | None = None) -> str:
    """
    Run the structured financial SuperAgent on a single question
    and return the formatted markdown answer.
    """
    result = run_super_agent(question, memory=memory if memory is not None else {})
    return result.get("output", "")
