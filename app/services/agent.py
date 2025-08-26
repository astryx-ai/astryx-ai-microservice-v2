from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .azure_openai import chat_model
from .tools.exa import exa_search, exa_find_similar, fetch_url, exa_live_search


def build_agent():
    llm = chat_model(temperature=0.2)
    tools = [exa_search, exa_find_similar, exa_live_search, fetch_url]
    return create_react_agent(llm, tools)


def agent_answer(question: str) -> str:
    """Run the LangGraph ReAct agent with EXA tools on a single question and return the final answer."""
    graph = build_agent()
    system_msg = SystemMessage(content=(
        "You can search the web using the provided EXA tools when helpful. "
        "Prefer up-to-date sources. Be concise and cite sources inline."
    ))
    user_msg = HumanMessage(content=(
        f"Task: {question}\n"
        "If web context is needed: 1) prefer exa_live_search for fresh results, otherwise exa_search; "
        "2) optionally use fetch_url on top 1-3 URLs to extract short summaries, "
        "3) answer concisely with bullet points and cite links inline."
    ))
    state = {"messages": [system_msg, user_msg]}
    result = graph.invoke(state)
    messages = result.get("messages", [])
    return messages[-1].content if messages else ""


