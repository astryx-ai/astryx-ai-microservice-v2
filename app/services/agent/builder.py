from langgraph.prebuilt import create_react_agent
from app.services.llms.azure_openai import chat_model
from app.services.agent_tools.registry import load_tools


def build_agent(use_cases: list[str] | None = None, structured: bool = False):
    llm = chat_model(temperature=0.2)
    tools = load_tools(use_cases=use_cases, structured=structured)
    return create_react_agent(llm, tools)
