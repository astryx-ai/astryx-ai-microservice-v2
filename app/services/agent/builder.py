from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
# SystemMessage and HumanMessage imported in runner.py where needed
from app.services.llms.azure_openai import chat_model
from app.services.agent_tools.registry import load_tools
from app.services.agent_tools.helper_tools import requires_deep_research
from app.services.agent_tools.deep_research import run_deep_research


def _route_decision(state):
    """Decide between standard search and deep research."""
    messages = state.get("messages", [])
    if not messages:
        print("[Router] No messages, routing to standard")
        return "standard"
    
    # Count previous messages to determine if there's conversation context
    message_count = len([msg for msg in messages if hasattr(msg, 'type') and msg.type in ["human", "ai"]])
    has_context = message_count > 2  # More than system + current user message
    
    # Get the user's question from the last human message
    user_message = None
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == "human":
            content = msg.content
            # Clean up the task format if present
            if content.startswith("Task: "):
                user_message = content[6:]  # Remove "Task: " prefix
            else:
                user_message = content
            break
    
    if user_message and requires_deep_research(user_message, has_context):
        print("[Router] Routing to deep_research")
        return "deep_research"
    else:
        print("[Router] Routing to standard")
        return "standard"


def _standard_agent_node(state):
    """Standard agent node for simple queries."""
    print("[StandardAgent] Processing with standard search tools")
    messages = state.get("messages", [])
    
    # Check if this is a context-aware query (references previous conversation)
    user_question = ""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == "human":
            content = msg.content
            if content.startswith("Task: "):
                user_question = content[6:]
            else:
                user_question = content
            break
    
    # Count conversation context
    context_messages = [msg for msg in messages if hasattr(msg, 'type') and msg.type in ["human", "ai"]]
    has_context = len(context_messages) > 2
    
    # If it's a context-aware query, add enhanced system prompt
    if has_context and any(ref in user_question.lower() for ref in ["you mentioned", "companies", "those", "these", "them", "comparison"]):
        print("[StandardAgent] Using context-aware mode")
        # Create enhanced system message for context-aware queries
        enhanced_system = (
            "You are a financial AI assistant with access to the conversation history. "
            "The user is asking a follow-up question that references previous discussion. "
            "Use web search tools if you need additional current information, but primarily "
            "focus on the conversation context to answer questions about previously mentioned companies or topics. "
            "For comparisons, use the information already discussed in the conversation."
        )
        
        # Update the system message in the state
        updated_messages = []
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == "system":
                # Replace system message with context-aware version
                from langchain_core.messages import SystemMessage
                updated_messages.append(SystemMessage(content=enhanced_system))
            else:
                updated_messages.append(msg)
        
        updated_state = {"messages": updated_messages}
    else:
        updated_state = state
    
    llm = chat_model(temperature=0.2)
    tools = load_tools(use_cases=["web_search"], structured=True)
    agent = create_react_agent(llm, tools)
    return agent.invoke(updated_state)


def _deep_research_node(state):
    """Deep research node for comprehensive analysis."""
    print("[DeepResearch] Processing with deep research subgraph")
    messages = state.get("messages", [])
    
    # Extract the user's question from the last human message
    user_question = ""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == "human":
            content = msg.content
            # Clean up the task format if present
            if content.startswith("Task: "):
                user_question = content[6:]  # Remove "Task: " prefix
            else:
                user_question = content
            break
    
    print(f"[DeepResearch] Extracted user question: '{user_question}'")
    
    try:
        # Run deep research with conversation context
        context_messages = [msg for msg in messages if hasattr(msg, 'type') and msg.type in ["human", "ai"]]
        research_result = run_deep_research(user_question, context_messages=context_messages)
        print(f"[DeepResearch] Research completed, result length: {len(research_result) if research_result else 0}")
        
        # Ensure we have a valid result
        if not research_result or not isinstance(research_result, str):
            research_result = "I apologize, but I encountered an issue while conducting the deep research. Please try again."
        
        # Create an AI message with the research result
        from langchain_core.messages import AIMessage
        ai_message = AIMessage(content=research_result)
        
        # Return state with the new message
        return {"messages": messages + [ai_message]}
        
    except Exception as e:
        print(f"[DeepResearch] Error during research: {e}")
        from langchain_core.messages import AIMessage
        ai_message = AIMessage(content=f"I apologize, but I encountered an error while conducting the deep research: {str(e)}")
        return {"messages": messages + [ai_message]}


def build_routed_agent():
    """Build a LangGraph with routing between standard and deep research."""
    # Create the graph
    graph = StateGraph(dict)
    
    # Add nodes - no router node needed
    graph.add_node("standard", _standard_agent_node)
    graph.add_node("deep_research", _deep_research_node)
    
    # Set conditional entry point based on the routing decision
    graph.add_conditional_edges(
        "__start__",  # Use the built-in start node
        _route_decision,
        {
            "standard": "standard",
            "deep_research": "deep_research"
        }
    )
    
    # Add edges to END
    graph.add_edge("standard", END)
    graph.add_edge("deep_research", END)
    
    return graph.compile()


def build_agent(use_cases: list[str] | None = None, structured: bool = False):
    """Legacy function for backward compatibility - now returns the routed agent."""
    return build_routed_agent()