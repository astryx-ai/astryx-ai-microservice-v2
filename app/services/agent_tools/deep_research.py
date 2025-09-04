from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from app.services.llms.azure_openai import chat_model
from app.utils.stream_utils import emit_process
from .exa import (
    exa_search as _exa_search,
    exa_live_search as _exa_live_search,
    fetch_url_text as _fetch_url_text,
)


# Minimal schemas for deep research tools
class ExaSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results")

class ExaLiveSearchInput(BaseModel):
    query: str = Field(..., description="Live search query")
    k: int = Field(8, ge=1, le=20, description="Number of documents to retrieve")
    max_chars: int = Field(1000, ge=100, le=8000, description="Maximum characters in each summary")

class FetchUrlTextInput(BaseModel):
    url: str = Field(..., description="Web page URL to fetch in raw text chunks")
    chunk_index: int = Field(1, ge=1, le=1000, description="1-based index of chunk to return")
    chunk_size: int = Field(4000, ge=500, le=20000, description="Approximate characters per chunk")
    max_total_chars: int = Field(120000, ge=5000, le=500000, description="Safety cap on total extracted characters")


def _get_research_tools():
    """Get the minimal set of tools needed for deep research."""
    return [
        StructuredTool.from_function(
            func=lambda query, max_results=5: _exa_search.func(query, max_results),
            name="exa_search",
            description="Search the web and return detailed results preserving all key facts and numbers.",
            args_schema=ExaSearchInput,
        ),
        StructuredTool.from_function(
            func=lambda query, k=8, max_chars=1000: _exa_live_search.func(query, k, max_chars),
            name="exa_live_search",
            description="Live-crawl search that returns detailed summaries preserving key facts and numbers.",
            args_schema=ExaLiveSearchInput,
        ),
        StructuredTool.from_function(
            func=lambda url, chunk_index=1, chunk_size=4000, max_total_chars=120000: _fetch_url_text.func(url, chunk_index, chunk_size, max_total_chars),
            name="fetch_url_text",
            description="Fetch a web page and return raw text in chunks without summarization.",
            args_schema=FetchUrlTextInput,
        ),
    ]


def run_deep_research(query: str, max_chunks_per_url: int = 3, chunk_size: int = 4000, k: int = 8, max_results: int = 5, context_messages = None) -> str:
    """Execute deep research using a specialized subgraph agent."""
    print(f"[DeepResearch] Starting deep research | query='{query}', max_chunks_per_url={max_chunks_per_url}, chunk_size={chunk_size}, k={k}, max_results={max_results}")
    
    try:
        # Build a lean subgraph with research-oriented prompt and only web tools (no recursion into itself)
        llm = chat_model(temperature=0.1)
        tools = _get_research_tools()
        agent = create_react_agent(llm, tools)
        
        # Build conversation context
        conversation_context = ""
        if context_messages:
            print(f"[DeepResearch] Including {len(context_messages)} context messages")
            context_parts = []
            for msg in context_messages[-6:]:  # Last 6 messages for context
                if hasattr(msg, 'type'):
                    role = "User" if msg.type == "human" else "Assistant"
                    content = getattr(msg, 'content', str(msg))
                    if content and not content.startswith("You are"):  # Skip system messages
                        context_parts.append(f"{role}: {content[:500]}")  # Truncate long messages
            if context_parts:
                conversation_context = "\n\nCONVERSATION CONTEXT:\n" + "\n".join(context_parts[-4:])  # Last 4 exchanges
        
        system = (
            "You are a financial research specialist. Conduct comprehensive research using all available tools: "
            "1) Use multiple search tools (exa_live_search, exa_search) to get broad coverage, "
            "2) Select the most relevant URLs from all search results, "
            "3) Use fetch_url_text to read selected pages in detail (chunk_index=1, increment as needed), "
            "4) Present findings in a well-structured format with tables for financial data, bullet points for insights, "
            "and proper headings. Include specific numbers, percentages, market caps, and growth rates in tables."
            f"{conversation_context}"
        )
        
        user = (
            f"Research topic: {query}\n"
            f"Constraints: Max {max_chunks_per_url} chunks per URL, {chunk_size} chars per chunk.\n"
            f"Present findings with clear structure:\n"
            f"- Executive summary\n"
            f"- Company/market overview tables\n"
            f"- Financial metrics in tabular format\n"
            f"- Key insights as bullet points\n"
            f"- Future outlook\n"
            f"- Proper citations"
        )
        
        try:
            emit_process({"message": "Deep Research started"})
        except Exception:
            pass

        result = agent.invoke({"messages": [
            {"type": "system", "content": system},
            {"type": "human", "content": user},
        ]})
        
        print(f"[DeepResearch] Agent invoke result type: {type(result)}")
        
        # Handle different response types from agent
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                content = getattr(last_message, "content", "") if hasattr(last_message, "content") else str(last_message)
            else:
                content = ""
        elif isinstance(result, str):
            content = result
        else:
            content = str(result)
            
        # Emit completion
        try:
            emit_process({"message": "Deep Research completed"})
        except Exception:
            pass

        return content if content else "No research results were generated."
        
    except Exception as e:
        return f"Deep research failed: {e}"
