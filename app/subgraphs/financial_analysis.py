"""
Financial Analysis subgraph for corporate data extraction and analysis.
"""
from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from app.services.llms.azure_openai import chat_model
from app.utils.stream_utils import emit_process
from app.agent_tools.financial_extraction import (
    shareholding_pattern_tool,
    ShareholdingPatternInput,
)
from app.agent_tools.exa import (
    exa_search as _exa_search,
    exa_live_search as _exa_live_search,
    fetch_url_text as _fetch_url_text,
)


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


def _get_financial_analysis_tools():
    """Get tools for financial analysis including XBRL extraction and research tools"""
    return [
        shareholding_pattern_tool,
        StructuredTool.from_function(
            func=lambda query, max_results=5: _exa_search.func(query, max_results),
            name="exa_search",
            description="Search the web for additional financial and corporate information.",
            args_schema=ExaSearchInput,
        ),
        StructuredTool.from_function(
            func=lambda query, k=8, max_chars=1000: _exa_live_search.func(query, k, max_chars),
            name="exa_live_search", 
            description="Live-crawl search for real-time financial information and explanations.",
            args_schema=ExaLiveSearchInput,
        ),
        StructuredTool.from_function(
            func=lambda url, chunk_index=1, chunk_size=4000, max_total_chars=120000: _fetch_url_text.func(url, chunk_index, chunk_size, max_total_chars),
            name="fetch_url_text",
            description="Fetch detailed content from financial websites and reports.",
            args_schema=FetchUrlTextInput,
        ),
    ]


def run_financial_analysis(query: str, context_messages=None) -> str:
    """
    Run financial analysis using specialized tools for XBRL extraction and corporate data analysis.
    
    This subgraph is designed to handle:
    - Shareholding pattern analysis
    - Financial data extraction from XBRL files
    - BSE corporate filings analysis
    
    Note: Corporate governance analysis is temporarily disabled and falls back to EXA search.
    """
    try:
        emit_process({"message": "Analyzing financial data and extracting corporate information"})
        
        # Create a specialized system prompt for financial analysis
        system_prompt = (
            "You are an expert financial analyst with access to corporate data extraction tools and research capabilities. "
            "Your role is to provide comprehensive financial analysis by combining OFFICIAL BSE XBRL filing data with detailed market research.\n\n"
            
            "🔧 **MANDATORY WORKFLOW**:\n"
            "1. **FIRST**: Always use `extract_shareholding_pattern` to get OFFICIAL BSE XBRL data\n"
            "2. **THEN**: Use research tools to enhance and explain the XBRL findings\n"
            "3. **COMBINE**: Present both official data and research insights together\n\n"
            
            "� **AVAILABLE TOOLS**:\n"
            "• extract_shareholding_pattern: Extract OFFICIAL shareholding data from BSE XBRL filings\n"
            "• exa_search: Search for additional financial and market information\n"
            "• exa_live_search: Get real-time financial news and insights\n"
            "• fetch_url_text: Fetch detailed content from financial websites\n\n"
            
            "📊 **ENHANCED OUTPUT STRUCTURE**:\n\n"
            "**📈 Company Overview & Context**\n"
            "- Research: Brief company background, business model, recent developments\n"
            "- XBRL Context: Period covered, filing details, data completeness\n\n"
            
            "**💡 Key Insights**\n"
            "- Present XBRL-extracted insights first\n"
            "- Add research context explaining WHY these patterns matter\n"
            "- Compare with market trends and peer companies\n\n"
            
            "**📊 Official BSE Filing Data (XBRL)**\n"
            "- Present the official XBRL shareholding composition table from BSE filings\n"
            "- Include exact percentages and entity names from regulatory filings\n"
            "- Note any limitations or context references in the official data\n\n"
            
            "**📊 Enhanced Shareholding Analysis (Research)**\n"
            "- Research comprehensive shareholding breakdown with:\n"
            "  * **Promoter Holdings**: Detailed breakdown by entity names, recent changes\n"
            "  * **Foreign Portfolio Investment**: Specific institutions, countries, recent flows\n"
            "  * **Institutional Holdings**: Mutual funds, insurance companies by name\n"
            "  * **Retail Holdings**: Distribution, trends, market participation\n"
            "- Compare XBRL data with researched market data to validate and enhance\n\n"
            
            "**🌍 Foreign Investment Deep Dive**\n"
            "- XBRL Data: Present official foreign shareholding from filings\n"
            "- Research Enhancement:\n"
            "  * Identify specific foreign institutions by name\n"
            "  * FPI regulations and limits context\n"
            "  * Recent foreign investment trends\n"
            "  * Geopolitical and currency factors\n\n"
            
            "**💼 Major Stakeholders Analysis**\n"
            "- XBRL Data: Present official stakeholder information from filings\n"
            "- Research Enhancement:\n"
            "  * Resolve entity names and provide backgrounds\n"
            "  * Investment philosophies and strategies\n"
            "  * Recent activity and portfolio changes\n\n"
            
            "**📈 Trend Analysis & Comparisons**\n"
            "- XBRL Trends: Year-over-year changes from official filings\n"
            "- Research Trends: Market data, peer comparisons, industry benchmarks\n"
            "- Validation: Cross-check XBRL data with market research\n\n"
            
            "**⚖️ Risk & Opportunity Assessment**\n"
            "- Based on XBRL patterns + market research\n"
            "- Industry-specific factors and regulatory environment\n"
            "- Future outlook based on current trends\n\n"
            
            "**🎯 Investment Implications**\n"
            "- What the combined data means for investors\n"
            "- Liquidity, volatility, and governance implications\n"
            "- Actionable insights and recommendations\n\n"
            
            "**CRITICAL REQUIREMENTS**:\n"
            "• ALWAYS start with extract_shareholding_pattern for official BSE data\n"
            "• Use research to ENHANCE, not replace, the XBRL data\n"
            "• Clearly distinguish between official XBRL data and research findings\n"
            "• Provide specific company names, percentages, and entities\n"
            "• Cross-validate research findings with XBRL data\n"
            "• Include data sources and time periods for both XBRL and research\n\n"
            
            "The goal is to provide the most comprehensive analysis possible by combining authoritative regulatory data with rich market intelligence."
        )
        
        # Get the LLM and tools
        llm = chat_model(temperature=0.1)  # Lower temperature for more focused analysis
        tools = _get_financial_analysis_tools()
        
        # Create the agent
        agent = create_react_agent(llm, tools)
        
        # Build the messages with enhanced system prompt
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add context if available
        if context_messages:
            # Add relevant context messages (limit to recent ones)
            recent_context = context_messages[-6:] if len(context_messages) > 6 else context_messages
            for msg in recent_context:
                if hasattr(msg, 'type') and msg.type in ['human', 'ai']:
                    messages.append(msg)
        
        # Add the current query
        messages.append(HumanMessage(content=query))
        
        # Run the agent
        result = agent.invoke({"messages": messages})
        
        # Extract the response content
        if isinstance(result, dict) and "messages" in result:
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                response = final_message.content
            else:
                response = str(final_message)
        else:
            response = str(result)
        
        # Emit completion message
        emit_process({"message": "Financial analysis completed"})
        
        return response
        
    except Exception as e:
        error_msg = f"Financial analysis failed: {str(e)}"
        emit_process({"message": error_msg})
        return f"❌ {error_msg}\n\nPlease try again with a specific company name and analysis type (shareholding or governance)."
