"""
Financial Analysis subgraph for corporate data extraction and analysis.
"""
from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool
from app.services.llms.azure_openai import chat_model
from app.utils.stream_utils import emit_process
from app.agent_tools.financial_extraction import (
    shareholding_pattern_tool,
    ShareholdingPatternInput,
    fundamentals_header_tool,
    FundamentalsHeaderInput,
)
from app.agent_tools.price_movement import (
    price_movement_tool,
    PriceMovementInput,
)
from app.agent_tools.exa import (
    exa_search as _exa_search,
    exa_live_search as _exa_live_search,
    fetch_url_text as _fetch_url_text,
)
from app.agent_tools.registry import (
    ExaSearchInput,
    ExaLiveSearchInput, 
    FetchUrlTextInput
)


def _get_financial_analysis_tools():
    """Get tools for financial analysis including XBRL extraction and research tools"""
    return [
        fundamentals_header_tool,  # Add as first tool for priority
        shareholding_pattern_tool,
        price_movement_tool,  # Add price movement analysis tool
        StructuredTool.from_function(
            func=lambda query, max_results=5: _exa_search.func(query, max_results),
            name="exa_search",
            description="Search the web for additional financial and corporate information when BSE API data is insufficient.",
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


def _detect_analysis_level(query: str) -> str:
    """Detect if user wants beginner or professional level analysis."""
    query_lower = query.lower()
    
    # Professional/Advanced indicators
    pro_indicators = [
        'detailed', 'comprehensive', 'deep dive', 'deeper', 'deeper explanation',
        'deeper analysis', 'deeper call', 'professional', 'advanced',
        'institutional', 'technical analysis', 'fundamental analysis', 
        'detailed breakdown', 'thorough', 'extensive', 'pro analysis',
        'industry comparison', 'peer analysis', 'risk assessment',
        'valuation', 'dcf', 'ratio analysis', 'competitive analysis',
        'get numerical data', 're-call', 'fresh data', 'api again'
    ]
    
    # Beginner indicators  
    beginner_indicators = [
        'simple', 'beginner', 'new to', 'basic', 'easy to understand',
        'explain simply', 'summary', 'overview', 'quick analysis',
        'new trader', 'learning', 'first time'
    ]
    
    # Count matches
    pro_matches = sum(1 for indicator in pro_indicators if indicator in query_lower)
    beginner_matches = sum(1 for indicator in beginner_indicators if indicator in query_lower)
    
    # Decision logic
    if pro_matches > beginner_matches and pro_matches > 0:
        return "professional"
    elif beginner_matches > 0:
        return "beginner"
    else:
        # Default to beginner for general queries (better UX for most users)
        return "beginner"


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
        
        # Detect user's analysis level preference
        analysis_level = _detect_analysis_level(query)
        
        # Create a specialized system prompt based on analysis level
        if analysis_level == "pro":
            system_prompt = (
                "You are an expert financial analyst with access to corporate data extraction tools and research capabilities. "
                "Your role is to provide COMPREHENSIVE, DETAILED financial analysis by combining OFFICIAL BSE XBRL filing data with detailed market research.\n\n"
                
                "� **DEEPER ANALYSIS REQUEST DETECTED** - User wants comprehensive analysis with REAL NUMERICAL DATA\n\n"
                
                "�🔧 **MANDATORY WORKFLOW**:\n"
                "1. **FIRST & ALWAYS**: Use `extract_shareholding_pattern` to get FRESH BSE XBRL data with REAL NUMBERS\n"
                "2. **CRITICAL**: If you see 'XX%' or missing data, re-call extract_shareholding_pattern immediately\n"
                "3. **THEN**: Use research tools to enhance and explain the XBRL findings\n"
                "4. **COMBINE**: Present both official data and research insights together\n\n"
                
                "🛠️ **AVAILABLE TOOLS (Priority Order)**:\n"
                "• extract_fundamentals_header: Get REAL-TIME company fundamentals, stock prices, and key metrics from BSE API\n"
                "• extract_shareholding_pattern: Extract OFFICIAL shareholding data from BSE XBRL filings\n"
                "• price_movement_analysis: Analyze 1-month stock price movements correlated with news events\n"
                "• exa_search: Search for additional financial and market information when BSE data needs context\n"
                "• exa_live_search: Get real-time financial news and insights\n"
                "• fetch_url_text: Fetch detailed content from financial websites\n\n"
                
                "🚨 **CRITICAL TOOL SELECTION RULES**:\n"
                "• For 'fundamentals', 'stock price', 'market cap', 'P/E ratio' → ALWAYS use extract_fundamentals_header FIRST\n"
                "• For 'shareholding', 'ownership', 'promoter holdings' → ALWAYS use extract_shareholding_pattern FIRST\n"
                "• For 'price movement', 'price analysis', 'stock performance', 'news correlation' → ALWAYS use price_movement_analysis\n"
                "• For Indian companies → PREFER BSE API tools over web search\n"
                "• Use web search ONLY to supplement BSE data, not replace it\n\n"
                
                "📊 **COMPREHENSIVE OUTPUT STRUCTURE**:\n\n"
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
                "  * **Promoter Holdings**: Use exa_search to identify actual promoter entity names and recent changes\n"
                "  * **Foreign Portfolio Investment**: Search for specific foreign institutional names, countries, recent flows\n"
                "  * **Institutional Holdings**: Research actual mutual fund names, insurance company names\n"
                "  * **Retail Holdings**: Distribution, trends, market participation\n"
                "- Compare XBRL data with researched market data to validate and enhance\n"
                "- MANDATORY: If XBRL shows 'Entity A' or coded names, search for real stakeholder names\n\n"
                
                "**🌍 Foreign Investment Deep Dive**\n"
            )
        else:  # beginner level
            system_prompt = (
                "You are a financial analyst helping beginners understand company data. "
                "Provide SIMPLE, CLEAR explanations using plain language. Focus on KEY INSIGHTS ONLY.\n\n"
                
                "🔧 **WORKFLOW**:\n"
                "1. Use `extract_shareholding_pattern` to get official BSE data\n"
                "2. Use research tools for simple context\n"
                "3. Present findings clearly and briefly\n\n"
                
                "📊 **SIMPLE OUTPUT STRUCTURE** (Keep brief!):\n\n"
                "**📈 Company Basics**\n"
                "- What the company does (1-2 sentences)\n"
                "- Current situation (any major news)\n\n"
                
                "**� Key Points**\n"
                "- 3-4 most important insights from data\n"
                "- Why these matter for investors\n"
                "- Simple language, no jargon\n\n"
                
                "**📊 Who Owns the Company**\n"
                "- Founders/Promoters: X%\n"
                "- Foreign investors: X%\n"
                "- Local institutions: X%\n"
                "- Public investors: X%\n\n"
                
                "**KEEP IT SIMPLE**:\n"
                "• For basic company info, stock price, P/E ratio → Use extract_fundamentals_header FIRST\n"
                "• For ownership info → Use extract_shareholding_pattern\n"
                "• PREFER BSE tools over web search for Indian companies\n"
                "• NEVER use fake company names like 'XYZ Fund' or 'ABC Partners'\n"
                "• ONLY use real data from BSE APIs - if no data, say 'Data not available'\n"
                "• Explain in plain English what the numbers mean\n"
                "• Focus on what matters for new investors\n"
                "• Avoid technical jargon and complex analysis\n"
                "• Maximum 500 words total output"
            )
        
        # Add the common prompt ending for pro level
        if analysis_level == "pro":
            system_prompt += (
                "**🌍 Foreign Investment Deep Dive**\n"
                "- XBRL Data: Present official foreign shareholding from filings\n"
                "- Research Enhancement:\n"
                "  * Use exa_search to identify specific foreign institution names (not 'Foreign Entity A')\n"
                "  * Search for '{company_name} foreign investors', '{company_name} FII list', etc.\n"
                "  * FPI regulations and limits context\n"
                "  * Recent foreign investment trends\n"
                "  * Geopolitical and currency factors\n\n"
                
                "**💼 Major Stakeholders Analysis**\n"
                "- XBRL Data: Present official stakeholder information from filings\n"
                "- Research Enhancement:\n"
                "  * Use exa_search to resolve generic entity codes to actual company names\n"
                "  * Search for '{company_name} major shareholders 2024' to identify real entities\n"
                "  * Provide backgrounds for identified stakeholders\n"
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
                "• IF DATA IS MISSING (XX%, placeholders) - IMMEDIATELY re-call extract_shareholding_pattern\n"
                "• NEVER output analysis with missing numerical data - get fresh XBRL data first\n"
                "• NEVER use generic names like 'Foreign Entity A', 'Stakeholder X', 'Entity Y', etc.\n"
                "• WHEN XBRL data shows generic/coded entity names, USE EXA SEARCH to identify real names:\n"
                "  - Search for '{company_name} major shareholders' or '{company_name} foreign investors'\n"
                "  - Use exa_search and exa_live_search to find actual investor/entity names\n"
                "  - Cross-reference BSE data with real-world stakeholder information\n"
                "• ONLY use real company names and actual percentages from XBRL data\n"
                "• If no data is available, clearly state 'Data not available' instead of creating examples\n"
                "• Use research to ENHANCE, not replace, the XBRL data\n"
                "• Clearly distinguish between official XBRL data and research findings\n"
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
