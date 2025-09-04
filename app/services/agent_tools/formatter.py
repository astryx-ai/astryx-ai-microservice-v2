from langchain_core.messages import HumanMessage, SystemMessage
from app.services.llms.azure_openai import chat_model


def _detect_duplicate_sections(content: str) -> bool:
    """Detect if content has obvious duplicate sections."""
    lines = content.split('\n')
    # Look for repeated headings or sections
    headings = [line.strip() for line in lines if line.strip() and (line.startswith('#') or line.isupper() or 'Executive Summary' in line or 'Company/Market Overview' in line)]
    
    # Check for repeated headings
    seen_headings = set()
    for heading in headings:
        normalized = heading.lower().strip('#').strip()
        if normalized in seen_headings and len(normalized) > 10:
            return True
        seen_headings.add(normalized)
    
    return False


def format_financial_content(content: str) -> str:
    """Use an intelligent agent to format and deduplicate financial content while preserving all important data."""
    print(f"[Formatter] Formatting financial content (length: {len(content)} chars)")
    
    if not content or len(content.strip()) < 100:
        return content
    
    # Check if content has obvious duplications
    has_duplicates = _detect_duplicate_sections(content)
    print(f"[Formatter] Duplicate sections detected: {has_duplicates}")
    
    try:
        # Build a focused formatting agent
        llm = chat_model(temperature=0.1)
        
        if has_duplicates:
            # Stronger deduplication prompt when duplicates are detected
            system_msg = (
                "You are an expert financial content editor. The content you receive contains DUPLICATE SECTIONS that need to be consolidated. "
                "Your task is to merge duplicate content intelligently while preserving ALL unique financial data. "
                "CRITICAL: If you see the same section repeated (like 'Executive Summary' appearing twice), merge them into ONE section. "
                "Keep all unique financial metrics, numbers, and insights. Remove redundant text but never omit data points."
            )
            
            user_msg = (
                f"The following content contains duplicate sections. Please consolidate and format it:\n\n{content}\n\n"
                f"REQUIREMENTS:\n"
                f"1. MERGE duplicate sections (don't repeat Executive Summary, Company Overview, etc.)\n"
                f"2. Preserve ALL unique financial data and metrics\n"
                f"3. Create ONE comprehensive document with clear structure\n"
                f"4. Use tables for numerical data\n"
                f"5. Keep all citations\n"
                f"6. Remove redundant text while keeping unique insights\n"
                f"7. Professional financial platform formatting"
            )
        else:
            # Standard formatting prompt
            system_msg = (
                "You are a financial content formatter. Your job is to take raw financial research content and format it professionally. "
                "Structure the content with clear headings, tables for numerical data, and proper citations. "
                "Ensure professional presentation suitable for a financial platform."
            )
            
            user_msg = (
                f"Format this financial research content for professional presentation:\n\n{content}\n\n"
                f"Requirements:\n"
                f"- Clear headings and structure\n"
                f"- Use tables for numerical comparisons\n"
                f"- Bold important metrics\n"
                f"- Maintain all citations\n"
                f"- Professional financial platform formatting"
            )
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ]
        
        response = llm.invoke(messages)
        formatted_content = response.content if hasattr(response, 'content') else str(response)
        
        print(f"[Formatter] Successfully formatted content (output length: {len(formatted_content)} chars)")
        return formatted_content
        
    except Exception as e:
        print(f"[Formatter] Formatting failed: {e}")
        return content  # Return original content if formatting fails
