from langchain_core.messages import HumanMessage, SystemMessage
from app.services.llms.azure_openai import chat_model
import re


def _detect_duplicate_sections(content: str) -> bool:
    """Dynamic duplicate detection for any content type."""
    content_lower = content.lower()
    
    # 1. Auto-detect markdown headings and check for duplicates
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    headings = heading_pattern.findall(content_lower)
    
    # Count heading duplicates
    heading_texts = [heading[1].strip() for heading in headings]
    heading_counts = {}
    for heading in heading_texts:
        heading_counts[heading] = heading_counts.get(heading, 0) + 1
    
    for heading, count in heading_counts.items():
        if count > 1:
            print(f"[Formatter] DUPLICATE HEADING: '{heading}' appears {count} times")
            return True
    
    # 2. Check for content similarity (paragraphs repeated)
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
    
    for i, para1 in enumerate(paragraphs):
        for j, para2 in enumerate(paragraphs[i+1:], i+1):
            words1 = set(para1.lower().split())
            words2 = set(para2.lower().split())
            
            if len(words1) > 5 and len(words2) > 5:
                intersection = len(words1.intersection(words2))
                overlap_sim = intersection / max(len(words1), len(words2))
                
                if overlap_sim > 0.7:  # 70% word overlap = duplicate
                    print(f"[Formatter] CONTENT SIMILARITY: {overlap_sim:.1%} overlap")
                    return True
    
    return False


def format_financial_content(content: str) -> str:
    print(f"[Formatter] Formatting financial content (length: {len(content)} chars)")
    if not content or len(content.strip()) < 100:
        return content
    has_duplicates = _detect_duplicate_sections(content)
    print(f"[Formatter] Duplicate sections detected: {has_duplicates}")
    try:
        llm = chat_model(temperature=0.1)
        if has_duplicates:
            system_msg = (
                "You are an expert financial content editor. The content you receive contains DUPLICATE SECTIONS that need to be consolidated. "
                "Your task is to merge duplicate content intelligently while preserving ALL unique financial data. "
                "CRITICAL: If you see the same section repeated (like 'Executive Summary' appearing twice), merge them into ONE section. "
                "Keep all unique financial metrics, numbers, and insights. Remove redundant text but never omit data points. "
                "Begin with a 1–2 sentence 'Chart insight' lead if present, then ensure correct markdown spacing (blank line before/after '##' headings). "
                "Do not include raw JSON or fenced code blocks in the final output."
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
                f"7. Professional financial platform formatting\n"
                f"8. Start with a succinct 'Chart insight' lead if provided\n"
                f"9. Ensure there is a blank line before and after each '##' heading\n"
                f"10. Do NOT include JSON or fenced code blocks"
            )
        else:
            system_msg = (
                "You are a financial content formatter and data visualization expert. Your job is to take raw financial research content and format it professionally. "
                "Structure the content with clear headings, intelligent table layouts for numerical data, and proper citations. "
                "CRITICAL: For financial metrics, analyze data structure and choose optimal table format: (1) Single comprehensive table for simple data, (2) Multiple categorized tables for complex grouped data, (3) Compact side-by-side tables for comparisons. "
                "Ensure professional presentation suitable for a financial platform. "
                "Begin with a 1–2 sentence 'Chart insight' lead if present, then ensure correct markdown spacing (blank line before/after '##' headings). "
                "Do not include raw JSON or fenced code blocks in the final output."
            )
            user_msg = (
                f"Format this financial research content for professional presentation:\n\n{content}\n\n"
                f"Requirements:\n"
                f"- Clear headings and structure\n"
                f"- Smart table design: analyze data relationships and choose best format (single comprehensive vs categorized sections vs compact layouts)\n"
                f"- Use tables for ALL numerical comparisons and metrics\n"
                f"- Bold important metrics and add helpful calculations (% changes, ratios) where relevant\n"
                f"- Maintain all citations\n"
                f"- Professional financial platform formatting\n"
                f"- Start with a succinct 'Chart insight' lead if present\n"
                f"- Ensure there is a blank line before and after each '##' heading\n"
                f"- Do NOT include JSON or fenced code blocks"
            )
        messages = [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]
        response = llm.invoke(messages)
        formatted_content = response.content if hasattr(response, 'content') else str(response)
        formatted_content = _insert_newlines_before_inline_headings(formatted_content)
        formatted_content = _normalize_markdown_spacing(formatted_content)
        print(f"[Formatter] Successfully formatted content (output length: {len(formatted_content)} chars)")
        return formatted_content
    except Exception as e:
        print(f"[Formatter] Formatting failed: {e}")
        try:
            return _normalize_markdown_spacing(_insert_newlines_before_inline_headings(content))
        except Exception:
            return content


def _insert_newlines_before_inline_headings(content: str) -> str:
    """Ensure headings like '# ' or '## ' are on their own lines by inserting newlines before inline headings."""
    # Insert two newlines before heading markers that are not already at line start
    content = re.sub(r"([^\n])\s*(#{1,6}\s+)", r"\1\n\n\2", content)
    return content


def _normalize_markdown_spacing(content: str) -> str:
    """Normalize markdown spacing: blank line before/after headings and collapse excessive blank lines."""
    lines = content.split('\n')
    normalized_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_heading = line.lstrip().startswith('#') and line.lstrip().split(' ')[0].count('#') <= 6
        if is_heading:
            # Ensure blank line before heading
            if normalized_lines and normalized_lines[-1].strip() != '':
                normalized_lines.append('')
            normalized_lines.append(line.rstrip())
            # Ensure blank line after heading (if next is not blank or end)
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            if next_line is not None and next_line.strip() != '':
                normalized_lines.append('')
        else:
            normalized_lines.append(line.rstrip())
        i += 1
    # Collapse 3+ blank lines to max 2
    out = []
    blank_streak = 0
    for l in normalized_lines:
        if l.strip() == '':
            blank_streak += 1
            if blank_streak <= 2:
                out.append('')
        else:
            blank_streak = 0
            out.append(l)
    return '\n'.join(out).strip() + '\n'


