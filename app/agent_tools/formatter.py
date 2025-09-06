from langchain_core.messages import HumanMessage, SystemMessage
from app.services.llms.azure_openai import chat_model
import re


def _detect_duplicate_sections(content: str) -> bool:
    lines = content.split('\n')
    headings = [line.strip() for line in lines if line.strip() and (line.startswith('#') or line.isupper() or 'Executive Summary' in line or 'Company/Market Overview' in line)]
    seen_headings = set()
    for heading in headings:
        normalized = heading.lower().strip('#').strip()
        if normalized in seen_headings and len(normalized) > 10:
            return True
        seen_headings.add(normalized)
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
                "You are a financial content formatter. Your job is to take raw financial research content and format it professionally. "
                "Structure the content with clear headings, tables for numerical data, and proper citations. "
                "Ensure professional presentation suitable for a financial platform. "
                "Begin with a 1–2 sentence 'Chart insight' lead if present, then ensure correct markdown spacing (blank line before/after '##' headings). "
                "Do not include raw JSON or fenced code blocks in the final output."
            )
            user_msg = (
                f"Format this financial research content for professional presentation:\n\n{content}\n\n"
                f"Requirements:\n"
                f"- Clear headings and structure\n"
                f"- Use tables for numerical comparisons\n"
                f"- Bold important metrics\n"
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


