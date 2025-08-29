import re
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())
