"""Personal-data redaction shared by the gateway guardrail and the audit log."""
import re

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "<CARD>"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    (re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d\b"), "<PHONE>"),
]


def redact(text):
    """Return (redacted_text, number_of_replacements)."""
    count = 0
    for pat, repl in PII_PATTERNS:
        text, n = pat.subn(repl, text)
        count += n
    return text, count
