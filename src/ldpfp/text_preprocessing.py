"""Phase 3: Clean literature text and assemble per-protein corpora T(p_i)."""
from __future__ import annotations
import re
import sqlite3
from lxml import etree

WHITESPACE_RE = re.compile(r"\s+")


def strip_jats_xml(xml_str: str) -> str:
    """Extract plain-text body from a PMC JATS XML full-text document."""
    try:
        root = etree.fromstring(xml_str.encode("utf-8"))
    except Exception:
        return ""
    body = root.find(".//body")
    if body is None:
        return ""
    texts = body.itertext()
    return normalize_text(" ".join(texts))


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def assemble_document(title: str | None, abstract: str | None, full_text_xml: str | None) -> str:
    """Aggregate one publication's text: Aⱼ ∪ Bⱼ from Equation 6."""
    parts = []
    if title:
        parts.append(normalize_text(title))
    if abstract:
        parts.append(normalize_text(abstract))
    if full_text_xml:
        body = strip_jats_xml(full_text_xml)
        if body:
            parts.append(body)
    return " ".join(parts)


def load_and_assemble(db_path: str) -> dict[str, str]:
    """Return {pmid: assembled_plain_text} for every successfully cached PMID."""
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT pmid, title, abstract, full_text FROM literature WHERE status = 'ok'"
    )
    out = {}
    for pmid, title, abstract, full_text in cur.fetchall():
        doc = assemble_document(title, abstract, full_text)
        if doc:
            out[pmid] = doc
    conn.close()
    return out
