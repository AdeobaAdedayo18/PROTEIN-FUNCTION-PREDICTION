"""Phase 2: Retrieve abstracts and PMC full text for cached PMIDs."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import requests
from Bio import Entrez
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


# Entrez.email = os.environ["NCBI_EMAIL"]
# Entrez.api_key = os.environ.get("NCBI_API_KEY")
# Entrez.tool = "ldpfp-literature-retrieval"
Entrez.email = os.getenv("NCBI_EMAIL", "test@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY")
Entrez.tool = "ldpfp-literature-retrieval"

BATCH_SIZE = 200

DDL = """
CREATE TABLE IF NOT EXISTS literature (
    pmid TEXT PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    full_text TEXT,
    has_full_text INTEGER DEFAULT 0,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'ok'
);
"""


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(DDL)
    return conn


def chunked(items: list[str], size: int = BATCH_SIZE):
    for i in range(0, len(items), size):
        yield items[i:i + size]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def fetch_abstract_batch(pmids: list[str]) -> dict[str, dict]:
    """
    Fetch up to ~200 PubMed records in one EFetch request.

    Returns:
        {
            "123": {
                "title": "...",
                "abstract": "...",
            }
        }
    """
    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml",
    )

    records = Entrez.read(handle)
    handle.close()

    results: dict[str, dict] = {}

    for record in records.get("PubmedArticle", []):
        citation = record["MedlineCitation"]
        article = citation["Article"]

        pmid = str(citation["PMID"])

        title = str(article.get("ArticleTitle", ""))

        abstract_parts = (
            article
            .get("Abstract", {})
            .get("AbstractText", [])
        )

        abstract = " ".join(str(part) for part in abstract_parts)

        results[pmid] = {
            "title": title,
            "abstract": abstract,
        }

    return results


def fetch_abstract(pmid: str) -> dict:
    """
    Backwards-compatible single-PMID helper, mainly useful for tests.
    """
    result = fetch_abstract_batch([pmid])

    if pmid not in result:
        raise ValueError(f"PMID {pmid} was not returned by PubMed")

    return result[pmid]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def resolve_pmcids(pmids: list[str]) -> dict[str, str]:
    """
    Resolve up to 200 PMIDs -> PMCIDs using PMC ID Converter.

    Only PMIDs present in PMC will appear in the returned mapping.
    """
    response = requests.get(
        "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
        params={
            "ids": ",".join(pmids),
            "idtype": "pmid",
            "format": "json",
            "tool": Entrez.tool,
            "email": Entrez.email,
        },
        timeout=60,
    )

    response.raise_for_status()

    payload = response.json()

    mapping: dict[str, str] = {}

    for record in payload.get("records", []):
        pmid = record.get("pmid")
        pmcid = record.get("pmcid")

        if pmid and pmcid:
            mapping[str(pmid)] = str(pmcid)

    return mapping


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
)
def fetch_full_text_by_pmcid(pmcid: str) -> str:
    """
    Fetch PMC JATS XML for an already-resolved PMCID.
    """
    handle = Entrez.efetch(
        db="pmc",
        id=pmcid,
        rettype="full",
        retmode="xml",
    )

    raw = handle.read()
    handle.close()

    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")

    return str(raw)


def retrieve_all(
    pmids: list[str],
    db_path: str | Path,
    with_full_text: bool = True,
) -> None:

    conn = get_conn(db_path)
    cur = conn.cursor()

    cur.execute(
        "SELECT pmid FROM literature WHERE status = 'ok'"
    )

    already = {row[0] for row in cur.fetchall()}

    todo = [p for p in pmids if p not in already]

    print(f"{len(already):,} cached, {len(todo):,} to fetch")

    batches = list(chunked(todo))

    for batch in tqdm(
        batches,
        desc="Fetching PubMed batches",
        unit="batch",
    ):
        try:
            # --------------------------------------------------
            # 1. Fetch ALL titles + abstracts in one request
            # --------------------------------------------------
            metadata = fetch_abstract_batch(batch)

            # --------------------------------------------------
            # 2. Resolve PMC IDs in one request
            # --------------------------------------------------
            pmc_map = {}

            if with_full_text:
                pmc_map = resolve_pmcids(batch)

            # --------------------------------------------------
            # 3. Save each PMID
            # --------------------------------------------------
            for pmid in batch:
                try:
                    meta = metadata.get(pmid)

                    if meta is None:
                        cur.execute(
                            """
                            INSERT OR REPLACE INTO literature
                            (pmid, status)
                            VALUES (?, ?)
                            """,
                            (
                                pmid,
                                "error: PMID not returned by PubMed",
                            ),
                        )
                        continue

                    full_text = None
                    has_ft = 0

                    pmcid = pmc_map.get(pmid)

                    if pmcid:
                        try:
                            full_text = fetch_full_text_by_pmcid(pmcid)
                            has_ft = 1
                        except Exception as exc:
                            # Abstract retrieval still succeeded.
                            # Keep record usable even if PMC fails.
                            full_text = None
                            has_ft = 0

                            print(
                                f"\nPMC fetch failed for "
                                f"{pmid} ({pmcid}): {exc}"
                            )

                    cur.execute(
                        """
                        INSERT OR REPLACE INTO literature
                        (
                            pmid,
                            title,
                            abstract,
                            full_text,
                            has_full_text,
                            status
                        )
                        VALUES (?, ?, ?, ?, ?, 'ok')
                        """,
                        (
                            pmid,
                            meta["title"],
                            meta["abstract"],
                            full_text,
                            has_ft,
                        ),
                    )

                except Exception as exc:
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO literature
                        (pmid, status)
                        VALUES (?, ?)
                        """,
                        (
                            pmid,
                            f"error: {exc}",
                        ),
                    )

            # Commit once per batch instead of once per PMID
            conn.commit()

        except Exception as exc:
            # If the whole batch request fails,
            # mark the batch as failed so it retries next run.
            for pmid in batch:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO literature
                    (pmid, status)
                    VALUES (?, ?)
                    """,
                    (
                        pmid,
                        f"error: batch failure: {exc}",
                    ),
                )

            conn.commit()

    conn.close()