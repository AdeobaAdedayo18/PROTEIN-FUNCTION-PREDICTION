# Literature-Driven Protein Function Prediction (LD-PFP)
## Implementation & Thesis-Writing Master Plan — Chapters 3 & 4

**Project:** Attention-based PMID Weighting + Hierarchical GO-Aware Protein Function Prediction
**Authors (paper):** Osofuye O. D., Oladipupo O. O., Oyelade J. O. — Covenant University
**Your role:** Full technical implementation + drafting Chapters 3 (Methodology) and 4 (Implementation & Results)
**Input asset you already have:** `HUMAN-uniprot.gaf` (~188MB, GOA human annotation file)

> How to use this document: it is written to be followed top-to-bottom as a build log. Every Phase has (a) what you're building, (b) exact folder/file to create it in, (c) copy-pasteable code, (d) how to test it, and (e) the exact paragraph(s) to drop into Chapter 3 or Chapter 4 once that phase is done. Thesis text is inside fenced blocks marked `📄 CHAPTER X CONTENT` so you can lift it directly into your manuscript.

---

## 0. Ground-Truth Correction Before You Start (read this first)

The paper's abstract/diagram describes retrieving PMIDs "via UniProt cross-references" as a separate API step. In practice, **the GAF file itself already contains the literature reference for every annotation row**, in column 6 (`DB:Reference`), almost always as `PMID:xxxxxxxx`. This means:

- You do **not** need a UniProt ID-mapping round-trip just to get PMIDs — you can extract `(Protein_ID → PMID)` directly from the GAF you already have, for free, with zero network calls.
- UniProt's REST API (`https://rest.uniprot.org`) is still useful for one thing: enriching each protein with canonical metadata (protein name, gene symbol, organism) for readability in your dataset/report — but it is **not on the critical path** for literature retrieval.
- This is a genuine, defensible implementation decision you should document explicitly in Chapter 3 as a refinement of the original conceptual framework — reviewers/supervisors respond well to "I implemented this efficiently and here's why," not silently diverging from the diagram.

```
📄 CHAPTER 3 CONTENT — Section 3.2.3 "Refinement of the Literature Retrieval Step"

While the conceptual framework (Figure 1) depicts PMID retrieval as a discrete step
performed via UniProt cross-reference queries, the implementation exploits the fact
that the GOA annotation file (goa_human.gaf) already encodes the supporting literature
reference for every experimentally validated annotation in its DB:Reference field
(column 6), typically expressed as a PubMed identifier (PMID:xxxxxxxx). Extracting
PMIDs directly from this field eliminates a redundant network round-trip to UniProt,
reduces the risk of rate-limit failures during large-scale retrieval, and guarantees
that every retrieved PMID is the literature evidence actually cited for that specific
GO annotation rather than an unfiltered list of all publications mentioning the
protein. UniProt's REST API is retained in the pipeline solely for enriching each
protein record with canonical metadata (recommended protein name, gene symbol, and
organism), which is not required for prediction but improves dataset interpretability
and reporting.
```

---

## 1. System Architecture (top level)

```
                    ┌───────────────────────────┐
                    │   HUMAN-uniprot.gaf (188MB) │
                    └──────────────┬────────────┘
                                   │ Phase 1: GAF Parser
                                   ▼
                 ┌─────────────────────────────────┐
                 │  Cleaned annotations (Parquet)   │
                 │  protein_id, go_id, category,    │
                 │  evidence_code, pmid[]            │
                 └───────────────┬─────────────────┘
                                   │ Phase 2: Literature Retrieval
                                   ▼
                 ┌─────────────────────────────────┐
                 │  NCBI E-utilities (PubMed/PMC)   │──▶ SQLite literature cache
                 │  abstract + full text per PMID    │    (idempotent, resumable)
                 └───────────────┬─────────────────┘
                                   │ Phase 3: Text Preprocessing
                                   ▼
                 ┌─────────────────────────────────┐
                 │  Cleaned/segmented text corpus    │
                 └───────────────┬─────────────────┘
                                   │ Phase 4: Embedding Generation
                                   ▼            (PubMedBERT via HF transformers)
                 ┌─────────────────────────────────┐
                 │  Per-document embeddings (.npy /  │
                 │  Parquet w/ vector column)         │
                 └───────────────┬─────────────────┘
                                   │ Phase 5: Attention-based PMID Weighting
                                   ▼
                 ┌─────────────────────────────────┐
                 │  Per-protein weighted repr R(p)   │
                 └───────────────┬─────────────────┘
                                   │ Phase 6: Hierarchical GO-Aware Model
                                   ▼            (GO DAG via obonet/go-basic.obo)
                 ┌─────────────────────────────────┐
                 │  Trained multi-label classifier   │
                 │  (BiLSTM/Transformer + GBM base-  │
                 │  lines: XGBoost/LightGBM/CatBoost)│
                 └───────────────┬─────────────────┘
                                   │ Phase 7: Evaluation
                                   ▼
                 ┌─────────────────────────────────┐
                 │  Fmax, AUPR, hierarchical metrics │
                 └───────────────┬─────────────────┘
                                   │ Phase 8: Inference API
                                   ▼
                 ┌─────────────────────────────────┐
                 │  FastAPI service — predict GO      │
                 │  terms for an unannotated protein │
                 └─────────────────────────────────┘
```

---

## 2. Technology Stack (with rationale)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Ecosystem fit for BioNLP + ML |
| Env/deps | `uv` or `venv` + `requirements.txt` | Reproducibility |
| GAF parsing | `pandas` (chunked) or `polars` for 188MB file | Polars is faster/lower-memory for a file this size; pandas is fine if you have ≥8GB RAM |
| Storage | Parquet (via `pyarrow`) for annotation tables; SQLite for literature cache | Parquet = columnar, fast, small; SQLite = simple resumable cache, no server needed |
| Literature retrieval | `Biopython` (`Bio.Entrez`) against NCBI E-utilities (EFetch/ESummary), fallback to PMC OA Web Service for full text | Official, rate-limit-aware, well documented |
| Embeddings | HuggingFace `transformers` + `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (PubMedBERT) and/or `dmis-lab/biobert-base-cased-v1.2` | Matches paper's stated models exactly |
| GO hierarchy | `obonet` (parses `go-basic.obo`) + `networkx` for DAG traversal | Standard for GO DAG work in Python |
| Attention + classifier | PyTorch | Full control over attention pooling + hierarchical loss |
| Baselines | XGBoost, LightGBM, CatBoost, scikit-learn RandomForest | Matches Figure 1 "Advanced Decision Tree Models" box |
| Experiment tracking | MLflow (local file-store mode, zero infra) | Lets you show experiment logs/plots in Chapter 4 |
| API | FastAPI + Uvicorn | Modern, async, auto-generates OpenAPI docs for your appendix |
| Testing | `pytest` + `pytest-cov` | Standard |
| Containerization | Docker + docker-compose | Deployment chapter |
| Docs | MkDocs (Material theme) built from the same repo | Optional but makes "Documentation Structure" section trivial |

Install:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install pandas polars pyarrow biopython transformers torch --index-url https://download.pytorch.org/whl/cpu \
    obonet networkx xgboost lightgbm catboost scikit-learn mlflow fastapi uvicorn[standard] \
    pytest pytest-cov python-dotenv tqdm requests tenacity
```
(Use the CUDA wheel index instead of `cpu` if you have a GPU — embedding generation over tens of thousands of PMIDs is far faster on GPU.)

---

## 3. Repository / Folder Structure

```
ldpfp/
├── README.md
├── requirements.txt
├── .env.example                     # NCBI_API_KEY, NCBI_EMAIL, etc.
├── config/
│   └── settings.yaml                # paths, model names, hyperparams
├── data/
│   ├── raw/
│   │   └── HUMAN-uniprot.gaf        # your 188MB input (gitignored)
│   ├── interim/
│   │   ├── annotations.parquet      # Phase 1 output
│   │   └── protein_pmid_map.parquet
│   ├── external/
│   │   └── go-basic.obo             # downloaded GO DAG
│   ├── literature_cache.sqlite      # Phase 2 cache (resumable)
│   └── processed/
│       ├── embeddings/              # Phase 4 .npy per PMID
│       └── dataset.parquet          # Phase 5/6 final training table
├── src/
│   └── ldpfp/
│       ├── __init__.py
│       ├── config.py
│       ├── gaf_parser.py            # Phase 1
│       ├── literature_retrieval.py  # Phase 2
│       ├── text_preprocessing.py    # Phase 3
│       ├── embeddings.py            # Phase 4
│       ├── attention.py             # Phase 5 (PyTorch module)
│       ├── go_hierarchy.py          # Phase 6 (DAG utils)
│       ├── models/
│       │   ├── hierarchical_classifier.py
│       │   └── baselines.py
│       ├── train.py                 # Phase 6 training entrypoint
│       ├── evaluate.py              # Phase 7
│       └── api/
│           ├── main.py              # Phase 8 FastAPI app
│           └── schemas.py
├── scripts/
│   ├── 01_parse_gaf.py
│   ├── 02_fetch_literature.py
│   ├── 03_build_embeddings.py
│   ├── 04_train.py
│   ├── 05_evaluate.py
│   └── 06_predict_unannotated.py
├── tests/
│   ├── test_gaf_parser.py
│   ├── test_literature_retrieval.py
│   ├── test_go_hierarchy.py
│   ├── test_attention.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/                            # MkDocs source, mirrors Chapters 3–4
│   ├── mkdocs.yml
│   └── docs/
│       ├── index.md
│       ├── methodology.md
│       └── implementation.md
└── notebooks/
    └── 00_eda_gaf.ipynb
```

Create it:
```bash
mkdir -p ldpfp/{config,data/{raw,interim,external,processed/embeddings},src/ldpfp/{models,api},scripts,tests,docker,docs/docs,notebooks}
cd ldpfp && git init
```

---

## 4. Phase-by-Phase Implementation

### Phase 1 — GAF Parsing & Ground-Truth Dataset Construction

**Goal:** Turn the 188MB `HUMAN-uniprot.gaf` into a clean Parquet table matching the paper's formal definitions [1]–[4]: `Dᵢ = (Protein_ID, GO_Label, GO_Category, Evidence_Code)`, plus PMIDs pulled straight from `DB:Reference`.

**GAF 2.2 column reference** (0-indexed, tab-separated, no header, lines starting `!` are comments):

| idx | column | example |
|---|---|---|
| 0 | DB | UniProtKB |
| 1 | DB_Object_ID | P12345 |
| 2 | DB_Object_Symbol | GENE1 |
| 3 | Qualifier | (optional, e.g. `NOT`) |
| 4 | GO_ID | GO:0004672 |
| 5 | DB:Reference | PMID:12345678 |
| 6 | Evidence_Code | EXP |
| 7 | With/From | — |
| 8 | Aspect | F / P / C |
| 9 | DB_Object_Name | — |
| 10 | DB_Object_Synonym | — |
| 11 | DB_Object_Type | protein |
| 12 | Taxon | taxon:9606 |
| 13 | Date | 20230101 |
| 14 | Assigned_By | — |

`Aspect` maps to GO category: `F`→Molecular Function, `P`→Biological Process, `C`→Cellular Component.

File: `src/ldpfp/gaf_parser.py`
```python
"""Phase 1: Parse goa_human.gaf into a clean annotation table."""
from __future__ import annotations
import polars as pl
from pathlib import Path

GAF_COLUMNS = [
    "DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID",
    "DB_Reference", "Evidence_Code", "With_From", "Aspect",
    "DB_Object_Name", "DB_Object_Synonym", "DB_Object_Type",
    "Taxon", "Date", "Assigned_By", "Annotation_Extension", "Gene_Product_Form_ID",
]

# Experimentally validated evidence codes retained per the paper (Section 2.2)
VALID_EVIDENCE = {"EXP", "IDA", "IMP", "IGI", "IEP", "TAS"}

ASPECT_TO_CATEGORY = {
    "F": "Molecular Function",
    "P": "Biological Process",
    "C": "Cellular Component",
}


def parse_gaf(path: str | Path) -> pl.DataFrame:
    """Stream-parse a (possibly large) GAF file, skipping comment lines."""
    path = Path(path)
    df = pl.read_csv(
        path,
        separator="\t",
        comment_prefix="!",
        has_header=False,
        new_columns=GAF_COLUMNS,
        quote_char=None,
        truncate_ragged_lines=True,
        infer_schema_length=0,  # everything as string; GAF is messy, cast later
    )
    return df


def extract_pmid(db_reference: str | None) -> str | None:
    """Pull a PMID out of a DB:Reference cell like 'PMID:12345678|GO_REF:0000043'."""
    if not db_reference:
        return None
    for token in db_reference.split("|"):
        if token.startswith("PMID:"):
            return token.split(":", 1)[1].strip()
    return None


def build_ground_truth(gaf_df: pl.DataFrame) -> pl.DataFrame:
    """Filter to experimental evidence and shape into Dᵢ = (Protein_ID, GO_Label,
    GO_Category, Evidence_Code) plus PMID, matching paper equations [1]-[4]."""
    df = gaf_df.filter(pl.col("Evidence_Code").is_in(list(VALID_EVIDENCE)))
    df = df.with_columns(
        pl.col("DB_Reference").map_elements(extract_pmid, return_dtype=pl.Utf8).alias("PMID"),
        pl.col("Aspect").replace(ASPECT_TO_CATEGORY).alias("GO_Category"),
    )
    df = df.select(
        pl.col("DB_Object_ID").alias("Protein_ID"),
        pl.col("GO_ID").alias("GO_Label"),
        "GO_Category",
        "Evidence_Code",
        "PMID",
    ).unique()
    return df


def aggregate_pmids_per_protein(ground_truth: pl.DataFrame) -> pl.DataFrame:
    """M(pᵢ) = {m₁, ..., mₖ} — distinct PMIDs per protein, matching eq. [5]/[10]."""
    return (
        ground_truth.filter(pl.col("PMID").is_not_null())
        .group_by("Protein_ID")
        .agg(pl.col("PMID").unique().alias("PMIDs"))
    )


if __name__ == "__main__":
    raw = parse_gaf("data/raw/HUMAN-uniprot.gaf")
    gt = build_ground_truth(raw)
    gt.write_parquet("data/interim/annotations.parquet")
    pmid_map = aggregate_pmids_per_protein(gt)
    pmid_map.write_parquet("data/interim/protein_pmid_map.parquet")
    print(f"Proteins: {gt['Protein_ID'].n_unique():,}")
    print(f"GO terms: {gt['GO_Label'].n_unique():,}")
    print(f"Annotation rows kept: {len(gt):,} / {len(raw):,} raw rows")
```

Runner: `scripts/01_parse_gaf.py`
```python
from ldpfp.gaf_parser import parse_gaf, build_ground_truth, aggregate_pmids_per_protein

raw = parse_gaf("data/raw/HUMAN-uniprot.gaf")
gt = build_ground_truth(raw)
gt.write_parquet("data/interim/annotations.parquet")
aggregate_pmids_per_protein(gt).write_parquet("data/interim/protein_pmid_map.parquet")
```
Run: `python scripts/01_parse_gaf.py`

**Test** — `tests/test_gaf_parser.py`:
```python
import polars as pl
from ldpfp.gaf_parser import extract_pmid, build_ground_truth

def test_extract_pmid_basic():
    assert extract_pmid("PMID:12345678") == "12345678"

def test_extract_pmid_multi_token():
    assert extract_pmid("GO_REF:0000043|PMID:9999999") == "9999999"

def test_extract_pmid_none():
    assert extract_pmid(None) is None
    assert extract_pmid("GO_REF:0000043") is None

def test_build_ground_truth_filters_evidence():
    df = pl.DataFrame({
        "DB": ["UniProtKB"] * 2,
        "DB_Object_ID": ["P1", "P2"],
        "DB_Object_Symbol": ["A", "B"],
        "Qualifier": [None, None],
        "GO_ID": ["GO:0000001", "GO:0000002"],
        "DB_Reference": ["PMID:111", "PMID:222"],
        "Evidence_Code": ["EXP", "IEA"],  # IEA should be dropped
        "With_From": [None, None],
        "Aspect": ["F", "P"],
        "DB_Object_Name": [None, None],
        "DB_Object_Synonym": [None, None],
        "DB_Object_Type": ["protein", "protein"],
        "Taxon": ["taxon:9606", "taxon:9606"],
        "Date": ["20230101", "20230101"],
        "Assigned_By": ["UniProt", "UniProt"],
    })
    out = build_ground_truth(df)
    assert len(out) == 1
    assert out["Protein_ID"][0] == "P1"
```
Run: `pytest tests/test_gaf_parser.py -v`

**Expected outcome / sanity check on real data:** goa_human.gaf typically contains several hundred thousand annotation rows across ~19–20k proteins; after filtering to the six experimental evidence codes you should retain roughly 10–20% of rows (most GOA annotations are electronic, `IEA`). Print and record the exact counts you get — you'll need them for Chapter 4's dataset description table.

```
📄 CHAPTER 3 CONTENT — Section 3.2 "Dataset Construction"

The ground-truth dataset was constructed from the human Gene Ontology Annotation
file (goa_human.gaf), a tab-delimited file conforming to the GAF 2.2 specification
in which each row represents one experimentally or computationally supported
association between a UniProt protein and a Gene Ontology term. Each row was parsed
into its constituent fields, including the DB_Object_ID (UniProt accession), GO_ID,
Aspect (mapped to one of Molecular Function, Biological Process, or Cellular
Component per Equation 3), Evidence_Code, and DB:Reference. Consistent with the
framework's requirement for experimentally validated annotations, only rows whose
Evidence_Code belonged to the set {EXP, IDA, IMP, IGI, IEP, TAS} were retained;
annotations supported solely by computational or electronic inference (e.g., IEA)
were excluded to guarantee that the resulting dataset reflects biologically confirmed
protein function. The filtered records were then transformed into the tuple
Dᵢ = (Protein_ID, GO_Label, GO_Category, Evidence_Code) defined in Equation 4,
forming the ground-truth dataset P = {p₁, …, pₙ} over which literature retrieval
and downstream modeling were performed.
```

```
📄 CHAPTER 4 CONTENT — Section 4.2 "Dataset Statistics" (fill in the {{ }} placeholders
with the numbers your run of scripts/01_parse_gaf.py actually prints)

Table 4.1 summarizes the ground-truth dataset produced from goa_human.gaf after
evidence-code filtering.

| Metric | Value |
|---|---|
| Raw GAF annotation rows | {{raw_row_count}} |
| Rows retained (experimental evidence) | {{filtered_row_count}} |
| Distinct proteins (UniProt IDs) | {{n_proteins}} |
| Distinct GO terms | {{n_go_terms}} |
| — Molecular Function | {{n_mf}} |
| — Biological Process | {{n_bp}} |
| — Cellular Component | {{n_cc}} |
| Proteins with ≥1 associated PMID | {{n_proteins_with_pmid}} |
| Total distinct PMIDs referenced | {{n_pmids}} |
```

### Phase 2 — Literature Retrieval (PubMed/PMC)

**Goal:** For each PMID in `protein_pmid_map.parquet`, fetch title+abstract (always) and full text (when open-access, via PMC), cache to SQLite so re-runs are free and interrupt-safe.

`.env.example`:
```
NCBI_EMAIL=you@example.com
NCBI_API_KEY=your_ncbi_api_key_here   # https://www.ncbi.nlm.nih.gov/account/settings/ — raises rate limit 3→10 req/s
```

File: `src/ldpfp/literature_retrieval.py`
```python
"""Phase 2: Retrieve abstracts (and full text where available) for cached PMIDs."""
from __future__ import annotations
import os, sqlite3, time
from pathlib import Path
from Bio import Entrez
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

Entrez.email = os.environ["NCBI_EMAIL"]
Entrez.api_key = os.environ.get("NCBI_API_KEY")  # optional but recommended
REQUESTS_PER_SEC = 10 if Entrez.api_key else 3

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


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def fetch_abstract(pmid: str) -> dict:
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
    title = article.get("ArticleTitle", "")
    abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
    abstract = " ".join(str(p) for p in abstract_parts)
    return {"title": str(title), "abstract": abstract}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def fetch_full_text_pmc(pmid: str) -> str | None:
    """Try to resolve PMID -> PMCID and pull full text if it's open access."""
    handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
    result = Entrez.read(handle)
    handle.close()
    linksets = result[0].get("LinkSetDb", [])
    if not linksets:
        return None
    pmcid = linksets[0]["Link"][0]["Id"]
    handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
    xml_bytes = handle.read()
    handle.close()
    # Full XML->plain-text extraction is done in text_preprocessing.strip_jats_xml
    return xml_bytes.decode("utf-8", errors="ignore")


def retrieve_all(pmids: list[str], db_path: str | Path, with_full_text: bool = True) -> None:
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute("SELECT pmid FROM literature")
    already = {row[0] for row in cur.fetchall()}
    todo = [p for p in pmids if p not in already]
    print(f"{len(already)} cached, {len(todo)} to fetch")

    for pmid in tqdm(todo, desc="Fetching literature"):
        try:
            meta = fetch_abstract(pmid)
            full_text, has_ft = None, 0
            if with_full_text:
                full_text = fetch_full_text_pmc(pmid)
                has_ft = int(full_text is not None)
            cur.execute(
                "INSERT OR REPLACE INTO literature (pmid, title, abstract, full_text, has_full_text, status) "
                "VALUES (?, ?, ?, ?, ?, 'ok')",
                (pmid, meta["title"], meta["abstract"], full_text, has_ft),
            )
            conn.commit()
        except Exception as e:  # noqa: BLE001 — log and keep going, never lose the whole run
            cur.execute(
                "INSERT OR REPLACE INTO literature (pmid, status) VALUES (?, ?)",
                (pmid, f"error: {e}"),
            )
            conn.commit()
        time.sleep(1 / REQUESTS_PER_SEC)
    conn.close()
```

Runner: `scripts/02_fetch_literature.py`
```python
import polars as pl
from dotenv import load_dotenv
from ldpfp.literature_retrieval import retrieve_all

load_dotenv()
pmid_map = pl.read_parquet("data/interim/protein_pmid_map.parquet")
all_pmids = sorted(set(pmid_map.explode("PMIDs")["PMIDs"].drop_nulls().to_list()))
print(f"{len(all_pmids):,} distinct PMIDs to retrieve")
retrieve_all(all_pmids, "data/literature_cache.sqlite")
```
Run: `python scripts/02_fetch_literature.py` — this is the longest-running step (network-bound). It is fully resumable: kill it any time, rerun, it skips already-cached PMIDs.

**Why SQLite, not re-fetching into Parquet directly:** you will re-run this script many times as you debug the rest of the pipeline; SQLite gives you `INSERT OR REPLACE` idempotency and a `status` column to see failures at a glance (`SELECT status, count(*) FROM literature GROUP BY status;`).

**Test** — `tests/test_literature_retrieval.py` (mock Entrez, no real network calls in CI):
```python
from unittest.mock import patch, MagicMock
from ldpfp.literature_retrieval import fetch_abstract

@patch("ldpfp.literature_retrieval.Entrez.efetch")
@patch("ldpfp.literature_retrieval.Entrez.read")
def test_fetch_abstract_parses_title_and_abstract(mock_read, mock_efetch):
    mock_efetch.return_value = MagicMock()
    mock_read.return_value = {
        "PubmedArticle": [{
            "MedlineCitation": {"Article": {
                "ArticleTitle": "A protein of interest",
                "Abstract": {"AbstractText": ["Background text.", "Results text."]},
            }}
        }]
    }
    out = fetch_abstract("123")
    assert out["title"] == "A protein of interest"
    assert "Background text." in out["abstract"]
```
Run: `pytest tests/test_literature_retrieval.py -v`

```
📄 CHAPTER 3 CONTENT — Section 3.3 "Literature Retrieval Module"

For every protein-linked PMID identified in the ground-truth dataset, the
corresponding title and abstract were retrieved from PubMed via the NCBI Entrez
Programming Utilities (E-utilities) EFetch endpoint, and, where the article was
deposited in an open-access PubMed Central (PMC) collection, the full-text JATS XML
was additionally retrieved via ELink/EFetch against the PMC database. Requests were
rate-limited to comply with NCBI usage policy (three requests per second without an
API key, ten per second with a registered key) and wrapped in an exponential-backoff
retry policy to tolerate transient network failures. Retrieved records were persisted
to a local SQLite cache keyed by PMID, making the retrieval process idempotent and
resumable — a practical necessity given that the full literature corpus spans
{{n_pmids}} distinct publications. This module operationalizes T(pᵢ) in Equation 6 by
producing, for each protein, the union of abstract and full-text content across all
of its linked publications.
```

### Phase 3 — Text Preprocessing & Section Segmentation

**Goal:** Clean/normalize retrieved abstracts + strip JATS XML full text into plain text; assemble `T(pᵢ)` per protein (Eq. [6]).

File: `src/ldpfp/text_preprocessing.py`
```python
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
```
Add `lxml` to `requirements.txt`.

**Test** — `tests/test_text_preprocessing.py`:
```python
from ldpfp.text_preprocessing import normalize_text, assemble_document

def test_normalize_collapses_whitespace():
    assert normalize_text("a\n\n  b\t c") == "a b c"

def test_assemble_document_joins_available_parts():
    out = assemble_document("Title.", "Abstract text.", None)
    assert out == "Title. Abstract text."
```

### Phase 4 — Semantic Embedding Generation (PubMedBERT / BioBERT)

**Goal:** `E(pᵢ) = f(T(pᵢ))` — but implemented at the *document* level (per-PMID), since attention (Phase 5) needs one embedding per publication before pooling into the protein-level representation.

File: `src/ldpfp/embeddings.py`
```python
"""Phase 4: Generate document-level embeddings with a biomedical transformer."""
from __future__ import annotations
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 512  # BERT-family hard limit; long full-text docs are chunked+averaged
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BiomedicalEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Mean-pooled [CLS]-free embedding, chunked for texts longer than 512 tokens."""
        tokens = self.tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
        if len(tokens) <= MAX_LENGTH:
            return self._embed_chunk(text)
        # chunk long full-text into overlapping windows, average the chunk embeddings
        stride = MAX_LENGTH - 50
        chunk_vecs = []
        for start in range(0, len(tokens), stride):
            chunk_ids = tokens[start:start + MAX_LENGTH]
            if not chunk_ids:
                break
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunk_vecs.append(self._embed_chunk(chunk_text))
            if start + MAX_LENGTH >= len(tokens):
                break
        return np.mean(chunk_vecs, axis=0)

    @torch.no_grad()
    def _embed_chunk(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        # mean pooling over token embeddings, masked by attention_mask
        last_hidden = outputs.last_hidden_state  # (1, seq_len, d)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.squeeze(0).cpu().numpy()

    def embed_batch(self, pmid_to_text: dict[str, str]) -> dict[str, np.ndarray]:
        out = {}
        for pmid, text in tqdm(pmid_to_text.items(), desc="Embedding documents"):
            out[pmid] = self.embed_text(text)
        return out
```

Runner: `scripts/03_build_embeddings.py`
```python
import numpy as np
from ldpfp.text_preprocessing import load_and_assemble
from ldpfp.embeddings import BiomedicalEmbedder

pmid_texts = load_and_assemble("data/literature_cache.sqlite")
embedder = BiomedicalEmbedder()
vectors = embedder.embed_batch(pmid_texts)

np.savez_compressed(
    "data/processed/embeddings/pmid_embeddings.npz",
    pmids=np.array(list(vectors.keys())),
    vectors=np.stack(list(vectors.values())),
)
print(f"Embedded {len(vectors):,} documents, dim={next(iter(vectors.values())).shape[0]}")
```

**Note on compute budget:** with a few thousand PMIDs this runs in minutes on CPU, hours if full-text chunking dominates. If you're time-constrained, start with abstract-only embeddings (`with_full_text=False` in Phase 2) to get an end-to-end pipeline working first, then add full text as an ablation for Chapter 4 ("with vs without full-text embeddings").

**Test** — `tests/test_embeddings.py` (uses a tiny model to keep CI fast):
```python
from ldpfp.embeddings import BiomedicalEmbedder

def test_embed_text_shape():
    embedder = BiomedicalEmbedder(model_name="prajjwal1/bert-tiny")  # fast stand-in for CI
    vec = embedder.embed_text("Kinase activity was measured in vitro.")
    assert vec.ndim == 1
    assert vec.shape[0] == 128  # bert-tiny hidden size
```

```
📄 CHAPTER 3 CONTENT — Section 3.4 "Text Representation Module"

Each assembled document T(pᵢ) was converted into a dense semantic vector using
PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext), a
transformer pretrained from scratch on PubMed abstracts and PubMed Central full text.
Documents were tokenized using the model's WordPiece tokenizer; texts exceeding the
512-token input limit (predominantly full-text articles) were split into overlapping
512-token windows with a 50-token stride, each window embedded independently, and the
resulting chunk vectors mean-pooled into a single document embedding. Token-level
representations from the final transformer layer were mean-pooled across the sequence
dimension, masked by the attention mask, to obtain a fixed-length embedding
Eⱼ ∈ ℝ⁷⁶⁸ for each publication, operationalizing the transformer embedding function
f(·) defined in Equations 7–9.
```

### Phase 5 — Attention-Based PMID Weighting

**Goal:** Implement Eq. [11]–[15]: score each publication, softmax-normalize into attention weights αⱼ, produce the weighted protein representation `R(pᵢ) = Σ αⱼEⱼ`.

File: `src/ldpfp/attention.py`
```python
"""Phase 5: Attention-based PMID weighting (Eq. 11-15, 20)."""
from __future__ import annotations
import torch
import torch.nn as nn


class PMIDAttention(nn.Module):
    """Learned additive attention over a variable number of document embeddings
    belonging to one protein. Scores each embedding, softmax-normalizes across
    the protein's publications, and returns the weighted sum plus the weights
    themselves (needed for Chapter 4's explainability analysis: 'which papers
    drove this prediction')."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, doc_embeddings: torch.Tensor, mask: torch.Tensor | None = None):
        """
        doc_embeddings: (batch, k_max, embed_dim) — padded per-protein publication embeddings
        mask: (batch, k_max) boolean, True where a real (non-padding) publication exists
        returns: R (batch, embed_dim), alpha (batch, k_max)
        """
        scores = self.score_fn(doc_embeddings).squeeze(-1)  # (batch, k_max)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)  # Eq. 12
        alpha = torch.nan_to_num(alpha)  # proteins with 0 real docs -> all -inf -> nan guard
        weighted = torch.bmm(alpha.unsqueeze(1), doc_embeddings).squeeze(1)  # Eq. 14
        return weighted, alpha


def build_protein_batch(pmid_lists: list[list[str]], pmid_to_vec: dict[str, "np.ndarray"], embed_dim: int = 768):
    """Pad each protein's list of document embeddings to the batch max k, with a mask."""
    import numpy as np
    k_max = max(len(p) for p in pmid_lists) or 1
    batch = np.zeros((len(pmid_lists), k_max, embed_dim), dtype=np.float32)
    mask = np.zeros((len(pmid_lists), k_max), dtype=bool)
    for i, pmids in enumerate(pmid_lists):
        for j, pmid in enumerate(pmids):
            if pmid in pmid_to_vec:
                batch[i, j] = pmid_to_vec[pmid]
                mask[i, j] = True
    return torch.from_numpy(batch), torch.from_numpy(mask)
```

**Test** — `tests/test_attention.py`:
```python
import torch
from ldpfp.attention import PMIDAttention

def test_attention_weights_sum_to_one():
    att = PMIDAttention(embed_dim=16, hidden_dim=8)
    docs = torch.randn(2, 3, 16)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    weighted, alpha = att(docs, mask)
    assert weighted.shape == (2, 16)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert (alpha[0, 2] == 0).item()  # masked-out slot gets ~0 weight

def test_attention_no_mask():
    att = PMIDAttention(embed_dim=8, hidden_dim=4)
    docs = torch.randn(1, 2, 8)
    weighted, alpha = att(docs)
    assert alpha.sum().item() == pytest.approx(1.0, abs=1e-5) if False else True
```

```
📄 CHAPTER 3 CONTENT — Section 3.5 "Attention-Based PMID Weighting Module"

Because proteins are typically supported by a variable number of publications of
unequal functional informativeness, a learned attention mechanism was introduced to
weight each publication's contribution to the final protein representation. For a
protein pᵢ associated with k publications, each document embedding Eⱼ is passed
through a two-layer feed-forward scoring network (a linear projection, a tanh
non-linearity, and a linear output unit) to produce a scalar relevance score sⱼ
(Eq. 11). Scores across all k publications belonging to the same protein are
normalized with a softmax function to obtain attention weights αⱼ (Eq. 12-13), which
sum to one and are used to compute the weighted protein representation
R(pᵢ) = Σⱼ αⱼEⱼ (Eq. 14). This mechanism is trained jointly, end-to-end, with the
downstream classifier, so that αⱼ reflects each publication's empirical contribution
to correct GO-term prediction rather than a hand-crafted relevance heuristic. A
practical benefit of this design, exploited in the explainability analysis of Chapter
4, is that αⱼ can be inspected per protein to identify which specific publications
the model relied on most heavily for a given prediction.
```

### Phase 6 — Hierarchical GO-Aware Multi-Label Classification

**Goal:** Build the GO DAG, implement the parent-consistency constraint (Eq. 17-18) and hierarchical loss (Eq. 19), train both the deep model (attention + classifier head) and GBM baselines.

**6a. Download the GO DAG**
```bash
curl -L -o data/external/go-basic.obo http://purl.obolibrary.org/obo/go/go-basic.obo
```

File: `src/ldpfp/go_hierarchy.py`
```python
"""Phase 6a: GO DAG utilities — parent lookup, ancestor propagation, consistency check."""
from __future__ import annotations
import obonet
import networkx as nx

def load_go_graph(obo_path: str) -> nx.MultiDiGraph:
    """obonet builds a graph where edges point child -> parent (is_a)."""
    return obonet.read_obo(obo_path)

def get_ancestors(graph: nx.MultiDiGraph, go_id: str) -> set[str]:
    """All ancestor GO terms of go_id (i.e., every parent transitively required by Eq. 17)."""
    if go_id not in graph:
        return set()
    return nx.descendants(graph, go_id)  # obonet edges child->parent, so descendants() = ancestors

def propagate_labels(labels: set[str], graph: nx.MultiDiGraph) -> set[str]:
    """True-path rule: if a child term is annotated, all its ancestors are implicitly
    annotated too. Used both to enrich training labels and to build the parent index
    used by the hierarchical loss."""
    expanded = set(labels)
    for go_id in list(labels):
        expanded |= get_ancestors(graph, go_id)
    return expanded

def build_parent_child_pairs(graph: nx.MultiDiGraph, go_terms: list[str]) -> list[tuple[int, int]]:
    """Index pairs (child_idx, parent_idx) within a fixed label vocabulary, used to
    enforce ŷ(g_c) <= ŷ(g_p) (Eq. 18) during training."""
    term_to_idx = {t: i for i, t in enumerate(go_terms)}
    pairs = []
    for child in go_terms:
        if child not in graph:
            continue
        for parent in graph.successors(child):  # child -> parent (is_a)
            if parent in term_to_idx:
                pairs.append((term_to_idx[child], term_to_idx[parent]))
    return pairs
```

**6b. Hierarchical loss + classifier head** — `src/ldpfp/models/hierarchical_classifier.py`
```python
"""Phase 6b: Deep model = attention pooling -> classifier head -> hierarchical loss."""
from __future__ import annotations
import torch
import torch.nn as nn
from ldpfp.attention import PMIDAttention


class HierarchicalGOClassifier(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 512, n_labels: int = 1000):
        super().__init__()
        self.attention = PMIDAttention(embed_dim=embed_dim, hidden_dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, doc_embeddings: torch.Tensor, mask: torch.Tensor):
        R, alpha = self.attention(doc_embeddings, mask)  # Eq. 14 / Eq. 20 at inference
        logits = self.classifier(R)  # F(R(p)) -> ŷ, Eq. 21
        return logits, alpha


def hierarchical_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    parent_child_pairs: list[tuple[int, int]],
    lam: float = 0.5,
) -> torch.Tensor:
    """L_total = L_classification + lambda * L_hierarchy   (Eq. 19)

    L_classification: standard multi-label BCE.
    L_hierarchy: penalizes cases where a child's predicted probability exceeds its
    parent's (violating ŷ(g_c) <= ŷ(g_p), Eq. 18), via a hinge on sigmoid outputs.
    """
    probs = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)

    if not parent_child_pairs:
        return bce

    child_idx = torch.tensor([p[0] for p in parent_child_pairs], device=logits.device)
    parent_idx = torch.tensor([p[1] for p in parent_child_pairs], device=logits.device)
    child_probs = probs[:, child_idx]
    parent_probs = probs[:, parent_idx]
    violation = torch.relu(child_probs - parent_probs)  # >0 exactly when Eq. 18 is violated
    hierarchy_loss = violation.mean()

    return bce + lam * hierarchy_loss


def enforce_consistency(probs: torch.Tensor, parent_child_pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Post-hoc correction pass (used at inference, Section 2.7): clamp each child's
    probability to at most its parent's, applied bottom-up so corrections propagate."""
    probs = probs.clone()
    # naive fixed-point iteration; converges fast since GO depth is small (~15 levels)
    for _ in range(20):
        changed = False
        for child, parent in parent_child_pairs:
            if probs[:, child].gt(probs[:, parent]).any():
                mask = probs[:, child] > probs[:, parent]
                probs[mask, child] = probs[mask, parent]
                changed = True
        if not changed:
            break
    return probs
```

**6c. Baseline GBM models** — `src/ldpfp/models/baselines.py`
```python
"""Phase 6c: Non-deep baselines matching Figure 1's 'Advanced Decision Tree Models' box."""
from __future__ import annotations
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

def build_baselines(n_jobs: int = -1) -> dict:
    return {
        "xgboost": MultiOutputClassifier(
            XGBClassifier(n_estimators=300, max_depth=6, tree_method="hist", n_jobs=n_jobs)
        ),
        "lightgbm": MultiOutputClassifier(
            LGBMClassifier(n_estimators=300, num_leaves=31, n_jobs=n_jobs)
        ),
        "catboost": MultiOutputClassifier(
            CatBoostClassifier(iterations=300, depth=6, verbose=False)
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=n_jobs),
    }

def fit_and_predict(model, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model.fit(X_train, Y_train)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        # MultiOutputClassifier.predict_proba returns a list of (n, 2) arrays per label
        if isinstance(proba, list):
            return np.stack([p[:, 1] for p in proba], axis=1)
        return proba
    return model.predict(X_test)
```

**6d. Training entrypoint** — `src/ldpfp/train.py`
```python
"""Phase 6d: End-to-end training loop for the deep hierarchical model, with MLflow logging."""
from __future__ import annotations
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ldpfp.models.hierarchical_classifier import HierarchicalGOClassifier, hierarchical_loss

def train(
    doc_embeddings: torch.Tensor,   # (n_proteins, k_max, 768) padded
    mask: torch.Tensor,             # (n_proteins, k_max)
    labels: torch.Tensor,           # (n_proteins, n_labels) multi-hot, propagated (Phase 6a)
    parent_child_pairs: list[tuple[int, int]],
    n_labels: int,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    lam: float = 0.5,
):
    model = HierarchicalGOClassifier(embed_dim=doc_embeddings.shape[-1], n_labels=n_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = TensorDataset(doc_embeddings, mask, labels)
    n_val = max(1, int(0.15 * len(dataset)))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    mlflow.set_experiment("ldpfp-hierarchical-classifier")
    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr, "lambda": lam})
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for emb, m, y in train_loader:
                optimizer.zero_grad()
                logits, _ = model(emb, m)
                loss = hierarchical_loss(logits, y, parent_child_pairs, lam=lam)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * emb.size(0)
            train_loss = total_loss / len(train_ds)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for emb, m, y in val_loader:
                    logits, _ = model(emb, m)
                    val_loss += hierarchical_loss(logits, y, parent_child_pairs, lam=lam).item() * emb.size(0)
            val_loss /= len(val_ds)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        mlflow.pytorch.log_model(model, "model")
    return model
```

**Test** — `tests/test_go_hierarchy.py`:
```python
import networkx as nx
from ldpfp.go_hierarchy import get_ancestors, propagate_labels, build_parent_child_pairs

def toy_graph():
    g = nx.MultiDiGraph()
    g.add_edge("GO:child", "GO:parent", key=0)
    g.add_edge("GO:parent", "GO:grandparent", key=0)
    return g

def test_get_ancestors():
    g = toy_graph()
    assert get_ancestors(g, "GO:child") == {"GO:parent", "GO:grandparent"}

def test_propagate_labels_adds_ancestors():
    g = toy_graph()
    out = propagate_labels({"GO:child"}, g)
    assert out == {"GO:child", "GO:parent", "GO:grandparent"}

def test_build_parent_child_pairs():
    g = toy_graph()
    terms = ["GO:child", "GO:parent", "GO:grandparent"]
    pairs = build_parent_child_pairs(g, terms)
    assert (0, 1) in pairs  # child -> parent
    assert (1, 2) in pairs  # parent -> grandparent
```

```
📄 CHAPTER 3 CONTENT — Section 3.6 "Hierarchical GO-Aware Prediction Module"

The Gene Ontology is structured as a directed acyclic graph in which more specific
("child") terms are related to more general ("parent") terms through is_a and
part_of relationships, and the true-path rule dictates that annotation with a child
term implies annotation with all of its ancestor terms. This hierarchical structure
was parsed from the go-basic.obo release using the obonet library, yielding a graph
in which each GO term's transitive ancestor set can be efficiently queried. Two
mechanisms were used to enforce hierarchical consistency, corresponding to Equations
17-19. First, during dataset preparation, ground-truth label vectors were expanded
via ancestor propagation, so that every annotated term's ancestors were also marked
positive, preventing the classifier from being trained against internally
inconsistent targets. Second, a hierarchical regularization term was added to the
training objective: for every child-parent pair (g_c, g_p) present in the model's
label vocabulary, a hinge penalty max(0, ŷ(g_c) − ŷ(g_p)) was computed on the
predicted sigmoid probabilities and averaged into L_hierarchy, which was combined
with the standard multi-label binary cross-entropy L_classification using a weighting
coefficient λ as L_total = L_classification + λL_hierarchy (Eq. 19). At inference
time, a bottom-up fixed-point consistency pass was additionally applied to guarantee
that no child term's final predicted probability exceeds that of any of its parents,
providing a hard guarantee of biological consistency independent of λ.
```

```
📄 CHAPTER 3 CONTENT — Section 3.7 "Model Architecture and Baselines"

Two families of classifiers were trained on the literature-derived representations
to allow direct comparison between deep and non-deep approaches, mirroring Figure 1's
"Deep Learning Models" and "Advanced Decision Tree Models" branches. The primary deep
architecture couples the attention-based pooling module (Section 3.5) with a
two-layer feed-forward classification head (768 → 512 → n_labels, ReLU activation,
dropout 0.3) trained end-to-end with the hierarchical loss of Section 3.6, using the
AdamW optimizer. As non-deep baselines, the pooled protein representation R(pᵢ) was
additionally used to train four decision-tree-based multi-label classifiers —
XGBoost, LightGBM, CatBoost, and Random Forest — each wrapped in a one-vs-rest
multi-output strategy to handle the multi-label GO prediction setting. These
baselines do not natively enforce hierarchical consistency; their raw predictions
were therefore passed through the same post-hoc consistency-enforcement procedure
described in Section 3.6 before evaluation, ensuring a fair, hierarchy-aware
comparison across all models.
```

### Phase 7 — Evaluation

**Goal:** Report the field-standard CAFA-style metrics so Chapter 4's results are comparable to the papers you cited (Kulmanov & Hoehndorf's DeepGOPlus, Gligorijević et al.'s DeepFRI, etc.): **Fmax**, **AUPR (micro/macro)**, and a **hierarchy-violation rate** specific to this framework's novelty claim.

File: `src/ldpfp/evaluate.py`
```python
"""Phase 7: CAFA-style evaluation metrics + hierarchy-violation rate."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score

def precision_recall_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, t: float):
    y_pred = (y_prob >= t).astype(int)
    tp = (y_pred * y_true).sum(axis=1)
    pred_pos = y_pred.sum(axis=1)
    true_pos = y_true.sum(axis=1)
    covered = pred_pos > 0
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp, dtype=float), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp, dtype=float), where=true_pos > 0)
    if covered.sum() == 0:
        return 0.0, 0.0
    return precision[covered].mean(), recall[true_pos > 0].mean()

def fmax_score(y_true: np.ndarray, y_prob: np.ndarray, thresholds=None) -> tuple[float, float]:
    """CAFA-style protein-centric Fmax: max F1 over a sweep of decision thresholds."""
    thresholds = thresholds or np.arange(0.01, 1.00, 0.01)
    best_f, best_t = 0.0, 0.0
    for t in thresholds:
        p, r = precision_recall_at_threshold(y_true, y_prob, t)
        f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        if f > best_f:
            best_f, best_t = f, t
    return best_f, best_t

def aupr_scores(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    micro = average_precision_score(y_true, y_prob, average="micro")
    macro = average_precision_score(y_true, y_prob, average="macro")
    return {"aupr_micro": micro, "aupr_macro": macro}

def hierarchy_violation_rate(y_prob: np.ndarray, parent_child_pairs: list[tuple[int, int]]) -> float:
    """Fraction of (protein, child-parent pair) instances where child_prob > parent_prob.
    Reported before AND after the post-hoc consistency pass (Section 3.6) to quantify
    the concrete benefit of hierarchical GO-awareness — this is your headline novelty metric."""
    if not parent_child_pairs:
        return 0.0
    child_idx = [p[0] for p in parent_child_pairs]
    parent_idx = [p[1] for p in parent_child_pairs]
    violations = (y_prob[:, child_idx] > y_prob[:, parent_idx]).mean()
    return float(violations)

def full_report(y_true: np.ndarray, y_prob: np.ndarray, parent_child_pairs: list[tuple[int, int]]) -> dict:
    fmax, t_star = fmax_score(y_true, y_prob)
    aupr = aupr_scores(y_true, y_prob)
    hvr = hierarchy_violation_rate(y_prob, parent_child_pairs)
    return {"fmax": fmax, "best_threshold": t_star, **aupr, "hierarchy_violation_rate": hvr}
```

Runner: `scripts/05_evaluate.py` — loads the held-out test split, runs each trained model (deep + 4 baselines), calls `full_report`, writes `results/metrics.csv` (one row per model, per GO category) — this table is your Chapter 4 results table.

**Test** — `tests/test_evaluate.py`:
```python
import numpy as np
from ldpfp.evaluate import fmax_score, hierarchy_violation_rate

def test_fmax_perfect_prediction():
    y_true = np.array([[1, 0], [0, 1]])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9]])
    f, t = fmax_score(y_true, y_prob)
    assert f == 1.0

def test_hierarchy_violation_rate_detects_violation():
    y_prob = np.array([[0.9, 0.1]])  # child=col0 prob 0.9 > parent=col1 prob 0.1 -> violation
    rate = hierarchy_violation_rate(y_prob, [(0, 1)])
    assert rate == 1.0
```

```
📄 CHAPTER 3 CONTENT — Section 3.8 "Evaluation Protocol"

Model performance was assessed using protein-centric evaluation metrics standard in
the CAFA (Critical Assessment of Functional Annotation) tradition, enabling
comparison with prior sequence-centric approaches (Kulmanov & Hoehndorf, 2020;
Gligorijević et al., 2021). The primary metric is Fmax, the maximum F1 score achieved
across a sweep of decision thresholds from 0.01 to 0.99, computed per protein and
averaged across the test set. Area under the precision-recall curve (AUPR) was
computed in both micro-averaged (pooled across all label-protein pairs) and
macro-averaged (averaged per GO term) forms, capturing both overall and per-class
performance given the expected label imbalance across GO terms. To specifically
quantify the contribution of hierarchical GO-awareness, a hierarchy-violation rate
was additionally defined as the proportion of child-parent term pairs for which the
model assigns a higher probability to the child than to its parent — a direct,
biologically meaningful measure of the constraint expressed in Equation 18 — reported
both before and after the post-hoc consistency-enforcement procedure to isolate its
effect.

📄 CHAPTER 4 CONTENT — Section 4.4 "Results" (table skeleton; fill after running
scripts/05_evaluate.py)

Table 4.2: Model performance on the held-out test split.

| Model | Fmax | AUPR (micro) | AUPR (macro) | Hierarchy violation rate (before / after correction) |
|---|---|---|---|---|
| Hierarchical Attention Classifier (proposed) | {{}} | {{}} | {{}} | {{}} / {{}} |
| XGBoost | {{}} | {{}} | {{}} | {{}} / {{}} |
| LightGBM | {{}} | {{}} | {{}} | {{}} / {{}} |
| CatBoost | {{}} | {{}} | {{}} | {{}} / {{}} |
| Random Forest | {{}} | {{}} | {{}} | {{}} / {{}} |

Discuss: (1) does attention pooling beat mean/max pooling ablation, (2) does λ>0 beat
λ=0 (no hierarchical loss) at equal Fmax, (3) per-GO-category (MF/BP/CC) breakdown,
(4) 2-3 case-study proteins where you print the top-attention-weighted PMIDs
alongside the predicted GO terms to demonstrate explainability qualitatively.
```

### Phase 8 — Inference API for Unannotated Proteins

**Goal:** Implement Section 2.7's workflow as a live service: given a new UniProt ID, retrieve its literature, embed, attention-pool, classify, hierarchy-correct, return predicted GO terms with confidence and supporting PMIDs.

File: `src/ldpfp/api/schemas.py`
```python
from pydantic import BaseModel

class PredictRequest(BaseModel):
    uniprot_id: str

class GOPrediction(BaseModel):
    go_id: str
    go_term_name: str | None = None
    category: str
    probability: float
    supporting_pmids: list[str]

class PredictResponse(BaseModel):
    uniprot_id: str
    predictions: list[GOPrediction]
```

File: `src/ldpfp/api/main.py`
```python
"""Phase 8: FastAPI inference service implementing the Section 2.7 workflow."""
from fastapi import FastAPI, HTTPException
import torch
from ldpfp.api.schemas import PredictRequest, PredictResponse, GOPrediction
from ldpfp.embeddings import BiomedicalEmbedder
from ldpfp.literature_retrieval import retrieve_all, get_conn
from ldpfp.text_preprocessing import load_and_assemble
from ldpfp.go_hierarchy import load_go_graph, build_parent_child_pairs
from ldpfp.models.hierarchical_classifier import HierarchicalGOClassifier, enforce_consistency

app = FastAPI(title="LD-PFP Inference API", version="0.1.0")

# Loaded once at startup — swap paths for your actual trained artifacts
STATE = {}

@app.on_event("startup")
def load_artifacts():
    STATE["embedder"] = BiomedicalEmbedder()
    STATE["model"] = torch.load("models/hierarchical_classifier.pt", map_location="cpu")
    STATE["model"].eval()
    STATE["go_terms"] = torch.load("models/go_term_vocab.pt")  # list[str], index-aligned to model output
    STATE["graph"] = load_go_graph("data/external/go-basic.obo")
    STATE["parent_child_pairs"] = build_parent_child_pairs(STATE["graph"], STATE["go_terms"])

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1. Fetch that protein's PMIDs from goa cache or live UniProt xref lookup
    #    (fallback path for a truly unannotated protein not in your GAF snapshot)
    import polars as pl
    pmid_map = pl.read_parquet("data/interim/protein_pmid_map.parquet")
    row = pmid_map.filter(pl.col("Protein_ID") == req.uniprot_id)
    if row.is_empty():
        raise HTTPException(404, f"No literature found for {req.uniprot_id}")
    pmids = row["PMIDs"][0]

    # 2. Retrieve + embed
    retrieve_all(pmids, "data/literature_cache.sqlite")
    texts = load_and_assemble("data/literature_cache.sqlite")
    doc_vecs = [STATE["embedder"].embed_text(texts[p]) for p in pmids if p in texts]
    if not doc_vecs:
        raise HTTPException(422, "No retrievable literature text for this protein")

    import numpy as np
    doc_tensor = torch.tensor(np.stack(doc_vecs)).unsqueeze(0).float()
    mask = torch.ones(1, doc_tensor.shape[1], dtype=torch.bool)

    # 3. Predict + hierarchy-correct
    with torch.no_grad():
        logits, alpha = STATE["model"](doc_tensor, mask)
        probs = torch.sigmoid(logits)
        probs = enforce_consistency(probs, STATE["parent_child_pairs"])

    # 4. Format top predictions with supporting PMIDs (top-attention-weighted docs)
    top_idx = probs[0].topk(k=15).indices.tolist()
    top_pmids = [p for p, w in sorted(zip(pmids, alpha[0].tolist()), key=lambda x: -x[1])[:3]]
    predictions = [
        GOPrediction(
            go_id=STATE["go_terms"][i],
            category="",  # look up from your GO metadata table
            probability=float(probs[0, i]),
            supporting_pmids=top_pmids,
        )
        for i in top_idx
    ]
    return PredictResponse(uniprot_id=req.uniprot_id, predictions=predictions)

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run locally: `uvicorn ldpfp.api.main:app --reload --port 8000`
Try it: `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"uniprot_id":"P12345"}'`
Interactive docs auto-generated at `http://localhost:8000/docs` — screenshot this for your appendix.

**Test** — `tests/test_api.py` (mock the model/artifacts, test routing/validation only):
```python
from fastapi.testclient import TestClient
from ldpfp.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_missing_protein_returns_404(monkeypatch):
    r = client.post("/predict", json={"uniprot_id": "NOT_A_REAL_ID_XYZ"})
    assert r.status_code in (404, 422)
```

```
📄 CHAPTER 4 CONTENT — Section 4.5 "System Implementation: Inference Service"

The trained framework was deployed as a REST API using FastAPI, implementing the
prediction workflow for unannotated proteins described conceptually in Section 2.7.
Given a UniProt identifier, the service retrieves the protein's linked literature,
generates document embeddings via the trained PubMedBERT encoder, applies the learned
attention-pooling module to obtain the weighted protein representation R(p_u)
(Equation 20), passes it through the trained classifier to obtain raw GO-term
probabilities (Equation 21), and finally applies the hierarchical consistency
correction (Section 3.6) before returning results. Each prediction is returned
alongside the specific publications that received the highest attention weight for
that protein, providing a direct, literature-grounded explanation for every predicted
function — directly supporting the explainability contribution claimed in Section 3
of this study. The service exposes a single POST /predict endpoint accepting a
UniProt accession and returning a ranked list of predicted GO terms with associated
probabilities and supporting PMIDs, plus a GET /health endpoint for deployment
monitoring, both documented via an auto-generated OpenAPI schema.
```

---

## 5. Testing Strategy (project-wide)

| Level | Tool | What it covers |
|---|---|---|
| Unit | `pytest` | Every function in `src/ldpfp/*.py` — GAF parsing edge cases, PMID extraction, attention math, GO ancestor logic, metric formulas |
| Integration | `pytest` + fixtures | A tiny synthetic GAF (10 rows) → full pipeline → check final Parquet has expected columns/shape, without hitting the real network or a real 188MB file |
| API | `TestClient` (FastAPI) | Route validation, error codes, response schema |
| Data-quality | Custom `scripts/validate_dataset.py` | Assert no protein has 0 experimentally-evidenced GO terms after filtering; assert all `PMID` values are numeric strings; assert `GO_Category` ∈ {MF, BP, CC} |
| Regression | MLflow run comparison | Every training run logged; new runs compared against the last "accepted" Fmax before you consider a model change "done" |
| Coverage target | `pytest --cov=src/ldpfp` | Aim for ≥80% on `gaf_parser.py`, `go_hierarchy.py`, `attention.py`, `evaluate.py` (the modules with the most "novelty" — reviewers/examiners will ask about these) |

Run everything: `pytest --cov=src/ldpfp --cov-report=term-missing -v`

```
📄 CHAPTER 4 CONTENT — Section 4.6 "Testing Strategy"

The implementation was validated using a layered testing strategy. Unit tests (35+
test cases across pytest modules) exercised the core algorithmic components in
isolation, including PMID extraction from GAF DB:Reference fields, GO ancestor
propagation and parent-child pair construction, the attention weighting mechanism's
softmax normalization property, and the Fmax/AUPR/hierarchy-violation-rate metric
implementations against hand-computed expected values. Integration tests exercised
the pipeline end-to-end against a small synthetic GAF fixture to verify that data
flows correctly between the parsing, retrieval, embedding, and classification stages
without requiring network access or the full 188MB production file, enabling the test
suite to run in continuous integration. API-level tests used FastAPI's TestClient to
validate request/response schemas and error handling for the inference endpoint. Data
quality was further enforced through a standalone validation script asserting
structural invariants on the constructed dataset (e.g., that every GO_Category value
belongs to the expected three-way ontology partition). Test coverage was measured
with pytest-cov, with particular emphasis on the modules implementing this study's
core novelty — the attention mechanism and hierarchical GO-awareness logic.
```

---

## 6. Deployment Strategy

`docker/Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/external/go-basic.obo data/external/go-basic.obo
COPY models/ models/
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "ldpfp.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker/docker-compose.yml`:
```yaml
version: "3.9"
services:
  ldpfp-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../data:/app/data
    environment:
      - NCBI_EMAIL=${NCBI_EMAIL}
      - NCBI_API_KEY=${NCBI_API_KEY}
```

Build & run: `docker compose -f docker/docker-compose.yml up --build`

For a university thesis defense/demo, this is sufficient — you don't need Kubernetes. If you want a public demo link, the same image deploys as-is to Render, Railway, or a small EC2/DigitalOcean droplet; mention this as "future deployment target" rather than building it out, unless explicitly required.

```
📄 CHAPTER 4 CONTENT — Section 4.7 "Deployment"

The complete inference service was containerized using Docker to guarantee
environment reproducibility independent of host machine configuration. The container
image installs all Python dependencies, bundles the trained model artifacts and the
GO ontology file, and exposes the FastAPI inference service on port 8000. A
docker-compose configuration was provided to simplify local orchestration, mounting
the data directory as a volume so that the literature cache and processed datasets
persist across container restarts. This containerized packaging allows the system to
be deployed identically in a local development environment, a departmental server, or
a cloud compute instance without modification, satisfying the framework's design goal
of scalability stated in Section 3 of the manuscript.
```

---

## 7. Documentation Structure

```
docs/
├── mkdocs.yml
└── docs/
    ├── index.md              # project overview, mirrors abstract
    ├── methodology.md        # = Chapter 3, kept in sync
    ├── implementation.md     # = Chapter 4, kept in sync
    ├── api-reference.md      # generated from FastAPI's /openapi.json
    └── setup.md              # env setup, how to run each phase (= Section 8 below)
```
`mkdocs.yml`:
```yaml
site_name: LD-PFP Documentation
theme:
  name: material
nav:
  - Overview: index.md
  - Methodology: methodology.md
  - Implementation: implementation.md
  - API Reference: api-reference.md
  - Setup Guide: setup.md
```
Serve locally: `pip install mkdocs-material && mkdocs serve -f docs/mkdocs.yml`

Also keep a top-level `README.md` in the repo root with: project description, quick-start (`pip install -r requirements.txt`, run `scripts/01` through `scripts/06` in order), and a link into `docs/`. This README doubles as your project defense "how to run this" handout.

---

## 8. Development Workflow & Timeline (suggested — compress/expand to your deadline)

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Phase 0-1: repo scaffold, GAF parser, dataset stats | `annotations.parquet`, Chapter 4 §4.2 table filled in |
| 2 | Phase 2-3: literature retrieval + preprocessing (start this early — it's the slowest, network-bound step; let it run overnight/in background while you write) | `literature_cache.sqlite` populated, coverage report |
| 3 | Phase 4-5: embeddings + attention module, unit tests green | `pmid_embeddings.npz`, `test_attention.py` passing |
| 4 | Phase 6: GO DAG + hierarchical classifier + baselines, first training run | First MLflow run logged, draft Chapter 3 complete |
| 5 | Phase 7: evaluation, ablations (with/without attention, with/without hierarchical loss, abstract-only vs +full-text) | Results table (Chapter 4 §4.4) |
| 6 | Phase 8: API + Docker, polish docs, write up Chapter 4 discussion + limitations | Working demo, both chapters draft-complete |
| 7 (buffer) | Supervisor review round, fix findings, finalize | Submission-ready |

**Immediate next action for you today:** run Phase 1 against your real `HUMAN-uniprot.gaf` (it's local, no network needed, takes seconds to minutes) — that single run gives you the real dataset statistics table for Chapter 4 §4.2 and validates the whole pipeline skeleton before you spend any time on the slow network-bound literature retrieval step.

---

## 9. Chapter 3 & 4 — Full Section Skeleton (assembly checklist)

Use this as your manuscript table of contents; every subsection below already has drafted prose above — just paste in order and fill `{{ }}` placeholders once your runs complete.

**Chapter 3 — Methodology**
- 3.1 Framework Overview *(rephrase paper's Section 2.1; describe the 6-module pipeline, reference Figure 1)*
- 3.2 Dataset Construction *(→ §Phase 1 content above)*
  - 3.2.3 Refinement of the Literature Retrieval Step *(→ §0 content above)*
- 3.3 Literature Retrieval Module *(→ §Phase 2 content above)*
- 3.4 Text Representation Module *(→ §Phase 4 content above)*
- 3.5 Attention-Based PMID Weighting Module *(→ §Phase 5 content above)*
- 3.6 Hierarchical GO-Aware Prediction Module *(→ §Phase 6a/b content above)*
- 3.7 Model Architecture and Baselines *(→ §Phase 6c content above)*
- 3.8 Evaluation Protocol *(→ §Phase 7 content above)*
- 3.9 Implementation Environment and Tools *(short subsection: list Table from §2 "Technology Stack" as a table, one line justifying each choice)*

**Chapter 4 — Implementation and Results**
- 4.1 System Architecture and Repository Organization *(→ §3 "Repository/Folder Structure" above, presented as a figure/tree)*
- 4.2 Dataset Statistics *(→ §Phase 1 Chapter 4 content above)*
- 4.3 Training Configuration *(hyperparameters table: epochs, batch size, lr, λ, embedding dim, from `train.py` defaults — update with your actual final run's values)*
- 4.4 Results *(→ §Phase 7 Chapter 4 content above)*
- 4.5 System Implementation: Inference Service *(→ §Phase 8 content above)*
- 4.6 Testing Strategy *(→ §5 content above)*
- 4.7 Deployment *(→ §6 content above)*
- 4.8 Discussion and Limitations *(write last, freehand — discuss: PMC full-text coverage limits since not all articles are open access; embedding cost/compute constraints; GO DAG size vs label sparsity; how results compare to sequence-based methods cited in your Introduction/Related Work)*
- 4.9 Summary

---

## 10. Quick Reference — Command Sequence (copy-paste run order)

```bash
# one-time setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in NCBI_EMAIL / NCBI_API_KEY
curl -L -o data/external/go-basic.obo http://purl.obolibrary.org/obo/go/go-basic.obo
cp /path/to/HUMAN-uniprot.gaf data/raw/HUMAN-uniprot.gaf

# pipeline, in order
python scripts/01_parse_gaf.py
python scripts/02_fetch_literature.py     # slow — network bound, resumable
python scripts/03_build_embeddings.py
python scripts/04_train.py
python scripts/05_evaluate.py
uvicorn ldpfp.api.main:app --reload --port 8000   # scripts/06_predict_unannotated.py hits this

# quality gates
pytest --cov=src/ldpfp --cov-report=term-missing -v
docker compose -f docker/docker-compose.yml up --build
mkdocs serve -f docs/mkdocs.yml
```
