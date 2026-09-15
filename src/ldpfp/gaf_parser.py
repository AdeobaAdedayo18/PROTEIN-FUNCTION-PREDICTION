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
