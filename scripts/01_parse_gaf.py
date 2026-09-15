import polars as pl
from ldpfp.gaf_parser import (
    parse_gaf,
    build_ground_truth,
    aggregate_pmids_per_protein,
)

raw = parse_gaf("data/raw/HUMAN-uniprot.gaf")

gt = build_ground_truth(raw)

gt.write_parquet("data/interim/annotations.parquet")

pmid_map = aggregate_pmids_per_protein(gt)
pmid_map.write_parquet("data/interim/protein_pmid_map.parquet")

# Dataset sanity-check statistics
raw_rows = len(raw)
kept_rows = len(gt)
retention_rate = (kept_rows / raw_rows) * 100

n_mf = (
    gt.filter(pl.col("GO_Category") == "Molecular Function")
    ["GO_Label"]
    .n_unique()
)

n_bp = (
    gt.filter(pl.col("GO_Category") == "Biological Process")
    ["GO_Label"]
    .n_unique()
)

n_cc = (
    gt.filter(pl.col("GO_Category") == "Cellular Component")
    ["GO_Label"]
    .n_unique()
)

n_pmids = (
    gt.filter(pl.col("PMID").is_not_null())
    ["PMID"]
    .n_unique()
)

print(f"Raw annotation rows: {raw_rows:,}")
print(f"Experimental annotation rows retained: {kept_rows:,}")
print(f"Retention rate: {retention_rate:.2f}%")
print(f"Unique proteins: {gt['Protein_ID'].n_unique():,}")
print(f"Unique GO terms: {gt['GO_Label'].n_unique():,}")
print(f"  Molecular Function GO terms: {n_mf:,}")
print(f"  Biological Process GO terms: {n_bp:,}")
print(f"  Cellular Component GO terms: {n_cc:,}")
print(f"Proteins with at least one PMID: {len(pmid_map):,}")
print(f"Total distinct PMIDs referenced: {n_pmids:,}")