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
