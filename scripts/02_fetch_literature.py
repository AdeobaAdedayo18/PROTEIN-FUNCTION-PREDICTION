from dotenv import load_dotenv

load_dotenv()

import polars as pl

from ldpfp.literature_retrieval import retrieve_all


pmid_map = pl.read_parquet(
    "data/interim/protein_pmid_map.parquet"
)

all_pmids = sorted(
    set(
        pmid_map
        .explode(
            "PMIDs",
            empty_as_null=True,
        )["PMIDs"]
        .drop_nulls()
        .to_list()
    )
)

print(
    f"{len(all_pmids):,} distinct PMIDs to retrieve"
)

retrieve_all(
    all_pmids,
    "data/literature_cache.sqlite",
)