from unittest.mock import patch, MagicMock
from ldpfp.literature_retrieval import fetch_abstract


@patch("ldpfp.literature_retrieval.Entrez.efetch")
@patch("ldpfp.literature_retrieval.Entrez.read")
def test_fetch_abstract_parses_title_and_abstract(mock_read, mock_efetch):
    mock_efetch.return_value = MagicMock()

    mock_read.return_value = {
        "PubmedArticle": [{
            "MedlineCitation": {
                "PMID": "123",
                "Article": {
                    "ArticleTitle": "A protein of interest",
                    "Abstract": {
                        "AbstractText": [
                            "Background text.",
                            "Results text.",
                        ]
                    },
                }
            }
        }]
    }

    out = fetch_abstract("123")

    assert out["title"] == "A protein of interest"
    assert "Background text." in out["abstract"]