from ldpfp.text_preprocessing import (
    normalize_text,
    assemble_document,
    strip_jats_xml,
)


def test_normalize_collapses_whitespace():
    assert normalize_text("a\n\n  b\t c") == "a b c"


def test_assemble_document_joins_available_parts():
    out = assemble_document(
        "Title.",
        "Abstract text.",
        None,
    )

    assert out == "Title. Abstract text."


def test_strip_jats_xml_extracts_body_text():
    xml = """
    <article>
        <front>
            <article-meta>
                <title-group>
                    <article-title>Test title</article-title>
                </title-group>
            </article-meta>
        </front>

        <body>
            <sec>
                <title>Introduction</title>
                <p>This is the introduction.</p>
            </sec>

            <sec>
                <title>Results</title>
                <p>These are the results.</p>
            </sec>
        </body>
    </article>
    """

    out = strip_jats_xml(xml)

    assert "Introduction" in out
    assert "This is the introduction." in out
    assert "Results" in out
    assert "These are the results." in out
    assert "Test title" not in out