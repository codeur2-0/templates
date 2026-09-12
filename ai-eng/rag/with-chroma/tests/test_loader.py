from docs_rag.loader import MarkdownDocumentLoader


def test_documents_are_validated():
    frame = MarkdownDocumentLoader("data/raw").load()
    assert {"getting_started", "security", "billing"} <= set(frame.document_id)
