"""Build the FAISS index locally (dev) from data/synthetic/documents.jsonl.

For the FULL corpus, indexing runs on Modal (modal_app.py::index_corpus) — this
script is the local/dev path over the small documents.jsonl produced by
`build_corpus.py --dev-docs N`.
"""
import _bootstrap  # noqa: F401

from temporalguard.config import load_yaml
from temporalguard.retrieval.build import build_index_from_docs
from temporalguard.schemas import Document
from temporalguard.utils.json_utils import iter_jsonl


def main():
    retr = load_yaml("configs/retrieval.yaml")
    app = load_yaml("configs/app.yaml")
    docs_path = app["paths"]["dev_documents"]
    index_dir = retr["paths"]["index_dir"]
    chunking = retr.get("chunking", {})

    docs = (Document.from_dict(r) for r in iter_jsonl(docs_path))
    index = build_index_from_docs(
        docs,
        index_dir=index_dir,
        embedding_model=retr["embedding_model"],
        chunk_size=chunking.get("chunk_size", 900),
        chunk_overlap=chunking.get("chunk_overlap", 120),
    )
    print(f"Local dev index ready: {index.count} chunks -> {index_dir}")


if __name__ == "__main__":
    main()
