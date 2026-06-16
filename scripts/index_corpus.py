"""Build a SMALL local FAISS index from HuggingFace (dev smoke test).

The full-corpus index is built on Modal (modal_app.py::index_corpus). This local
path streams just the first N HF documents so you can exercise retrieval/baseline
end-to-end on a laptop without the GPU run.

Usage:
  python scripts/index_corpus.py            # first 2000 HF docs
  python scripts/index_corpus.py --limit 500
"""
import argparse
import itertools
import os

import _bootstrap  # noqa: F401

from temporalguard.config import load_dotenv, load_yaml
from temporalguard.corpus.hf_loader import iter_hf_docs
from temporalguard.retrieval.build import build_index_from_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000, help="Number of HF docs to index locally.")
    args = ap.parse_args()

    load_dotenv()
    retr = load_yaml("configs/retrieval.yaml")
    chunking = retr.get("chunking", {})
    token = os.environ.get("HF_TOKEN")

    docs = itertools.islice(iter_hf_docs(token=token), args.limit)
    index = build_index_from_docs(
        docs,
        index_dir=retr["paths"]["index_dir"],
        embedding_model=retr["embedding_model"],
        chunk_size=chunking.get("chunk_size", 900),
        chunk_overlap=chunking.get("chunk_overlap", 120),
    )
    print(f"Local dev index ready: {index.count} chunks -> {retr['paths']['index_dir']}")


if __name__ == "__main__":
    main()
