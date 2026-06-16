"""Modal app for TemporalGuard heavy compute (full-corpus FAISS indexing on GPU).

The corpus is loaded **directly from HuggingFace** (onyx-dot-app/EnterpriseRAG-Bench)
on Modal — no local files, no uploads, no tarballs. The HF cache lives on the
Modal Volume so re-runs don't re-download. Embedding the full ~512k-doc corpus
runs on an L4 GPU; the FAISS index is written back to the Volume.

One-time setup (run locally, from the repo root):

    # 1. Auth (already done if ~/.modal.toml exists)
    modal token new

    # 2. Secret with the keys the run needs (DeepSeek for baseline; HF token for the dataset)
    modal secret create temporalguard-secrets \
        DEEPSEEK_API_KEY=<key> HF_TOKEN=<token>

Build the full index on GPU (downloads the dataset from HF on first run):

    modal run modal_app.py::index_corpus

Volume layout:
    /data/hf_cache/...                       <- HuggingFace dataset cache (download once)
    /data/index/full/{index.faiss, chunks.jsonl, meta.json}
"""
from __future__ import annotations

import modal

app = modal.App("temporalguard")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "faiss-cpu",
        "sentence-transformers",
        "torch",
        "numpy",
        "pyyaml",
        "openai",
        "datasets",
        "huggingface-hub",
        "langchain-text-splitters",
    )
    .add_local_dir("src", remote_path="/root/src")
)

volume = modal.Volume.from_name("temporalguard-data", create_if_missing=True)
secrets = [modal.Secret.from_name("temporalguard-secrets")]

VOLUME_MOUNT = "/data"
HF_CACHE = "/data/hf_cache"          # HuggingFace dataset cache (persisted on the Volume)
INDEX_DIR = "/data/index/full"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SRC_PATH = "/root/src"


@app.function(image=image, gpu="L4", volumes={VOLUME_MOUNT: volume}, secrets=secrets, timeout=6 * 60 * 60)
def index_corpus(chunk_size: int = 900, chunk_overlap: int = 120):
    """Stream docs from HF -> chunk -> embed (GPU) -> FAISS -> save to Volume."""
    import os
    import sys
    import time

    sys.path.insert(0, SRC_PATH)
    from temporalguard.corpus.hf_loader import iter_hf_docs
    from temporalguard.retrieval.build import build_index_from_docs

    token = os.environ.get("HF_TOKEN")
    t0 = time.time()

    def log(msg: str):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    log("starting full-corpus index build (GPU=L4), loading dataset from HuggingFace")
    docs = iter_hf_docs(cache_dir=HF_CACHE, token=token)
    index = build_index_from_docs(
        docs,
        index_dir=INDEX_DIR,
        embedding_model=EMBED_MODEL,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device="cuda",
        embed_batch=8192,
        log=log,
    )
    volume.commit()
    log(f"DONE: {index.count} chunks indexed -> {INDEX_DIR}")
    return {"chunks": index.count, "elapsed_s": round(time.time() - t0, 1)}


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, secrets=secrets, timeout=60 * 60)
def run_baseline(questions: list, top_k: int = 5, mock: bool = False):
    """Run naive baseline over `questions` (list of dicts) against the full index on the Volume."""
    import sys

    sys.path.insert(0, SRC_PATH)
    from temporalguard.eval.baseline import answer_baseline
    from temporalguard.llm.cache import LLMCache
    from temporalguard.llm.provider import LLMProvider
    from temporalguard.retrieval.retriever import Retriever

    prompts = {
        "baseline_answer": {
            "system": "You are a helpful assistant answering questions using the provided context.",
            "user": "Answer the question using the context.\nIf possible, cite the source document IDs.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
        }
    }
    retriever = Retriever(INDEX_DIR, EMBED_MODEL, top_k=top_k, device="cpu")
    provider = LLMProvider(
        {"provider": "deepseek", "base_url": "https://api.deepseek.com",
         "api_key_env": "DEEPSEEK_API_KEY", "model": "deepseek-v4-flash", "mock": mock},
        cache=LLMCache("/data/cache/llm_calls"),
    )

    rows = []
    for q in questions:
        res = answer_baseline(q["question"], retriever, provider, prompts, top_k=top_k)
        res.update(question_id=q["question_id"], category=q.get("category"), expected_decision=q.get("expected_decision"))
        rows.append(res)
    volume.commit()
    return rows


@app.local_entrypoint()
def main():
    """`modal run modal_app.py` -> build the full index on GPU."""
    result = index_corpus.remote()
    print("index_corpus result:", result)
