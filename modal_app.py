"""Modal app for TemporalGuard heavy compute (full-corpus FAISS indexing on GPU).

Why Modal: embedding the full EnterpriseRAG-Bench (~512k docs, ~630M tokens) is
far too slow on a laptop CPU. We run the embed+index pass on an L4 GPU, reading
the bench corpus from a Modal Volume and writing the FAISS index back to it.

One-time setup (run locally, from the repo root):

    # 1. Auth (already done if ~/.modal.toml exists)
    modal token new

    # 2. Create the secret with the API keys the run needs
    modal secret create temporalguard-secrets \
        DEEPSEEK_API_KEY=<key> HF_TOKEN=<token>

    # 3. Upload the bench corpus into the Volume (once; a few hundred MB)
    modal volume create temporalguard-data
    modal volume put temporalguard-data \
        EnterpriseRAG-Bench/generated_data /bench/generated_data

Then build the full index on GPU:

    modal run modal_app.py::index_corpus

And (optionally) run the baseline on Modal CPU against the full index:

    modal run modal_app.py::run_baseline --questions-jsonl <local path or volume path>

The Volume layout this app uses:
    /data/bench/generated_data/...     <- uploaded bench corpus
    /data/index/full/{index.faiss, chunks.jsonl, meta.json}
    /data/outputs/baseline_outputs.jsonl
"""
from __future__ import annotations

import modal

app = modal.App("temporalguard")

# Source is added to the image so `import temporalguard...` works on Modal.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "faiss-cpu",
        "sentence-transformers",
        "torch",
        "numpy",
        "pyyaml",
        "openai",
        "langchain-text-splitters",
    )
    .add_local_dir("src", remote_path="/root/src")
)

volume = modal.Volume.from_name("temporalguard-data", create_if_missing=True)
secrets = [modal.Secret.from_name("temporalguard-secrets")]

VOLUME_MOUNT = "/data"
BENCH_ROOT = "/data/bench"          # generated_data/ lives directly under here
INDEX_DIR = "/data/index/full"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@app.function(image=image, gpu="L4", volumes={VOLUME_MOUNT: volume}, secrets=secrets, timeout=4 * 60 * 60)
def index_corpus(chunk_size: int = 900, chunk_overlap: int = 120):
    """Stream every bench doc -> chunk -> embed (GPU) -> FAISS -> save to Volume."""
    import sys
    import time

    sys.path.insert(0, "/root/src")
    from temporalguard.corpus.bench_import import iter_all_bench_docs
    from temporalguard.retrieval.build import build_index_from_docs

    t0 = time.time()

    def log(msg: str):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    log("starting full-corpus index build (GPU=L4)")
    docs = iter_all_bench_docs(BENCH_ROOT)
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
    """Run naive baseline over `questions` (list of dicts) against the full index.

    Returns the rows; the caller persists them locally. Reads the index from the
    Volume. `mock=True` makes it $0.
    """
    import sys

    sys.path.insert(0, "/root/src")
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
         "api_key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat", "mock": mock},
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
