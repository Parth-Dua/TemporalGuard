"""Structured decision output via DeepSeek JSON mode (v2+).

Instead of free text + regex (baseline), v2 asks the LLM to return a JSON object:

    {"decision": "ANSWER" | "NOT_FOUND" | "CONFLICT_DETECTED",
     "answer": str,
     "used_doc_ids": [str, ...],
     "confidence": float 0..1}

This makes the decision explicit and reliable (no regex guessing) and is the
foundation the reliability layer builds on. Cached by request hash, same as the
free-text provider; `mock=True` returns a deterministic stub at $0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from temporalguard.eval.augment_generate.cache import LLMCache, cache_key
from temporalguard.eval.augment_generate.provider import PRICES, estimate_cost

VALID_DECISIONS = {"ANSWER", "NOT_FOUND", "CONFLICT_DETECTED", "LOW_CONFIDENCE_ABSTAIN"}


@dataclass
class StructuredResult:
    decision: str
    answer: str
    used_doc_ids: List[str]
    confidence: float
    cached: bool
    mock: bool
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_estimate: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


def _coerce(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a parsed JSON object into our decision shape (defensive)."""
    decision = str(obj.get("decision", "")).upper().strip()
    if decision == "ERROR":
        pass  # explicit failure marker — keep it visible, don't coerce
    elif decision not in VALID_DECISIONS:
        # If the model answered but used an unknown label, fall back sensibly.
        decision = "ANSWER" if obj.get("answer") else "NOT_FOUND"
    conf = obj.get("confidence", 0.0)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.0
    used = obj.get("used_doc_ids") or []
    if not isinstance(used, list):
        used = []
    return {
        "decision": decision,
        "answer": str(obj.get("answer", "") or ""),
        "used_doc_ids": [str(d) for d in used],
        "confidence": conf,
    }


class StructuredProvider:
    """OpenAI-compatible structured-output client (DeepSeek JSON mode)."""

    def __init__(self, cfg: Dict[str, Any], cache: Optional[LLMCache] = None):
        self.provider = cfg.get("provider", "deepseek")
        self.base_url = cfg.get("base_url", "https://api.deepseek.com")
        self.api_key_env = cfg.get("api_key_env", "DEEPSEEK_API_KEY")
        self.model = cfg.get("model", "deepseek-v4-flash")
        self.max_tokens = int(cfg.get("max_tokens", 600))
        self.temperature = float(cfg.get("temperature", 0.1))
        self.mock = bool(cfg.get("mock", False))
        self.cache = cache
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import os
            from openai import OpenAI

            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} not set. Set it in .env or use mock mode.")
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def generate(self, system: str, user: str) -> StructuredResult:
        if self.mock:
            stub = {"decision": "ANSWER", "answer": "[mock v2 structured answer]", "used_doc_ids": [], "confidence": 0.5}
            return StructuredResult(**stub, cached=False, mock=True, model="mock", raw=stub)

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        key = cache_key({
            "provider": self.provider, "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens, "mode": "json",
        })
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                c = _coerce(hit["parsed"])
                return StructuredResult(**c, cached=True, mock=False, model=hit.get("model", self.model),
                                        prompt_tokens=hit.get("prompt_tokens", 0),
                                        completion_tokens=hit.get("completion_tokens", 0),
                                        cost_estimate=hit.get("cost_estimate", 0.0), raw=hit["parsed"])

        client = self._ensure_client()
        pt = ct = 0
        parsed = None
        # Retry with a larger token budget if the model truncates (finish_reason=length
        # -> empty/partial content). Silently degrading to NOT_FOUND would corrupt metrics.
        for attempt, max_tok in enumerate([self.max_tokens, max(self.max_tokens * 2, 2000)]):
            resp = client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature, max_tokens=max_tok,
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            usage = getattr(resp, "usage", None)
            pt += getattr(usage, "prompt_tokens", 0) if usage else 0
            ct += getattr(usage, "completion_tokens", 0) if usage else 0
            if text:
                try:
                    parsed = json.loads(text)
                    break
                except json.JSONDecodeError:
                    pass
            # empty or unparseable (often finish_reason == "length") -> retry once larger
            if choice.finish_reason != "length" and text:
                break
        if parsed is None:
            # genuine failure after retry; mark explicitly so it's visible, not a fake NOT_FOUND
            parsed = {"decision": "ERROR", "answer": "", "used_doc_ids": [], "confidence": 0.0}
        cost = estimate_cost(self.model, pt, ct)

        if self.cache is not None:
            self.cache.set(key, {"parsed": parsed, "model": self.model,
                                 "prompt_tokens": pt, "completion_tokens": ct, "cost_estimate": cost})

        c = _coerce(parsed)
        return StructuredResult(**c, cached=False, mock=False, model=self.model,
                                prompt_tokens=pt, completion_tokens=ct, cost_estimate=cost, raw=parsed)
