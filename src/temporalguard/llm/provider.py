"""LLM provider (OpenAI-compatible) with caching, cost tracking, and a mock mode.

Defaults to DeepSeek (`deepseek-v4-flash`) via its OpenAI-compatible endpoint; the
model / base_url / key env come from configs/app.yaml so swapping providers is a
config change. ``mock=True`` returns an extractive stitch of the context with no
API call — for $0 dry runs and tests.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from temporalguard.llm.cache import LLMCache, cache_key

# Approx USD per 1M tokens (prompt, completion). DeepSeek pricing is approximate.
PRICES = {
    "deepseek-v4-flash": (0.27, 1.10),
    "deepseek-v4-pro": (0.55, 2.19),
    "gpt-4o-mini": (0.15, 0.60),
    "mock": (0.0, 0.0),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens / 1_000_000) * p_in + (completion_tokens / 1_000_000) * p_out


@dataclass
class LLMResult:
    text: str
    cached: bool
    mock: bool
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_estimate: float = 0.0


class LLMProvider:
    def __init__(self, cfg: Dict[str, Any], cache: Optional[LLMCache] = None):
        self.provider = cfg.get("provider", "deepseek")
        self.base_url = cfg.get("base_url", "https://api.deepseek.com")
        self.api_key_env = cfg.get("api_key_env", "DEEPSEEK_API_KEY")
        self.model = cfg.get("model", "deepseek-v4-flash")
        self.max_tokens = int(cfg.get("max_tokens", 400))
        self.temperature = float(cfg.get("temperature", 0.2))
        self.mock = bool(cfg.get("mock", False))
        self.cache = cache
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"{self.api_key_env} not set. Set it in .env or use mock mode.")
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def generate(self, system: str, user: str) -> LLMResult:
        if self.mock:
            return LLMResult(text=self._mock_answer(user), cached=False, mock=True, model="mock")

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        key = cache_key(
            {
                "provider": self.provider,
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )

        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return LLMResult(
                    text=hit["text"],
                    cached=True,
                    mock=False,
                    model=hit.get("model", self.model),
                    prompt_tokens=hit.get("prompt_tokens", 0),
                    completion_tokens=hit.get("completion_tokens", 0),
                    cost_estimate=hit.get("cost_estimate", 0.0),
                )

        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) if usage else 0
        ct = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = estimate_cost(self.model, pt, ct)

        if self.cache is not None:
            self.cache.set(
                key,
                {"text": text, "model": self.model, "prompt_tokens": pt, "completion_tokens": ct, "cost_estimate": cost},
            )

        return LLMResult(
            text=text, cached=False, mock=False, model=self.model,
            prompt_tokens=pt, completion_tokens=ct, cost_estimate=cost,
        )

    @staticmethod
    def _mock_answer(user_prompt: str) -> str:
        """Extractive stand-in: echoes the first context line. Mirrors a naive
        baseline that always produces *something*, even with weak evidence."""
        ctx = user_prompt
        if "Context:" in user_prompt and "Question:" in user_prompt:
            ctx = user_prompt.split("Context:", 1)[1].split("Question:", 1)[0]
        first = next((ln.strip() for ln in ctx.splitlines() if ln.strip()), "")
        return f"[mock baseline] Based on the context: {first[:200]}"
