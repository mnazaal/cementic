#!/usr/bin/env python
"""Check that a full-size chunk survives the embedding model's context window.

``pipeline.chunk_size`` counts tokens with a tiktoken encoding; the embedding
server's ``llama_cpp.n_ctx`` counts them with the *model's own* tokenizer, and a
task prefix is added on top. A full chunk can therefore overflow -- either
failing to embed, or embedding with its tail silently dropped, which degrades
retrieval with no error anywhere.

This measures rather than assumes, across text where the two tokenizers diverge
most, using two independent signals:

1. **Exact count.** The daemon's ``/extras/tokenize/count`` applies the model's
   own tokenizer, so the budget question is answered directly rather than
   inferred. This is the primary verdict.
2. **Differential probe.** Two variants of the chunk differing only in their
   final characters -- and identical in length -- are embedded. If the tail is
   being truncated, the vectors come back identical. This catches a truncation
   the count-based check would miss (for example a server-side cap below
   ``n_ctx``).

An earlier version of this script measured only ``chunk[:len(chunk) * 0.75]``
while printing the *full* chunk's token count beside the verdict, so it reported
"ok" for chunks whose last quarter -- exactly where overflow begins -- was never
sent. Both signals below run against the whole chunk.

Run with the daemon up (`cementic embedding start`), after changing chunk_size,
n_ctx, or the embedding model:

    python scripts/measure_chunk_context_fit.py
"""

from __future__ import annotations

import sys

import requests
import tiktoken

from cementic.chunk import TOKENIZER, chunk_text
from cementic.config import Config
from cementic.embedding_text import describe_text_policy, format_document_text_for_model

CASES: dict[str, str] = {
    "english": "The quick brown fox jumps over the lazy dog near the riverbank. " * 200,
    "cjk": "机器学习模型的向量检索与语义搜索技术研究进展综述分析。" * 200,
    "code": "def f(x):\n    return {'k': [i**2 for i in range(x)]}\n" * 200,
    "diacritics": "Ωμέγα ñoño çedilla — Straße Ünïcödé тест δοκιμή " * 200,
}

#: Equal-length tails swapped in for the differential probe. Same length so the
#: two variants are byte-for-byte the same size as each other and as the chunk:
#: appending instead would make the probe test a *longer* text than the one the
#: pipeline actually embeds.
_TAIL_A = " ALPHA_MARKER alpha alpha "
_TAIL_B = " OMEGA_MARKER omega omega "


def main() -> int:
    config = Config()
    base_url = f"http://{config.llama_cpp.daemon_host}:{config.llama_cpp.daemon_port}"
    try:
        served = requests.get(f"{base_url}/v1/models", timeout=5).json()["data"]
    except Exception as error:
        print(f"embedding daemon not reachable at {base_url}: {error}", file=sys.stderr)
        print("start it with `cementic embedding start`", file=sys.stderr)
        return 1
    alias = served[0]["id"]
    encoding = tiktoken.get_encoding(TOKENIZER)
    budget = config.llama_cpp.n_ctx

    print(f"model      : {config.llama_cpp.model_path}")
    print(f"text policy: {describe_text_policy(config.llama_cpp.model_path)}")
    print(f"chunk_size : {config.pipeline.chunk_size} ({TOKENIZER} tokens)")
    print(f"n_ctx      : {budget} (model tokens)")
    print()

    def formatted(text: str) -> str:
        return format_document_text_for_model(text, config.llama_cpp.model_path)

    def model_tokens(text: str) -> int:
        response = requests.post(
            f"{base_url}/extras/tokenize/count", json={"input": text}, timeout=60
        )
        response.raise_for_status()
        return int(response.json()["count"])

    def embed(text: str) -> list[float]:
        response = requests.post(
            f"{base_url}/v1/embeddings",
            json={"model": alias, "input": text},
            timeout=180,
        )
        response.raise_for_status()
        return list(response.json()["data"][0]["embedding"])

    print(f"{'case':<12} {'cl100k':>7} {'chars':>7} {'model':>7} {'ratio':>6}  verdict")

    failures = 0
    worst_ratio = 0.0
    for name, body in CASES.items():
        chunk = chunk_text(
            body,
            chunk_size=config.pipeline.chunk_size,
            chunk_overlap=config.pipeline.chunk_overlap,
        )[0].content
        cl100k = len(encoding.encode(chunk))

        try:
            exact = model_tokens(formatted(chunk))
            # Equal-length tail swap: same size as the real chunk, different end.
            variant_a = chunk[: -len(_TAIL_A)] + _TAIL_A
            variant_b = chunk[: -len(_TAIL_B)] + _TAIL_B
            tail_ignored = embed(formatted(variant_a)) == embed(formatted(variant_b))
        except Exception as error:
            print(f"{name:<12} {cl100k:>7} {len(chunk):>7} {'-':>7} {'-':>6}  PROBE FAILED: {error}")
            failures += 1
            continue

        ratio = exact / cl100k if cl100k else 0.0
        worst_ratio = max(worst_ratio, ratio)
        if exact > budget:
            verdict = f"OVER BUDGET by {exact - budget} model tokens"
            failures += 1
        elif tail_ignored:
            # Fits by count but the tail still does not move the vector: the
            # server is capping input somewhere below n_ctx.
            verdict = "TRUNCATED (fits by count, but tail ignored)"
            failures += 1
        else:
            verdict = "ok (fits, tail affects the vector)"
        print(
            f"{name:<12} {cl100k:>7} {len(chunk):>7} {exact:>7} {ratio:>6.2f}  {verdict}"
        )

    print()
    if worst_ratio:
        safe = int(budget / worst_ratio)
        print(
            f"worst observed ratio {worst_ratio:.2f} model tokens per {TOKENIZER} token "
            f"(incl. task prefix)"
        )
        print(f"=> chunk_size must not exceed {safe} for these cases to fit in {budget}")
    if failures:
        print(f"{failures} case(s) failed: chunk_size and n_ctx are not compatible.")
        return 1
    print("All cases fit: full chunks embed intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
