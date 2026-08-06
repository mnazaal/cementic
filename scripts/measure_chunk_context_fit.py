#!/usr/bin/env python
"""Check that a full-size chunk survives the embedding model's context window.

``pipeline.chunk_size`` counts tokens with a tiktoken encoding; the embedding
server's ``llama_cpp.n_ctx`` counts them with the *model's own* tokenizer, and a
task prefix is added on top. Both default to 512, so a full chunk could in
principle overflow -- either failing to embed, or embedding with its tail
silently dropped, which degrades retrieval with no error anywhere.

This measures rather than assumes, across text where the two tokenizers diverge
most. For each case it embeds two chunks differing only in their final words: if
the tail is being truncated the vectors come back identical.

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

    print(f"model      : {config.llama_cpp.model_path}")
    print(f"text policy: {describe_text_policy(config.llama_cpp.model_path)}")
    print(f"chunk_size : {config.pipeline.chunk_size} ({TOKENIZER} tokens)")
    print(f"n_ctx      : {config.llama_cpp.n_ctx} (model tokens)")
    print()
    print(f"{'case':<12} {'tokens':>7} {'chars':>7}  verdict")

    def embed(text: str) -> list[float]:
        response = requests.post(
            f"{base_url}/v1/embeddings",
            json={
                "model": alias,
                "input": format_document_text_for_model(text, config.llama_cpp.model_path),
            },
            timeout=180,
        )
        response.raise_for_status()
        return list(response.json()["data"][0]["embedding"])

    failures = 0
    for name, body in CASES.items():
        chunk = chunk_text(
            body,
            chunk_size=config.pipeline.chunk_size,
            chunk_overlap=config.pipeline.chunk_overlap,
        )[0].content
        token_count = len(encoding.encode(chunk))
        head = chunk[: int(len(chunk) * 0.75)]
        try:
            first = embed(head + " ALPHA_MARKER alpha alpha")
            second = embed(head + " OMEGA_MARKER omega omega")
        except Exception as error:
            print(f"{name:<12} {token_count:>7} {len(chunk):>7}  EMBED FAILED: {error}")
            failures += 1
            continue
        if first == second:
            print(f"{name:<12} {token_count:>7} {len(chunk):>7}  TRUNCATED (tail ignored)")
            failures += 1
        else:
            print(f"{name:<12} {token_count:>7} {len(chunk):>7}  ok (tail affects the vector)")

    print()
    if failures:
        print(f"{failures} case(s) failed: chunk_size and n_ctx are not compatible.")
        return 1
    print("All cases fit: full chunks embed intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
