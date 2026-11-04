#!/usr/bin/env python3
"""Concurrent prefill + decode bench for any OpenAI-compatible chat endpoint.

Reports per-concurrency:
  - TTFT (time to first token)
  - Prefill tok/s (prompt_tokens / TTFT)
  - Decode tok/s (completion tokens after first / decode span)
  - Aggregate decode tok/s across streams

Usage:
  python3 bench_concurrent.py [base_url] [--conc 1,2,4] [--prompt-tokens 2048]
                              [--gen-tokens 256] [--runs 2]

Notes:
  - base_url is the ROOT (without /v1); the health check hits /health.
  - The prompt carries a TARGET_TOKENS_HINT line so the suffix differs per size;
    for prefix-cache-sensitive backends (llama.cpp) pass a unique prefix and set
    "cache_prompt": false in your own variant, otherwise prompt_tokens reports
    only the newly evaluated tokens.
  - temperature 0 + top_k 1 can make hybrid-arch chat models emit EOS on the very
    first token; if you see predicted_n=1 / 0.0 tok/s, switch to temp 0.7 / top_k 20
    and take the second round.
  - A streaming chunk may hold several tokens, so never estimate tok/s from chunk
    counts; cross-check against the server's own statistics lines.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def make_prompt(n_tokens: int) -> str:
    # ~1 token per word for English filler; pad to approximate length.
    unit = "river stone sky clear number "
    words = max(8, n_tokens)
    return (
        "Reply with a short numbered list of facts after reading this filler.\n\n"
        + (unit * ((words // 6) + 1))[: words * 6]
        + f"\n\nTARGET_TOKENS_HINT={n_tokens}\nWrite {min(12, max(4, n_tokens // 500))} bullet points."
    )


def stream_once(url: str, prompt: str, max_tokens: int, timeout: float = 600.0):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    n_out = 0
    last = t0
    prompt_tokens = None
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = obj.get("usage") or {}
            if usage.get("prompt_tokens"):
                prompt_tokens = usage["prompt_tokens"]
            if usage.get("completion_tokens"):
                completion_tokens = usage["completion_tokens"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                last = now
                n_out += 1
    if ttft is None:
        ttft = last - t0
    decode_span = max(1e-6, last - t0 - ttft)
    n_completion = completion_tokens if completion_tokens is not None else n_out
    # first token counted in TTFT; remaining for decode rate
    n_decode = max(0, n_completion - 1)
    return {
        "ttft_s": ttft,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": n_completion,
        "prefill_tps": (prompt_tokens / ttft) if (prompt_tokens and ttft > 0) else None,
        "decode_tps": (n_decode / decode_span) if n_decode else 0.0,
        "wall_s": last - t0,
        "n_stream_chunks": n_out,
    }


def run_conc(url: str, conc: int, prompt: str, max_tokens: int, runs: int):
    rows = []
    for run in range(runs):
        t_batch0 = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(stream_once, url, prompt, max_tokens) for _ in range(conc)]
            for f in as_completed(futs):
                results.append(f.result())
        wall = time.perf_counter() - t_batch0
        agg_decode_tokens = sum(max(0, (r["completion_tokens"] or 0) - 1) for r in results)
        mean_ttft = st.mean(r["ttft_s"] for r in results)
        decode_window = max(1e-6, wall - mean_ttft)
        rows.append(
            {
                "run": run,
                "results": results,
                "wall_s": wall,
                "mean_ttft_s": mean_ttft,
                "median_ttft_s": st.median(r["ttft_s"] for r in results),
                "mean_prefill_tps": st.mean(
                    [r["prefill_tps"] for r in results if r["prefill_tps"]]
                )
                if any(r["prefill_tps"] for r in results)
                else None,
                "mean_decode_tps": st.mean(r["decode_tps"] for r in results),
                "median_decode_tps": st.median(r["decode_tps"] for r in results),
                "agg_decode_tps": agg_decode_tokens / decode_window,
                "prompt_tokens": results[0]["prompt_tokens"],
            }
        )
    return rows


def summarize(conc: int, rows):
    print(f"\n=== concurrency={conc}  runs={len(rows)} ===")
    pt = rows[0]["prompt_tokens"]
    print(f"prompt_tokens≈{pt}")
    print(
        f"TTFT      mean {st.mean(r['mean_ttft_s'] for r in rows):6.3f}s  "
        f"median {st.median(r['median_ttft_s'] for r in rows):6.3f}s"
    )
    pref = [r["mean_prefill_tps"] for r in rows if r["mean_prefill_tps"]]
    if pref:
        print(f"Prefill   mean {st.mean(pref):6.1f} tok/s (per-request, prompt/TTFT)")
    print(
        f"Decode/req mean {st.mean(r['mean_decode_tps'] for r in rows):6.1f}  "
        f"median {st.median(r['median_decode_tps'] for r in rows):6.1f} tok/s"
    )
    print(
        f"Decode/agg mean {st.mean(r['agg_decode_tps'] for r in rows):6.1f} tok/s "
        f"(all streams combined)"
    )
    print(f"Batch wall mean {st.mean(r['wall_s'] for r in rows):6.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url", nargs="?", default="http://127.0.0.1:8080")
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    root = args.base_url.rstrip("/")
    url = root + "/v1/chat/completions"
    prompt = make_prompt(args.prompt_tokens)

    health = root + "/health"
    try:
        with urllib.request.urlopen(health, timeout=10) as r:
            print("health", r.read().decode()[:200])
    except urllib.error.HTTPError as exc:
        # A loading server answers 503 *with* a body; do not treat that as ready.
        print(f"NOT READY: /health -> HTTP {exc.code}")
        raise SystemExit(2)

    if not args.no_warmup:
        print("warmup...")
        stream_once(url, make_prompt(256), 32)

    for c in [int(x) for x in args.conc.split(",") if x.strip()]:
        rows = run_conc(url, c, prompt, args.gen_tokens, args.runs)
        summarize(c, rows)


if __name__ == "__main__":
    main()
