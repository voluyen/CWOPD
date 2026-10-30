"""E5 cost (RQ3): teacher wall-clock and peak memory with and without packed probes, against K.

Each rollout is scored three ways: the plain OPD teacher pass, the packed pass (probes in the same
sequence) and, for small K, one pass per prefix. K is varied by re-cutting the response into K
equal token chunks, plus the natural sentence segmentation.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from answergain.cwopd.data import read_jsonl
from answergain.cwopd.probes import encode_probe, response_logits_only, score_rollout
from answergain.cwopd.steps import rollout_step_spans


def chunk_spans(n_tokens: int, k: int) -> list[tuple[int, int]]:
    k = max(1, min(k, n_tokens))
    bounds = [round(i * n_tokens / k) for i in range(k + 1)]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def timed(fn, device) -> tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    return time.perf_counter() - start, peak


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-rollouts", type=int, default=64)
    parser.add_argument("--k-values", default="4,16,64,128")
    parser.add_argument("--separate-max-k", type=int, default=16, help="one-pass-per-prefix only up to this K")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation=args.attn_implementation,
    ).to(device).eval()
    rollouts = [r for r in read_jsonl(args.rollouts) if r.get("teacher_answer")][: args.n_rollouts]

    settings = [("sentence", None)] + [(f"K={k}", int(k)) for k in args.k_values.split(",")]
    results = {}
    for index, rollout in enumerate(rollouts):
        prompt, response = rollout["prompt_ids"], rollout["response_ids"]
        probe_ids, answer_ids = encode_probe(tokenizer, rollout["teacher_answer"])
        runs = [("plain", None, lambda: response_logits_only(teacher, prompt, response))]
        for name, k in settings:
            spans = rollout_step_spans(tokenizer, response) if k is None else chunk_spans(len(response), k)
            runs.append((f"packed/{name}", spans, lambda s=spans: score_rollout(
                teacher, prompt, response, s, probe_ids, answer_ids, mode="packed")))
            if len(spans) <= args.separate_max_k:
                runs.append((f"separate/{name}", spans, lambda s=spans: score_rollout(
                    teacher, prompt, response, s, probe_ids, answer_ids, mode="separate")))
        for name, spans, fn in runs:
            seconds, peak = timed(fn, device)
            if index < args.warmup:
                continue  # CUDA kernels / allocator warm-up
            entry = results.setdefault(name, {"seconds": [], "peak_gb": [], "K": [], "tokens": []})
            entry["seconds"].append(seconds)
            entry["peak_gb"].append(peak)
            base = len(prompt) + len(response)
            entry["K"].append(len(spans) if spans else 0)
            entry["tokens"].append(base + ((len(spans) + 1) * (len(probe_ids) + len(answer_ids)) if spans else 0))

    plain = statistics.fmean(results["plain"]["seconds"]) if results.get("plain") else None
    report = {}
    for name, entry in results.items():
        mean = statistics.fmean(entry["seconds"])
        report[name] = {
            "rollouts": len(entry["seconds"]), "mean_seconds": mean,
            "overhead_vs_plain": mean / plain - 1 if plain else None,
            "mean_K": statistics.fmean(entry["K"]), "mean_sequence_tokens": statistics.fmean(entry["tokens"]),
            "peak_memory_gb": max(entry["peak_gb"]),
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    for name, row in report.items():
        print(f"{name:20s} {row['mean_seconds'] * 1000:8.1f} ms  overhead {row['overhead_vs_plain'] or 0:+.1%}  "
              f"K={row['mean_K']:.1f}  tokens={row['mean_sequence_tokens']:.0f}  peak={row['peak_memory_gb']:.1f} GB")


if __name__ == "__main__":
    main()
