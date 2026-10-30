"""E6 analysis of a training run: where the weight goes, value paths, sparsity over training.

Reads <run>/rollouts/step-*.jsonl and <run>/metrics.jsonl written by answergain.cwopd.train.
Answer restatement uses ALERD's strict answer-only rule (answergain.signals.answer_only) with the
teacher target a_T in place of the gold answer, plus a looser "after the first \\boxed" category.
"""

import argparse
import json
import statistics
from pathlib import Path

from answergain.cwopd.data import read_jsonl
from answergain.signals.answer_only import is_answer_only_step

BINS = 10


def weight_shares(rows: list[dict]) -> dict:
    mass = {"all": 0.0, "answer_only": 0.0, "after_first_box": 0.0}
    tokens = dict.fromkeys(mass, 0)
    for row in rows:
        if not row.get("weights") or not row.get("teacher_answer"):
            continue
        boxed_seen = False
        for text, n, w in zip(row["steps"], row["step_tokens"], row["weights"]):
            labels = {"all": True, "answer_only": is_answer_only_step(text, row["teacher_answer"]),
                      "after_first_box": boxed_seen}
            boxed_seen = boxed_seen or "\\boxed" in text
            for name, hit in labels.items():
                if hit:
                    mass[name] += w * n
                    tokens[name] += n
    return {
        name: {"weight_share": mass[name] / mass["all"] if mass["all"] else None,
               "token_share": tokens[name] / tokens["all"] if tokens["all"] else None}
        for name in ("answer_only", "after_first_box")
    }


def value_paths(rows: list[dict]) -> dict:
    """Mean V over normalized position k/K (BINS bins), for rollouts reaching / missing a_T."""
    paths = {}
    for label, keep in (("reaches_teacher_answer", True), ("misses_teacher_answer", False)):
        bins = [[] for _ in range(BINS + 1)]
        for row in rows:
            values = row.get("values")
            if not values or row.get("fidelity") is not keep or len(values) < 2:
                continue
            for k, value in enumerate(values):
                bins[round(BINS * k / (len(values) - 1))].append(value)
        paths[label] = [statistics.fmean(b) if b else None for b in bins]
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="training output dir (checkpoints/cwopd/<run>)")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run = Path(args.run_dir)
    rows = [row for path in sorted((run / "rollouts").glob("step-*.jsonl")) for row in read_jsonl(path)]
    metrics = read_jsonl(run / "metrics.jsonl") if (run / "metrics.jsonl").exists() else []
    report = {
        "dumped_rollouts": len(rows),
        "weight_on_answer_restatement": weight_shares(rows),
        "value_path": value_paths(rows),
        "sparsity_ratio_over_training": [
            {"step": m["step"], "sparsity_ratio": m.get("sparsity_ratio"), "g_bar": m.get("g_bar"),
             "fidelity": m.get("fidelity")}
            for m in metrics
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sparsity_ratio_over_training"}, indent=2))


if __name__ == "__main__":
    main()
