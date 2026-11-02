"""RIVAD / TSD-KD evaluation suite through lm-evaluation-harness (vLLM backend).

TSD-KD (whose numbers RIVAD Table 2 reuses) evaluates with lm-evaluation-harness, shots from its
Appendix A.10: GSM8K 5, GSM-Plus 5, MATH 4 (minerva_math), MMLU-Pro-Math 5, MMLU-STEM 5, SciQ 0,
MBPP 0, BBH 3 (CoT). lm-eval runs in its own environment (scripts/cwopd/setup_eval.sh) since it
pins its own vLLM; this wrapper is stdlib-only and calls it as a subprocess.

The primary metric per benchmark (e.g. flexible- vs strict-extract on GSM8K) is not stated by
TSD-KD; calibrate it by evaluating the untrained teacher and student first and comparing with the
first two rows of RIVAD Table 2 (scripts/cwopd/phase0_calibrate.sh). Every metric is kept.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# benchmark -> (lm-eval task, shots, primary metric key, fallback keys)
SUITE = {
    "gsm8k": ("gsm8k", 5, "exact_match,flexible-extract", ["exact_match,strict-match"]),
    "gsm_plus": ("gsm_plus", 5, "exact_match,flexible-extract", ["exact_match,strict-match"]),
    "math": ("minerva_math", 4, "exact_match,none", ["math_verify,none"]),
    # lm-eval's mbpp / mbpp_instruct filters score ~0 on Qwen2.5-Instruct (see lm_eval_tasks/utils.py)
    "mbpp": ("mbpp_cwopd", 0, "pass_at_1,extract_code", ["pass_at_1,none"]),
    "sciq": ("sciq", 0, "acc,none", ["acc_norm,none"]),
    "mmlu_stem": ("mmlu_stem", 5, "acc,none", []),
    "mmlu_pro_math": ("mmlu_pro_math", 5, "exact_match,custom-extract", []),
    "bbh_cot": ("bbh_cot_fewshot", 3, "exact_match,get-answer", []),
    # not in RIVAD's average: TSD-KD's instruction-following check, the proposal's "no harm" test
    "ifeval": ("ifeval", 0, "prompt_level_strict_acc,none", ["inst_level_strict_acc,none"]),
}
RIVAD_BENCHMARKS = ["gsm8k", "gsm_plus", "math", "mbpp", "sciq", "mmlu_stem", "mmlu_pro_math", "bbh_cot"]
CODE_TASKS = {"mbpp"}
TASKS_DIR = Path(__file__).parent / "lm_eval_tasks"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--benchmarks", default=",".join(RIVAD_BENCHMARKS), help=f"comma list of {sorted(SUITE)}")
    parser.add_argument("--lm-eval-python", default=os.environ.get("LMEVAL_PYTHON", sys.executable),
                        help="python of the lm-eval environment (default: $LMEVAL_PYTHON)")
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--limit", type=float, help="lm-eval --limit, for smoke runs only")
    parser.add_argument("--dry-run", action="store_true", help="print the lm-eval commands and exit")
    return parser


def commands(args) -> list[tuple[list[str], list[str]]]:
    """One lm-eval call per shot count (the flag is global to a call)."""
    names = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    unknown = sorted(set(names) - set(SUITE))
    if unknown:
        raise SystemExit(f"unknown benchmarks {unknown}; choose from {sorted(SUITE)}")
    by_shots: dict[int, list[str]] = {}
    for name in names:
        by_shots.setdefault(SUITE[name][1], []).append(name)
    model_args = (
        f"pretrained={args.model},dtype=bfloat16,gpu_memory_utilization={args.gpu_memory_utilization},"
        f"data_parallel_size={args.data_parallel_size},max_model_len={args.max_model_len},seed={args.seed}"
    )
    out = []
    for shots, group in sorted(by_shots.items()):
        cmd = [
            args.lm_eval_python, "-m", "lm_eval", "run", "--model", "vllm", "--model_args", model_args,
            "--tasks", ",".join(SUITE[name][0] for name in group), "--num_fewshot", str(shots),
            "--include_path", str(TASKS_DIR), "--batch_size", str(args.batch_size), "--seed", str(args.seed),
            "--output_path", str(Path(args.results_dir) / args.tag / "lm_eval" / f"{shots}shot"),
        ]
        if args.apply_chat_template:
            cmd += ["--apply_chat_template", "--fewshot_as_multiturn"]
        if any(name in CODE_TASKS for name in group):
            cmd.append("--confirm_run_unsafe_code")
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        out.append((cmd, group))
    return out


def read_metrics(output_dir: Path) -> dict[str, dict]:
    """task/group name -> metrics of the newest results_*.json under output_dir."""
    files = sorted(output_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no lm-eval results under {output_dir}")
    data = json.loads(files[-1].read_text())
    merged = dict(data.get("results", {}))
    merged.update(data.get("groups", {}))
    return merged


def primary(name: str, metrics: dict[str, dict]) -> tuple[float | None, str | None]:
    task, _, key, fallbacks = SUITE[name]
    row = metrics.get(task, {})
    for candidate in (key, *fallbacks):
        if isinstance(row.get(candidate), (int, float)):
            return float(row[candidate]), candidate
    return None, None


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    plan = commands(args)
    if args.dry_run:
        for cmd, _ in plan:
            print(" ".join(cmd))
        return
    env = dict(os.environ, HF_ALLOW_CODE_EVAL="1")
    rows = []
    for cmd, group in plan:
        print("$ " + " ".join(cmd), flush=True)
        started = time.time()
        output_dir = Path(cmd[cmd.index("--output_path") + 1])
        returncode = subprocess.run(cmd, env=env).returncode
        # vLLM can abort while tearing down ("terminate called without an active exception") after
        # lm-eval has saved its results; only trust that exit if a fresh results file exists.
        fresh = any(f.stat().st_mtime >= started for f in output_dir.rglob("results_*.json"))
        if returncode and not fresh:
            raise subprocess.CalledProcessError(returncode, cmd)
        if returncode:
            print(f"warning: lm-eval exited {returncode} after saving results; continuing", flush=True)
        metrics = read_metrics(output_dir)
        for name in group:
            value, key = primary(name, metrics)
            rows.append({"benchmark": name, "accuracy": value, "metric": key, "model": args.tag,
                         "all_metrics": metrics.get(SUITE[name][0], {})})
            print(f"[{name}] {key} = {value}")
    path = Path(args.results_dir) / args.tag / "summary.json"
    if path.exists():  # a partial re-run (e.g. --benchmarks mbpp) updates its rows and keeps the others
        done = {row["benchmark"] for row in rows}
        rows = [row for row in json.loads(path.read_text()) if row["benchmark"] not in done] + rows
    path.write_text(json.dumps(rows, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
