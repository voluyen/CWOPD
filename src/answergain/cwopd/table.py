"""Results table in RIVAD Table 2's column order, from <results-dir>/<tag>/summary.json."""

import argparse
import json
from pathlib import Path

from answergain.cwopd.lm_eval_suite import RIVAD_BENCHMARKS

HEADERS = {"gsm8k": "GSM8K", "gsm_plus": "GSM-Plus", "math": "MATH", "mbpp": "MBPP", "sciq": "SciQ",
           "mmlu_stem": "STEM", "mmlu_pro_math": "Pro-Math", "bbh_cot": "BBH-COT", "ifeval": "IFEval"}

# RIVAD Table 2 (Qwen2.5-14B -> 1.5B) and Table 6 (Gemma2-9B -> 2B), % accuracy, for side-by-side reading.
PUBLISHED = {
    "qwen25-1.5b": {
        "Qwen2.5-14B-Instruct (paper)": [80.3, 59.7, 21.7, 78.9, 86.4, 70.5, 77.6, 60.8],
        "Qwen2.5-1.5B-Instruct (paper)": [57.1, 38.8, 16.9, 38.4, 85.5, 49.5, 34.4, 36.5],
        "Sequence-Level KD (paper)": [56.5, 38.2, 16.5, 40.6, 87.9, 48.9, 32.6, 36.6],
        "Supervised-KD (paper)": [56.3, 37.8, 17.9, 41.0, 86.6, 48.4, 35.3, 36.4],
        "GKD beta=0.9 (paper)": [57.9, 39.9, 18.1, 41.8, 86.6, 47.7, 35.7, 36.2],
        "TSD-KD (paper)": [60.1, 40.5, 26.1, 42.1, 93.0, 50.0, 36.9, 40.2],
        "CSD (paper)": [61.5, 41.5, 29.3, 43.8, 81.9, 51.3, 44.4, 42.1],
        "AMiD (paper)": [61.1, 42.1, 28.3, 43.6, 86.7, 51.6, 40.0, 42.0],
        "RIVAD (paper)": [68.8, 50.2, 53.0, 49.2, 82.4, 52.6, 45.5, 42.0],
    },
    "gemma2-2b": {
        "Gemma2-9B (paper)": [57.7, 40.8, 25.8, 40.0, None, 50.2, None, None],
        "Gemma2-2B (paper)": [53.7, 36.2, 16.1, 39.2, None, 25.5, None, None],
        "Sequence-Level KD (paper)": [53.4, 33.8, 17.7, 39.2, None, 25.9, None, None],
        "Supervised-KD (paper)": [53.6, 36.8, 17.9, 39.0, None, 25.5, None, None],
        "GKD beta=0.9 (paper)": [54.1, 36.9, 17.2, 40.0, None, 26.2, None, None],
        "TSD-KD (paper)": [55.3, 37.9, 18.9, 40.4, None, 27.0, None, None],
        "RIVAD (paper)": [54.8, 37.0, 24.1, 38.0, None, 46.2, None, None],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--pair", help="key of PUBLISHED rows to print alongside (qwen25-1.5b, gemma2-2b)")
    parser.add_argument("--benchmarks", default=",".join(RIVAD_BENCHMARKS),
                        help="averaged columns (RIVAD Table 6, Gemma2: gsm8k,gsm_plus,math,mbpp,mmlu_stem)")
    parser.add_argument("--extra", default="ifeval", help="benchmarks shown after the average")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.results_dir)
    averaged = [name for name in args.benchmarks.split(",") if name]
    extra = [name for name in args.extra.split(",") if name]
    columns = averaged + extra
    lines = ["| Method | " + " | ".join(HEADERS[c] for c in averaged) + " | Avg. | "
             + " | ".join(HEADERS[c] for c in extra) + " |",
             "|" + "---|" * (len(columns) + 2)]

    def row(name, values):
        main_values = values[: len(averaged)]
        present = [v for v in main_values if v is not None]
        avg = f"{sum(present) / len(present):.1f}" if len(present) == len(averaged) else "-"
        cells = ["-" if v is None else f"{v:.1f}" for v in values]
        return f"| {name} | " + " | ".join(cells[: len(averaged)]) + f" | {avg} | " + " | ".join(
            cells[len(averaged):]) + " |"

    for name, values in PUBLISHED.get(args.pair, {}).items():
        by_name = dict(zip(RIVAD_BENCHMARKS, values))
        lines.append(row(name, [by_name.get(c) for c in columns]))
    for path in sorted(root.glob("*/summary.json")):
        found = {r["benchmark"]: r.get("accuracy") for r in json.loads(path.read_text())}
        values = [None if found.get(c) is None else 100 * found[c] for c in columns]
        lines.append(row(path.parent.name, values))
    table = "\n".join(lines)
    print(table)
    (root / "table.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
