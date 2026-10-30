"""E1 (distinctness) and E2 (probe validity) report from pilot.jsonl and, if present, mc.jsonl.

E1, within each rollout: Spearman of g_k with step-mean student entropy and step-mean divergence,
the partial correlation of g with each local signal controlling for the other, and the Jaccard
overlap of the top-10% steps. A strong overlap with divergence is the proposal's stop signal for H1.
E2: correlation of the probe value V_k with the Monte Carlo value, and of g_k with |G_k|.
"""

import argparse
import json
import math
from pathlib import Path

from answergain.cwopd.data import read_jsonl
from answergain.cwopd.objective import consequence
from answergain.cwopd.stats import jaccard_top, partial, pearson, spearman, summarize

# Our reading of "strong overlap" for the H1 decision; not fixed by the proposal.
STOP_RHO, STOP_JACCARD = 0.5, 0.5


def values_from(logps: list[float] | None, n_answer: int, form: str) -> list[float] | None:
    if logps is None:
        return None
    return [math.exp(v / n_answer if form == "geomean" else v) for v in logps]


def e1(rows: list[dict], min_steps: int, top_fraction: float) -> dict:
    per = {key: [] for key in (
        "rho_g_entropy", "rho_g_divergence", "rho_entropy_divergence", "partial_g_entropy|div",
        "partial_g_divergence|ent", f"jaccard_top{int(top_fraction * 100)}_g_entropy",
        f"jaccard_top{int(top_fraction * 100)}_g_divergence", "rho_g_raw_vs_geomean", "rho_g_teacher_vs_student",
    )}
    sparsity_num = sparsity_den = 0.0
    used = 0
    for row in rows:
        g = consequence(values_from(row["teacher_logp"], row["n_answer_tokens"], "geomean"))
        sparsity_num += sum(g)
        sparsity_den += len(g)
        if len(g) < min_steps:
            continue
        used += 1
        h, d = row["entropy"], row["divergence"]
        r_gh, r_gd, r_hd = spearman(g, h), spearman(g, d), spearman(h, d)
        per["rho_g_entropy"].append(r_gh)
        per["rho_g_divergence"].append(r_gd)
        per["rho_entropy_divergence"].append(r_hd)
        per["partial_g_entropy|div"].append(partial(r_gh, r_gd, r_hd))
        per["partial_g_divergence|ent"].append(partial(r_gd, r_gh, r_hd))
        per[f"jaccard_top{int(top_fraction * 100)}_g_entropy"].append(jaccard_top(g, h, top_fraction))
        per[f"jaccard_top{int(top_fraction * 100)}_g_divergence"].append(jaccard_top(g, d, top_fraction))
        g_raw = consequence(values_from(row["teacher_logp"], row["n_answer_tokens"], "raw"))
        per["rho_g_raw_vs_geomean"].append(spearman(g, g_raw))
        student = values_from(row.get("student_logp"), row["n_answer_tokens"], "geomean")
        if student:
            per["rho_g_teacher_vs_student"].append(spearman(g, consequence(student)))
    report = {name: summarize(values) for name, values in per.items()}
    report["rollouts"] = len(rows)
    report[f"rollouts_with_at_least_{min_steps}_steps"] = used
    report["sparsity_ratio"] = sparsity_num / sparsity_den if sparsity_den else None  # E[sum g]/E[K]
    report["fidelity"] = sum(1 for r in rows if r.get("fidelity")) / len(rows) if rows else None
    rho = report["rho_g_divergence"].get("mean")
    jac = report[f"jaccard_top{int(top_fraction * 100)}_g_divergence"].get("mean")
    report["h1_stop_signal"] = bool(
        (rho is not None and abs(rho) >= STOP_RHO) or (jac is not None and jac >= STOP_JACCARD)
    )
    return report


def e2(rows: list[dict], mc_rows: list[dict]) -> dict:
    by_id = {row["id"]: row for row in rows}
    v_geo, v_raw, v_mc, g_probe, g_mc, gaps = [], [], [], [], [], []
    for mc in mc_rows:
        row = by_id.get(mc["id"])
        if row is None:
            continue
        geo = values_from(row["teacher_logp"], row["n_answer_tokens"], "geomean")
        raw = values_from(row["teacher_logp"], row["n_answer_tokens"], "raw")
        if len(geo) != len(mc["mc_value"]):
            continue
        v_geo += geo
        v_raw += raw
        v_mc += mc["mc_value"]
        g_probe += consequence(geo)
        g_mc += [abs(b - a) for a, b in zip(mc["mc_value"], mc["mc_value"][1:])]
        gaps += [abs(a - b) for a, b in zip(geo, mc["mc_value"])]
    return {
        "points": len(v_mc),
        "pearson_V_geomean_vs_mc": pearson(v_geo, v_mc), "spearman_V_geomean_vs_mc": spearman(v_geo, v_mc),
        "pearson_V_raw_vs_mc": pearson(v_raw, v_mc), "spearman_V_raw_vs_mc": spearman(v_raw, v_mc),
        "pearson_g_vs_abs_G": pearson(g_probe, g_mc), "spearman_g_vs_abs_G": spearman(g_probe, g_mc),
        # Sec. 5.3(1): |g_k - |G_k|| <= 2 eps with eps = sup |V_probe - V|; report its empirical size
        "epsilon_mean": sum(gaps) / len(gaps) if gaps else None, "epsilon_max": max(gaps) if gaps else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--mc", help="mc.jsonl from answergain.cwopd.mc_value")
    parser.add_argument("--output", required=True, help="report JSON (a .md summary is written next to it)")
    parser.add_argument("--min-steps", type=int, default=5, help="rollouts with fewer steps are skipped in E1")
    parser.add_argument("--top-fraction", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = read_jsonl(args.pilot)
    report = {"E1": e1(rows, args.min_steps, args.top_fraction)}
    if args.mc and Path(args.mc).exists():
        report["E2"] = e2(rows, read_jsonl(args.mc))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    lines = ["# CW-OPD pilot report", "", "## E1 distinctness (within-rollout, mean / median over rollouts)", ""]
    for key, value in report["E1"].items():
        if isinstance(value, dict) and value.get("n"):
            lines.append(f"- {key}: mean {value['mean']:.3f}, median {value['median']:.3f} (n={value['n']})")
        elif not isinstance(value, dict):
            lines.append(f"- {key}: {value}")
    if "E2" in report:
        lines += ["", "## E2 probe validity", ""] + [f"- {k}: {v}" for k, v in report["E2"].items()]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
