"""E1 distinctness pilot (RQ1), before any training.

On rollouts of the initial student, record per step: the probe values under the teacher and under
the student (both value forms are recoverable from the stored log-probability sums), step-mean
student entropy and step-mean teacher-student divergence. answergain.cwopd.analyze_pilot turns
this into the correlations, partial correlations and top-k overlaps of E1.
"""

import argparse
from pathlib import Path

import torch

from answergain.cwopd.data import read_jsonl, write_jsonl
from answergain.cwopd.objective import step_means
from answergain.cwopd.probes import PROBE_MODES
from answergain.cwopd.scoring import ScoredRollout, student_probe_logps, student_stats, teacher_pass, vocab_size
from answergain.cwopd.steps import step_texts
from answergain.eval.graders.answer_scoring import extract_boxed, math_answers_equal
from answergain.segment import SEGMENTERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--segmenter", choices=sorted(SEGMENTERS), default="sentence")
    parser.add_argument("--probe-mode", choices=PROBE_MODES, default="packed")
    parser.add_argument("--teacher-max-tokens", type=int, default=2560)
    parser.add_argument("--divergence", choices=("rkl", "fkl", "jsd"), default="rkl")
    parser.add_argument("--jsd-beta", type=float, default=0.9)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int)
    return parser


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    load = lambda name: AutoModelForCausalLM.from_pretrained(  # noqa: E731
        name, dtype=dtype, attn_implementation=args.attn_implementation).to(device).eval()
    teacher, student = load(args.teacher), load(args.student)
    vocab = vocab_size(tokenizer, teacher, student)

    rows = []
    rollouts = read_jsonl(args.rollouts)[: args.limit]
    for index, source in enumerate(rollouts, start=1):
        rollout = ScoredRollout(prompt_ids=source["prompt_ids"], response_ids=source["response_ids"],
                                answer=source.get("teacher_answer"))
        teacher_pass(teacher, tokenizer, rollout, probes=True, probe_mode=args.probe_mode,
                     max_tokens=args.teacher_max_tokens, vocab=vocab, keep_logits=True, cache_device=device,
                     segmenter=args.segmenter)
        if rollout.answer_logps is None:
            continue
        teacher_logps = rollout.answer_logps
        student_stats(student, rollout, vocab, args.divergence, args.jsd_beta)
        student_logps = student_probe_logps(student, tokenizer, rollout, args.probe_mode, args.teacher_max_tokens)
        spans = rollout.probed_spans
        text = tokenizer.decode(rollout.response_ids, skip_special_tokens=True)
        rows.append({
            "id": source["id"], "finish": source.get("finish"), "teacher_answer": rollout.answer,
            "student_answer": extract_boxed(text),
            "fidelity": math_answers_equal(extract_boxed(text), rollout.answer),
            "n_answer_tokens": rollout.n_answer_tokens,
            "n_steps_raw": len(rollout.spans), "step_tokens": [end - start for start, end in spans],
            "steps": step_texts(tokenizer, rollout.response_ids, spans),
            # sum_i log p(a_i | h_k, probe): geomean V = exp(sum / N), raw V = exp(sum)
            "teacher_logp": teacher_logps.sum(-1).tolist(),
            "student_logp": student_logps.sum(-1).tolist() if student_logps is not None else None,
            "entropy": step_means(rollout.entropy.tolist(), spans),
            "divergence": step_means(rollout.divergence.tolist(), spans),
        })
        rollout.teacher_logits = None
        if index % 25 == 0:
            print(f"scored {index}/{len(rollouts)} rollouts ({len(rows)} with a probe target)", flush=True)
    write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} scored rollouts -> {args.output}")


if __name__ == "__main__":
    main()
