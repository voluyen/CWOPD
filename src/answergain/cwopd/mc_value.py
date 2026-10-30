"""E2 probe validity (RQ2): Monte Carlo continuation values of the teacher at every step boundary.

For a subset of pilot rollouts, sample M teacher continuations from each prefix h_k = (x, y_1:k) and
record the fraction whose final boxed answer equals a_T: an estimate of V(h_k) of Sec. 5.1, to be
compared with the probe value V_k (answergain.cwopd.analyze_pilot).
"""

import argparse
import json
from pathlib import Path

from answergain.cwopd.data import read_jsonl, write_jsonl
from answergain.eval.graders.answer_scoring import extract_boxed, math_answers_equal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", required=True, help="pilot.jsonl (gives the probed step boundaries)")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--samples", type=int, default=8, help="M continuations per boundary")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    rollouts = {row["id"]: row for row in read_jsonl(args.rollouts)}
    pilot = read_jsonl(args.pilot)[: args.n_rollouts]

    prompts, index = [], []
    for row in pilot:
        source = rollouts[row["id"]]
        cut = 0
        cuts = [0]
        for length in row["step_tokens"]:
            cut += length
            cuts.append(cut)
        for k, cut in enumerate(cuts):
            prompts.append(TokensPrompt(prompt_token_ids=source["prompt_ids"] + source["response_ids"][:cut]))
            index.append((row["id"], k, cut))
    print(f"{len(prompts)} prefixes x {args.samples} continuations over {len(pilot)} rollouts")

    llm = LLM(model=args.teacher, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, seed=args.seed, enable_prefix_caching=True)
    sampling = SamplingParams(n=args.samples, temperature=args.temperature, max_tokens=args.max_tokens,
                              seed=args.seed)
    outputs = llm.generate(prompts, sampling)

    values: dict[str, dict[int, float]] = {}
    for (rid, k, cut), output in zip(index, outputs):
        source = rollouts[rid]
        prefix = tokenizer.decode(source["response_ids"][:cut], skip_special_tokens=True)
        hits = [math_answers_equal(extract_boxed(prefix + c.text), source["teacher_answer"]) for c in output.outputs]
        values.setdefault(rid, {})[k] = sum(hits) / len(hits)
    rows = [{"id": rid, "mc_value": [by_k[k] for k in sorted(by_k)]} for rid, by_k in values.items()]
    write_jsonl(Path(args.output), rows)
    Path(args.output).with_suffix(".args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"wrote Monte Carlo values for {len(rows)} rollouts -> {args.output}")


if __name__ == "__main__":
    main()
