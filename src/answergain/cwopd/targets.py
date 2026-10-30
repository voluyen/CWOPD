"""Teacher responses and targets a_T with vLLM, for teachers without a released generation set
(Gemma2-9B) or to regenerate them. Rewrites train.jsonl / heldout.jsonl in place.

Proposal Sec. 4.2: one response per prompt, generated once before training; a_T is not checked
for correctness.
"""

import argparse
import json
from pathlib import Path

from answergain.cwopd.data import read_jsonl, teacher_answer, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--max-tokens", type=int, default=512, help="teacher response budget (student total 1024)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--summary", help="JSON written when done (the stage's `creates`)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    llm = LLM(
        model=args.teacher, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len, seed=args.seed,
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed)
    summary = {}
    for name in ("train", "heldout"):
        path = Path(args.data_dir) / f"{name}.jsonl"
        records = read_jsonl(path)
        conversations = [[{"role": "user", "content": record["instruction"]}] for record in records]
        outputs = llm.chat(conversations, sampling, use_tqdm=True)
        for record, output in zip(records, outputs):
            response = output.outputs[0].text
            record["teacher_response"] = response
            record["teacher_answer"] = teacher_answer(tokenizer, response, args.max_answer_tokens)
        write_jsonl(path, records)
        summary[name] = {
            "prompts": len(records),
            "with_target": sum(1 for record in records if record["teacher_answer"]),
        }
        print(f"{name}: a_T on {summary[name]['with_target']}/{len(records)} prompts -> {path}")
    if args.summary:
        Path(args.summary).write_text(json.dumps({"teacher": args.teacher, **summary}, indent=2))


if __name__ == "__main__":
    main()
