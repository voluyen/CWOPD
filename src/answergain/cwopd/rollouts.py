"""Student rollouts with vLLM on held-out prompts, for the pilots that run before training (E1, E2, E5)."""

import argparse
from pathlib import Path

from answergain.cwopd.data import read_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="heldout.jsonl (never trained on)")
    parser.add_argument("--model", required=True, help="the initial student")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-prompts", type=int, default=300)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--only-with-target", action=argparse.BooleanOptionalAction, default=True,
                        help="keep prompts whose teacher response has a probe target a_T")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0, help="training rollouts sample at 1.0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer, GenerationConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from answergain.cwopd.train import stop_ids

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stops = stop_ids(tokenizer, GenerationConfig.from_pretrained(args.model))
    records = read_jsonl(args.data_path)
    if args.only_with_target:
        records = [record for record in records if record.get("teacher_answer")]
    records = records[: args.n_prompts]
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, seed=args.seed)
    sampling = SamplingParams(n=args.samples_per_prompt, temperature=args.temperature, top_p=1.0,
                              max_tokens=args.max_tokens, stop_token_ids=stops, seed=args.seed)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in records], sampling)
    rollouts = []
    for record, output in zip(records, outputs):
        for sample, completion in enumerate(output.outputs):
            ids = list(completion.token_ids)
            if completion.finish_reason == "stop" and (not ids or ids[-1] not in stops):
                ids.append(stops[0])  # keep the turn-closing token, as the trainer's rollouts do
            rollouts.append({
                "id": f"{record['id']}-{sample}", "prompt_id": record["id"], "instruction": record["instruction"],
                "prompt_ids": record["prompt_ids"], "response_ids": ids, "finish": completion.finish_reason,
                "teacher_answer": record.get("teacher_answer"),
            })
    write_jsonl(Path(args.output), rollouts)
    print(f"{len(rollouts)} rollouts over {len(records)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
