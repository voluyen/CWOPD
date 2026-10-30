"""Training prompts and teacher targets, following the TSD-KD / RIVAD data setup.

TSD-KD (and RIVAD, which reuses its table) trains on the 79,751 UltraInteract prompts released as
`Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen` (only `instruction` is used on-policy) and holds out 1%
(their `train_test_split(test_size=0.01)`), leaving ~78K. Off-policy baselines use the teacher's
responses `Minsang/TSD-KD-Qwen2.5-14B-Instruct-Gen`. The teacher target a_T of CW-OPD is the last
`\\boxed{}` of the teacher response; prompts without one (code, truncated responses) keep uniform
weights, i.e. plain OPD.
"""

import argparse
import json
import random
from pathlib import Path

from answergain.eval.graders.answer_scoring import extract_boxed

PROMPT_DATASET = "Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen"
TEACHER_DATASET = "Minsang/TSD-KD-Qwen2.5-14B-Instruct-Gen"


def chat_prompt_ids(tokenizer, instruction: str) -> list[int]:
    """The instruction as one user turn, as TRL's GKD collator renders it (no added template)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def teacher_answer(tokenizer, response: str | None, max_answer_tokens: int) -> str | None:
    """a_T: content of the last \\boxed{} if short enough to probe, else None (uniform weights)."""
    if not response:
        return None
    answer = extract_boxed(response)
    if not answer:
        return None
    if len(tokenizer(answer + "}", add_special_tokens=False)["input_ids"]) > max_answer_tokens:
        return None
    return answer


def load_rows(name: str, split: str = "train") -> list[dict]:
    from datasets import load_dataset

    if Path(name).is_file():
        return list(load_dataset("json", data_files=name, split="train"))
    return list(load_dataset(name, split=split))


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    tmp.replace(path)


def read_jsonl(path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dataset", default=PROMPT_DATASET, help="HF id or local JSONL with `instruction`")
    parser.add_argument("--teacher-dataset", help="HF id or JSONL with `instruction` and the teacher's `response`")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", required=True, help="student tokenizer (teacher must share it)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-prompt-tokens", type=int, default=512, help="RIVAD Table 5 student prompt limit")
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = load_rows(args.prompt_dataset, args.split)
    teacher = {}
    if args.teacher_dataset:
        for row in load_rows(args.teacher_dataset, args.split):
            teacher.setdefault(row["instruction"], row["response"])

    records, too_long, seen = [], 0, set()
    for row in rows:
        instruction = row["instruction"]
        if instruction in seen:
            continue
        seen.add(instruction)
        prompt_ids = chat_prompt_ids(tokenizer, instruction)
        if len(prompt_ids) > args.max_prompt_tokens:
            too_long += 1
            continue
        record = {"id": len(records), "instruction": instruction, "prompt_ids": prompt_ids}
        if args.teacher_dataset:
            response = teacher.get(instruction)
            record["teacher_response"] = response
            record["teacher_answer"] = teacher_answer(tokenizer, response, args.max_answer_tokens)
        records.append(record)

    random.Random(args.seed).shuffle(records)
    if args.limit:
        records = records[: args.limit]
    n_heldout = max(1, round(len(records) * args.heldout_fraction)) if args.heldout_fraction > 0 else 0
    heldout, train = records[:n_heldout], records[n_heldout:]
    out = Path(args.output_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "heldout.jsonl", heldout)
    with_target = sum(1 for record in train if record.get("teacher_answer"))
    print(
        f"{len(rows)} rows -> {len(train)} train / {len(heldout)} held-out prompts "
        f"(dropped {too_long} over {args.max_prompt_tokens} prompt tokens, {len(rows) - len(seen)} duplicates); "
        f"teacher target a_T on {with_target}/{len(train)} train prompts -> {out}"
    )


if __name__ == "__main__":
    main()
