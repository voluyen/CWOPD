"""On-policy distillation trainer: OPD / GKD / off-policy KD baselines, consequence-weighted OPD and
the E4 ablations, in one loop so every arm shares prompts, rollout budget, teacher and schedule.

One optimizer step (proposal Algorithm 1), on every rank:
  A. rollouts y ~ p_S(. | x) for the local prompts (or the teacher's responses for the off-policy
     baselines); one teacher pass per rollout gives the logits on y and, with consequence
     weighting, the probe values V_0..V_K packed into the same pass; g_k = |V_k - V_{k-1}|;
     g-bar is updated with the global batch; w_k = 1 + g_k / g-bar (stop-gradient).
  B. student forward/backward in micro-batches of rollouts on sum_t w_t D_t / sum_t w_t, both sums
     over the global batch, so the supervision budget is fixed per batch (Eq. 5).

Rollouts come from HF generate or, on GPU, from a vLLM engine in the same process that receives the
student's weights before every step (answergain.cwopd.vllm_rollouts).

Defaults follow TSD-KD's released setup (RIVAD Table 2 reuses its numbers) and RIVAD Table 5:
batch 128 prompts, lr 5e-6 cosine, warmup 0.1, 3 epochs, KD temperature 1, 512 prompt + 512 new
tokens, teacher limit 2560, weight decay 0.01, clip 1.0, seed 10.
"""

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from answergain.cwopd.data import read_jsonl
from answergain.cwopd.objective import (
    DIVERGENCES,
    NORMALIZATIONS,
    SELECTIONS,
    SIGNALS,
    SIGNS,
    RunningMean,
    batch_signal_mass,
    consequence,
    expand_to_tokens,
    permute_within_batch,
    select_top_fraction,
    step_means,
    step_weights_for,
    token_divergence,
    token_entropy,
)
from answergain.cwopd.probes import PROBE_MODES, VALUE_FORMS, response_logits_only
from answergain.cwopd.scoring import (
    ScoredRollout,
    student_logits_batch,
    student_probe_logps,
    student_stats,
    teacher_pass,
    vocab_size,
)
from answergain.cwopd.steps import rollout_step_spans, step_texts
from answergain.eval.graders.answer_scoring import extract_boxed, math_answers_equal
from answergain.segment import SEGMENTERS

OBJECTIVES = (*DIVERGENCES, "ce")  # ce = SFT on the response (sequence-level KD with teacher responses)
PROBE_SIGNALS = ("consequence", "permuted")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--data-path", required=True, help="train.jsonl from answergain.cwopd.data")
    parser.add_argument("--output-dir", required=True)
    # schedule (TSD-KD Table 10 / RIVAD Table 5)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (smoke runs)")
    parser.add_argument("--prompts-per-step", type=int, default=128, help="global batch of prompts per update")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--limit", type=int, help="use only the first N training prompts")
    # rollouts
    parser.add_argument("--rollout-source", choices=("student", "teacher"), default="student",
                        help="teacher = off-policy baselines on the teacher's responses (Supervised KD, SeqKD)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--generation-batch-size", type=int, default=32, help="HF generate only")
    parser.add_argument("--rollout-engine", choices=("hf", "vllm"), default="hf",
                        help="vllm: colocated engine synced to the student every step (one rank only)")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2,
                        help="share of the GPU the engine reserves (weights + KV cache)")
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--train-micro-batch", type=int, default=1,
                        help="rollouts per student forward/backward in phase B (gradient sum is unchanged)")
    # objective
    parser.add_argument("--objective", choices=OBJECTIVES, default="rkl")
    parser.add_argument("--jsd-beta", type=float, default=0.9, help="GKD's beta (TSD-KD and RIVAD: 0.9)")
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    # consequence weighting and its ablations (E4)
    parser.add_argument("--signal", choices=SIGNALS, default="uniform")
    parser.add_argument("--sign", choices=SIGNS, default="abs")
    parser.add_argument("--scorer", choices=("teacher", "student"), default="teacher")
    parser.add_argument("--normalization", choices=NORMALIZATIONS, default="running")
    parser.add_argument("--floor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value", choices=VALUE_FORMS, default="geomean")
    parser.add_argument("--segmenter", choices=sorted(SEGMENTERS), default="sentence")
    parser.add_argument("--probe-mode", choices=PROBE_MODES, default="packed")
    parser.add_argument("--teacher-max-tokens", type=int, default=2560, help="RIVAD Table 5 teacher total limit")
    # token-selective baselines and TSD-KD-style entropy minimization (E4g)
    parser.add_argument("--selection", choices=SELECTIONS, default="none")
    parser.add_argument("--selection-fraction", type=float, default=0.1)
    parser.add_argument("--entropy-min-coef", type=float, default=0.0)
    parser.add_argument("--entropy-min-fraction", type=float, default=0.1)
    # memory / precision
    parser.add_argument("--student-dtype", choices=("float32", "bfloat16"), default="float32",
                        help="float32 master weights with bf16 autocast; bfloat16 = TSD-KD's pure-bf16 setting")
    parser.add_argument("--attn-implementation", default="sdpa", help="eager for Gemma2 (attention softcapping)")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-cache", choices=("cpu", "gpu", "recompute"), default="cpu",
                        help="where teacher logits wait between the scoring and the gradient phase; "
                        "cpu holds ~R x V bf16 per rollout in host RAM (Qwen2.5: ~0.15 GB at R=512), "
                        "recompute runs a second plain teacher pass instead (no host memory, more compute)")
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # bookkeeping
    parser.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dump-rollouts-every", type=int, default=50, help="E6 rollout dumps (0 = never)")
    parser.add_argument("--log-local-signals", action=argparse.BooleanOptionalAction, default=False,
                        help="also compute student entropy / divergence on every rollout for logging")
    return parser


# --------------------------------------------------------------------------------------- setup


def setup_distributed() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world, device


def multi_rank() -> bool:
    # an in-process vLLM engine initializes a one-rank process group of its own; nothing to reduce there
    return dist.is_initialized() and dist.get_world_size() > 1


def all_sum(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if multi_rank():
        dist.all_reduce(tensor)
    return tensor.tolist()


def barrier() -> None:
    if multi_rank():
        dist.barrier()


def stop_ids(tokenizer, generation_config) -> list[int]:
    """Generation stop ids: the model's eos ids plus the chat end-of-turn token."""
    ids = generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    for marker in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>"):
        token = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token, int) and token != tokenizer.unk_token_id and token not in ids:
            ids.append(token)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


def turn_end_id(tokenizer, ids: list[int]) -> int:
    """Token that closes an assistant turn, appended to teacher responses that were not truncated."""
    for marker in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>"):
        token = tokenizer.convert_tokens_to_ids(marker)
        if token in ids:
            return token
    return ids[0]


def unwrap(model):
    return model.module if hasattr(model, "module") else model


# ------------------------------------------------------------------------------------- rollouts


@torch.no_grad()
def generate_rollouts(model, tokenizer, prompts, args, stops, vocab, device, autocast):
    """Sampled responses (ids up to and including the first stop token) and finish reasons."""
    module = unwrap(model)
    was_training = module.training
    module.eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stops[0]
    out_vocab = module.get_output_embeddings().weight.shape[0]
    suppress = list(range(vocab, out_vocab)) or None  # padded embedding rows are not tokens
    results = []
    for start in range(0, len(prompts), args.generation_batch_size):
        chunk = prompts[start: start + args.generation_batch_size]
        width = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, prompt in enumerate(chunk):  # left padding
            ids[row, width - len(prompt):] = torch.tensor(prompt)
            mask[row, width - len(prompt):] = 1
        # every field set explicitly: generate() would otherwise take the checkpoint's generation_config
        # defaults (Qwen2.5-Instruct ships repetition_penalty > 1), and rollouts must be plain samples of p_S
        sampling = (
            {"do_sample": True, "temperature": args.rollout_temperature, "top_p": args.rollout_top_p, "top_k": 0}
            if args.rollout_temperature > 0 else {"do_sample": False}
        )
        sampling["repetition_penalty"] = 1.0
        with autocast():
            generated = module.generate(
                input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=args.max_new_tokens,
                eos_token_id=stops, pad_token_id=pad, suppress_tokens=suppress, use_cache=True, **sampling,
            )
        for row in generated[:, width:].tolist():
            response, finish = [], "length"
            for token in row:
                response.append(token)
                if token in stops:
                    finish = "stop"
                    break
            results.append((response, finish))
    module.train(was_training)
    return results


def teacher_rollouts(tokenizer, records, args, end_id):
    results = []
    for record in records:
        ids = tokenizer(record.get("teacher_response") or "", add_special_tokens=False)["input_ids"]
        if len(ids) >= args.max_new_tokens:
            results.append((ids[: args.max_new_tokens], "length"))
        else:
            results.append((ids + [end_id], "stop") if ids else ([], "empty"))
    return results


# ------------------------------------------------------------------------------------- one step


def train_step(step, batch, model, teacher, tokenizer, optimizer, scheduler, running, args, ctx):
    device, world, vocab = ctx["device"], ctx["world"], ctx["vocab"]
    autocast = ctx["autocast"]
    module = unwrap(model)
    timings = {}
    torch.manual_seed(args.seed * 100_003 + step * 1_009 + ctx["rank"])  # rollouts reproducible on resume
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- phase A: rollouts, teacher pass with packed probes, step signals
    clock = time.perf_counter()
    prompts = [record["prompt_ids"] for record in batch]
    if args.rollout_source == "student" and ctx["engine"] is not None:
        ctx["engine"].load(module)
        generations = ctx["engine"].generate(prompts, seed=args.seed * 100_003 + step * 1_009)
    elif args.rollout_source == "student":
        generations = generate_rollouts(model, tokenizer, prompts, args, ctx["stops"], vocab, device, autocast)
    else:
        generations = teacher_rollouts(tokenizer, batch, args, ctx["end_id"])
    timings["generate_s"] = time.perf_counter() - clock

    clock = time.perf_counter()
    rollouts, finishes, sources = [], [], []
    for record, (response, finish) in zip(batch, generations):
        if not response:
            continue
        rollouts.append(ScoredRollout(prompt_ids=record["prompt_ids"], response_ids=response,
                                      answer=record.get("teacher_answer")))
        finishes.append(finish)
        sources.append(record)

    probe_teacher = args.signal in PROBE_SIGNALS and args.scorer == "teacher"
    probe_student = args.signal in PROBE_SIGNALS and args.scorer == "student"
    keep_teacher_logits = (
        args.objective != "ce" or args.signal == "divergence" or args.selection == "entropy-divergence"
    )
    need_stats = args.signal in ("entropy", "divergence") or args.selection != "none" or args.log_local_signals
    cache = device if args.teacher_cache == "gpu" else "cpu"
    # recompute: phase A keeps teacher logits only where a local signal needs them, then drops them
    keep_in_phase_a = keep_teacher_logits and (
        args.teacher_cache != "recompute" or args.signal == "divergence" or args.selection == "entropy-divergence"
        or args.log_local_signals
    )
    module.eval()
    for rollout in rollouts:
        if teacher is not None:
            teacher_pass(teacher, tokenizer, rollout, probes=probe_teacher, probe_mode=args.probe_mode,
                         max_tokens=args.teacher_max_tokens, vocab=vocab, keep_logits=keep_in_phase_a,
                         cache_device=cache, segmenter=args.segmenter)
        else:
            rollout.spans = rollout_step_spans(tokenizer, rollout.response_ids, args.segmenter)
        with autocast():
            if probe_student:
                rollout.answer_logps = student_probe_logps(module, tokenizer, rollout, args.probe_mode,
                                                           args.teacher_max_tokens)
            if need_stats:
                student_stats(module, rollout, vocab, args.objective, args.jsd_beta, args.kd_temperature)
        if args.teacher_cache == "recompute":
            rollout.teacher_logits = None
    module.train()
    timings["score_s"] = time.perf_counter() - clock

    weight_spans, signals = [], []
    for rollout in rollouts:
        spans = rollout.spans
        signal = None
        if args.signal in PROBE_SIGNALS and rollout.answer_logps is not None:
            spans = rollout.probed_spans
            signal = consequence(rollout.values(args.value), args.sign)
        elif args.signal == "entropy":
            signal = step_means(rollout.entropy.tolist(), spans)
        elif args.signal == "divergence" and rollout.divergence is not None:
            signal = step_means(rollout.divergence.tolist(), spans)
        weight_spans.append(spans)
        signals.append(signal)
    if args.signal == "permuted":
        rng = random.Random(args.seed * 7 + step * 31 + ctx["rank"])
        present = [i for i, s in enumerate(signals) if s is not None]
        for i, permuted in zip(present, permute_within_batch([signals[i] for i in present], rng)):
            signals[i] = permuted

    lengths = [[end - start for start, end in spans] for spans in weight_spans]
    total, tokens = all_sum(list(batch_signal_mass(signals, lengths)), device)
    mean = running.update(total, tokens) if args.signal != "uniform" else 0.0

    step_w, token_w = [], []
    for rollout, signal, spans, lens in zip(rollouts, signals, weight_spans, lengths):
        weights = step_weights_for(signal, lens, args.normalization, mean, args.floor)
        step_w.append(weights)
        token_w.append(torch.tensor(expand_to_tokens(weights, spans, len(rollout.response_ids))))
    if args.selection != "none":
        scores = [r.entropy if args.selection == "entropy" else r.entropy * r.divergence for r in rollouts]
        for i, keep in enumerate(select_top_fraction(scores, args.selection_fraction) if scores else []):
            token_w[i] = token_w[i] * keep
    weight_total, n_global = all_sum([float(sum(w.sum() for w in token_w)), float(len(rollouts))], device)

    # ---- phase B: weighted divergence on the student's own tokens
    clock = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    sums = {"div": 0.0, "loss": 0.0, "em": 0.0}
    # micro-batches of similar length waste little padding; the order does not change the gradient sum
    order = sorted(range(len(rollouts)), key=lambda i: len(rollouts[i].prompt_ids) + len(rollouts[i].response_ids))
    chunks = [order[i: i + args.train_micro_batch] for i in range(0, len(order), args.train_micro_batch)]
    for chunk_index, chunk in enumerate(chunks):
        # gradients are all-reduced only on the last backward of the step
        last = world == 1 or chunk_index == len(chunks) - 1
        with contextlib.nullcontext() if last else model.no_sync():
            with autocast():
                chunk_logits = student_logits_batch(model, [rollouts[i] for i in chunk])
            loss = 0.0
            for index, logits in zip(chunk, chunk_logits):
                rollout = rollouts[index]
                logits = logits[:, :vocab].float() / args.kd_temperature
                weights = token_w[index].to(device)
                if args.objective == "ce":
                    targets = torch.tensor(rollout.response_ids, device=device)
                    per_token = F.cross_entropy(logits, targets, reduction="none")
                else:
                    cached = rollout.teacher_logits
                    if cached is None:  # --teacher-cache recompute
                        cached = response_logits_only(teacher, rollout.prompt_ids, rollout.response_ids)[:, :vocab]
                    teacher_logits = cached.to(device).float() / args.kd_temperature
                    per_token = token_divergence(logits, teacher_logits, args.objective, args.jsd_beta)
                # DDP averages gradients over ranks; scale so the result is the global sum / global weight.
                row_loss = (weights * per_token).sum() / max(weight_total, 1e-12) * world
                if args.entropy_min_coef > 0:
                    entropy = token_entropy(logits)
                    keep = select_top_fraction([entropy.detach()], args.entropy_min_fraction)[0]
                    em = (keep * entropy).sum() / keep.sum().clamp_min(1.0)
                    row_loss = row_loss + args.entropy_min_coef * em / max(n_global, 1.0) * world
                    sums["em"] += float(em)
                loss = loss + row_loss
                sums["div"] += float(per_token.detach().sum())
                sums["loss"] += float(row_loss.detach()) / world
            loss.backward()
    if not rollouts:  # keep DDP's collective calls matched on a rank with nothing to train on
        (sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    timings["train_s"] = time.perf_counter() - clock

    # ---- metrics (global sums)
    fidelity_hits = fidelity_total = 0
    texts = []
    for rollout in rollouts:
        text = tokenizer.decode(rollout.response_ids, skip_special_tokens=True)
        texts.append(text)
        if rollout.answer:
            fidelity_total += 1
            fidelity_hits += int(math_answers_equal(extract_boxed(text), rollout.answer))
    with_signal = [s for s in signals if s is not None]
    response_tokens = sum(len(r.response_ids) for r in rollouts)
    base_tokens = sum(len(r.prompt_ids) + len(r.response_ids) for r in rollouts if r.packed_tokens)
    vector = [
        len(rollouts), response_tokens, sum(len(r.spans) for r in rollouts), sum(1 for r in rollouts if r.answer),
        len(with_signal), sum(sum(s) for s in with_signal), sum(len(s) for s in with_signal),
        fidelity_hits, fidelity_total, sums["div"], sums["loss"], sums["em"],
        sum(1 for f in finishes if f == "length"), sum(r.packed_tokens for r in rollouts), base_tokens,
        sum(1 for r in rollouts if r.probed_spans is not None and len(r.probed_spans) < len(r.spans)),
    ]
    (n, resp_tok, n_steps, n_target, n_sig, sum_g, sum_k, fid_hit, fid_tot, div, loss_sum, em, truncated,
     packed, base, coarsened) = all_sum(vector, device)
    metrics = {
        "step": step + 1,
        "lr": scheduler.get_last_lr()[0],
        "loss": loss_sum,
        "mean_divergence": div / max(resp_tok, 1),
        "entropy_min": em / max(n, 1),
        "grad_norm": float(grad_norm),
        "rollouts": int(n),
        "mean_response_tokens": resp_tok / max(n, 1),
        "truncated_rate": truncated / max(n, 1),
        "mean_steps": n_steps / max(n, 1),
        "target_rate": n_target / max(n, 1),
        "signal_rate": n_sig / max(n, 1),
        "g_bar": running.value,
        # Corollary 1: E[sum_k g_k] / E[K], small when only a few steps move the value
        "sparsity_ratio": (sum_g / n_sig) / (sum_k / n_sig) if n_sig and sum_k else None,
        "fidelity": fid_hit / fid_tot if fid_tot else None,  # student's boxed answer == a_T
        "probe_overhead_tokens": (packed - base) / base if base else None,
        "coarsened_rollouts": int(coarsened),
        **timings,
    }
    if device.type == "cuda":
        metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 2**30

    dump = None
    if args.dump_rollouts_every and (step + 1) % args.dump_rollouts_every == 0:
        dump = []
        for record, rollout, text, signal, spans, weights, finish in zip(
            sources, rollouts, texts, signals, weight_spans, step_w, finishes
        ):
            dump.append({
                "step": step + 1, "id": record["id"], "teacher_answer": rollout.answer,
                "student_answer": extract_boxed(text), "finish": finish,
                "fidelity": (math_answers_equal(extract_boxed(text), rollout.answer) if rollout.answer else None),
                "steps": step_texts(tokenizer, rollout.response_ids, spans),
                "step_tokens": [end - start for start, end in spans],
                "values": rollout.values(args.value), "signal": signal, "weights": weights,
            })
    for rollout in rollouts:  # free cached logits before the next step
        rollout.teacher_logits = None
    return metrics, dump


# ---------------------------------------------------------------------------------- checkpoints


def trainable_state(model) -> dict:
    return {name: p.detach().cpu() for name, p in unwrap(model).named_parameters() if p.requires_grad}


def save_model(model, tokenizer, path: Path, use_lora: bool, merge: bool) -> None:
    module = unwrap(model)
    path.mkdir(parents=True, exist_ok=True)
    if use_lora and merge:
        merged = module.merge_and_unload()
        merged.save_pretrained(path, state_dict={k: v.to(torch.bfloat16) for k, v in merged.state_dict().items()})
    elif use_lora:
        module.save_pretrained(path)
    else:
        module.save_pretrained(path, state_dict={k: v.to(torch.bfloat16) for k, v in module.state_dict().items()})
    tokenizer.save_pretrained(path)


def save_checkpoint(model, tokenizer, optimizer, scheduler, running, step, args):
    path = Path(args.output_dir) / f"checkpoint-{step}"
    save_model(model, tokenizer, path, args.use_lora, merge=False)
    torch.save({
        "step": step, "trainable": trainable_state(model), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "running": running.state_dict(),
    }, path / "trainer_state.pt")
    checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in checkpoints[: -args.save_total_limit] if args.save_total_limit > 0 else []:
        shutil.rmtree(old, ignore_errors=True)


def latest_checkpoint(output_dir: str) -> Path | None:
    found = [p for p in Path(output_dir).glob("checkpoint-*") if (p / "trainer_state.pt").exists()]
    return max(found, key=lambda p: int(p.name.split("-")[1])) if found else None


# ------------------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

    if args.objective != "ce" and not args.teacher:
        raise SystemExit(f"--objective {args.objective} needs --teacher")
    if args.signal in PROBE_SIGNALS + ("divergence",) and not args.teacher:
        raise SystemExit(f"--signal {args.signal} needs --teacher")

    rank, world, device = setup_distributed()
    if args.prompts_per_step % world:
        raise SystemExit(f"--prompts-per-step {args.prompts_per_step} is not divisible by {world} ranks")
    set_seed(args.seed)  # identical initialization (LoRA adapters included) on every rank

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    records = read_jsonl(args.data_path)[: args.limit]
    use_vllm = args.rollout_engine == "vllm" and args.rollout_source == "student"
    if use_vllm and (world > 1 or device.type != "cuda" or args.use_lora):
        raise SystemExit("--rollout-engine vllm needs one CUDA rank and full fine-tuning (no LoRA)")
    engine = None
    if use_vllm:
        from transformers import AutoConfig, GenerationConfig

        from answergain.cwopd.vllm_rollouts import VLLMRollouts

        # built before the student and teacher take the GPU, so vLLM's memory profiling sees a clean device
        engine = VLLMRollouts(
            args.student, stops=stop_ids(tokenizer, GenerationConfig.from_pretrained(args.student)),
            vocab=len(tokenizer), out_vocab=AutoConfig.from_pretrained(args.student).vocab_size,
            max_new_tokens=args.max_new_tokens, temperature=args.rollout_temperature, top_p=args.rollout_top_p,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization, max_model_len=args.vllm_max_model_len,
            seed=args.seed,
        )
    dtype = torch.float32 if args.student_dtype == "float32" or device.type == "cpu" else torch.bfloat16
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=dtype, attn_implementation=args.attn_implementation
    ).to(device)
    student.config.use_cache = False
    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        student = get_peft_model(student, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
    if args.gradient_checkpointing and device.type == "cuda":
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.use_lora:
            student.enable_input_require_grads()

    teacher = None
    if args.teacher:
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
        probe = "Therefore, the final answer is \\boxed{12}."
        if teacher_tokenizer(probe)["input_ids"] != tokenizer(probe)["input_ids"]:
            raise SystemExit("teacher and student tokenizers differ; token-level distillation needs a shared one")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation=args.attn_implementation,
        ).to(device).eval()
        teacher.requires_grad_(False)
    vocab = vocab_size(tokenizer, student, *([teacher] if teacher is not None else []))

    model = student
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[device.index] if device.type == "cuda" else None
        )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = len(records) // args.prompts_per_step
    if steps_per_epoch == 0:
        raise SystemExit(f"{len(records)} prompts < one batch of {args.prompts_per_step}")
    total_steps = math.ceil(args.epochs * steps_per_epoch)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    running = RunningMean()

    start = 0
    resume_from = latest_checkpoint(args.output_dir) if args.resume else None
    if resume_from:
        state = torch.load(resume_from / "trainer_state.pt", map_location="cpu", weights_only=False)
        named = dict(unwrap(model).named_parameters())
        with torch.no_grad():
            for name, value in state["trainable"].items():
                named[name].copy_(value.to(named[name].device, named[name].dtype))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        running = RunningMean(**state["running"])
        start = state["step"]
        if rank == 0:
            print(f"resumed from {resume_from} at step {start}")

    autocast_on = device.type == "cuda" and dtype == torch.float32

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_on)

    stops = stop_ids(tokenizer, unwrap(model).generation_config)
    ctx = {"device": device, "world": world, "rank": rank, "vocab": vocab, "autocast": autocast,
           "stops": stops, "engine": engine, "end_id": turn_end_id(tokenizer, stops)}
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "train-args.json").write_text(json.dumps({**vars(args), "world_size": world}, indent=2))
        print(f"{len(records)} prompts, {steps_per_epoch} steps/epoch, {total_steps} steps on {world} rank(s); "
              f"objective={args.objective} signal={args.signal} selection={args.selection} vocab={vocab}")
    metrics_log = out / "metrics.jsonl"
    started = time.perf_counter()
    for step in range(start, total_steps):
        epoch, inner = divmod(step, steps_per_epoch)
        order = list(range(len(records)))
        random.Random(args.seed + epoch).shuffle(order)
        chosen = order[inner * args.prompts_per_step: (inner + 1) * args.prompts_per_step][rank::world]
        metrics, dump = train_step(step, [records[i] for i in chosen], model, teacher, tokenizer, optimizer,
                                   scheduler, running, args, ctx)
        metrics["epoch"] = epoch + (inner + 1) / steps_per_epoch
        if rank == 0:
            with metrics_log.open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()),
                  flush=True)
            if dump:
                dump_dir = out / "rollouts"
                dump_dir.mkdir(exist_ok=True)
                with (dump_dir / f"step-{step + 1:06d}.jsonl").open("w") as handle:
                    for row in dump:
                        handle.write(json.dumps(row) + "\n")
        last = step + 1 == total_steps
        at_epoch = args.save_every_epoch and inner + 1 == steps_per_epoch
        at_steps = args.save_steps and (step + 1) % args.save_steps == 0
        if not last and (at_epoch or at_steps):
            if rank == 0:
                save_checkpoint(model, tokenizer, optimizer, scheduler, running, step + 1, args)
            barrier()

    if rank == 0:
        save_model(model, tokenizer, out, args.use_lora, merge=True)
        (out / "run-summary.json").write_text(json.dumps({
            "data_path": args.data_path, "prompts": len(records), "steps": total_steps, "world_size": world,
            "seed": args.seed, "objective": args.objective, "signal": args.signal, "g_bar": running.value,
            "train_runtime_s": time.perf_counter() - started,
        }, indent=2))
    barrier()
    if world > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
