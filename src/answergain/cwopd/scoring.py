"""Per-rollout teacher/student passes shared by the trainer and the E1 pilot."""

from dataclasses import dataclass, field

import torch

from answergain.cwopd.objective import token_divergence, token_entropy
from answergain.cwopd.probes import (
    encode_probe,
    fit_spans,
    probe_values,
    response_logits_only,
    score_rollout,
)
from answergain.cwopd.steps import rollout_step_spans
from answergain.signals.answer_gain import PROBE


@dataclass
class ScoredRollout:
    prompt_ids: list[int]
    response_ids: list[int]
    answer: str | None  # a_T
    spans: list[tuple[int, int]] = field(default_factory=list)
    probed_spans: list[tuple[int, int]] | None = None  # spans the probes used (coarsened if needed)
    teacher_logits: torch.Tensor | None = None  # (R, V), cached between the two phases
    answer_logps: torch.Tensor | None = None  # (K+1, N) under the probe scorer
    n_answer_tokens: int = 0
    packed_tokens: int = 0  # length of the teacher sequence actually run (E5)
    entropy: torch.Tensor | None = None  # (R,) student entropy
    divergence: torch.Tensor | None = None  # (R,) D(p_S, p_T), no grad

    def values(self, form: str = "geomean") -> list[float] | None:
        return None if self.answer_logps is None else probe_values(self.answer_logps, form)


def vocab_size(tokenizer, *models) -> int:
    """Shared support for token-level distillation: padded embedding rows beyond the tokenizer
    (Qwen2.5-1.5B pads to 151936, 14B to 152064) are never real tokens."""
    sizes = [model.get_output_embeddings().weight.shape[0] for model in models]
    return min([len(tokenizer), *sizes])


def teacher_pass(teacher, tokenizer, rollout: ScoredRollout, *, probes: bool, probe_mode: str, max_tokens: int,
                 vocab: int, keep_logits: bool, cache_device: str | torch.device, segmenter: str) -> ScoredRollout:
    """Teacher logits on the rollout and, when asked, the probe values packed into the same pass."""
    if not rollout.spans:
        rollout.spans = rollout_step_spans(tokenizer, rollout.response_ids, segmenter)
    logits = None
    if probes and rollout.answer:
        probe_ids, answer_ids = encode_probe(tokenizer, rollout.answer, PROBE)
        spans = fit_spans(rollout.spans, len(rollout.prompt_ids), len(rollout.response_ids),
                          len(probe_ids) + len(answer_ids), max_tokens)
        if spans is not None:
            logits, rollout.answer_logps = score_rollout(
                teacher, rollout.prompt_ids, rollout.response_ids, spans, probe_ids, answer_ids,
                mode=probe_mode, keep_response_logits=keep_logits,
            )
            rollout.probed_spans = spans
            rollout.n_answer_tokens = len(answer_ids)
            rollout.packed_tokens = (len(rollout.prompt_ids) + len(rollout.response_ids)
                                     + (len(spans) + 1) * (len(probe_ids) + len(answer_ids)))
    if keep_logits and logits is None:
        logits = response_logits_only(teacher, rollout.prompt_ids, rollout.response_ids)
        rollout.packed_tokens = len(rollout.prompt_ids) + len(rollout.response_ids)
    if keep_logits:
        rollout.teacher_logits = logits[:, :vocab].to(cache_device, dtype=torch.bfloat16)
    return rollout


def student_logits(model, prompt_ids: list[int], response_ids: list[int]) -> torch.Tensor:
    """(R, V_model) student logits predicting the response; keeps the graph when grad is enabled.

    The response is the tail of the sequence, so `logits_to_keep=R+1` keeps exactly the rows from
    the last prompt token on; the final row (after the last response token) is dropped.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(response_ids)], device=device)
    return model(input_ids=ids, logits_to_keep=len(response_ids) + 1, use_cache=False).logits[0, :-1]


def student_logits_batch(model, rollouts: list[ScoredRollout]) -> list[torch.Tensor]:
    """`student_logits` for several rollouts in one right-padded forward pass.

    Padding sits after every real token, so causal attention never lets a real token see it. Only the
    last `width - shortest prompt + 1` positions get logits, which covers every row's response.
    """
    device = next(model.parameters()).device
    sequences = [list(r.prompt_ids) + list(r.response_ids) for r in rollouts]
    width = max(len(seq) for seq in sequences)
    ids = torch.zeros((len(sequences), width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq)
        mask[row, : len(seq)] = 1
    keep = width - min(len(r.prompt_ids) for r in rollouts) + 1
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), logits_to_keep=keep,
                   use_cache=False).logits
    offset = width - keep
    return [logits[row, len(r.prompt_ids) - 1 - offset: len(r.prompt_ids) + len(r.response_ids) - 1 - offset]
            for row, r in enumerate(rollouts)]


@torch.no_grad()
def student_stats(model, rollout: ScoredRollout, vocab: int, divergence: str, jsd_beta: float,
                  temperature: float = 1.0) -> ScoredRollout:
    """Student entropy and divergence per token (local signals of E1 and the E4a/selection arms)."""
    logits = student_logits(model, rollout.prompt_ids, rollout.response_ids)[:, :vocab].float() / temperature
    rollout.entropy = token_entropy(logits).cpu()
    if rollout.teacher_logits is not None:
        teacher = rollout.teacher_logits.to(logits.device).float() / temperature
        kind = divergence if divergence in ("rkl", "fkl", "jsd") else "rkl"
        rollout.divergence = token_divergence(logits, teacher, kind, jsd_beta).cpu()
    return rollout


@torch.no_grad()
def student_probe_logps(model, tokenizer, rollout: ScoredRollout, probe_mode: str, max_tokens: int):
    """Ablation E4c: the student scores the same probes on its own rollout."""
    if not rollout.answer:
        return None
    probe_ids, answer_ids = encode_probe(tokenizer, rollout.answer, PROBE)
    spans = rollout.probed_spans or fit_spans(
        rollout.spans, len(rollout.prompt_ids), len(rollout.response_ids), len(probe_ids) + len(answer_ids), max_tokens
    )
    if spans is None:
        return None
    rollout.probed_spans = spans
    rollout.n_answer_tokens = len(answer_ids)
    _, logps = score_rollout(model, rollout.prompt_ids, rollout.response_ids, spans, probe_ids, answer_ids,
                             mode=probe_mode, keep_response_logits=False)
    return logps
