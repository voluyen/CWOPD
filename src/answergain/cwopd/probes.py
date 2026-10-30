"""Answer probes packed into one teacher forward pass (proposal Sec. 4.3 and 4.5).

Layout of one packed sequence (P prompt tokens, R response tokens, K steps, B = |probe| + N):

    [x][y][probe a_T]_0 [probe a_T]_1 ... [probe a_T]_K

Block k attends to x, to y up to the end of step k (cut_k, cut_0 = 0) and causally to itself; its
position ids continue from that boundary. Rollout tokens never attend to blocks, so the logits on
x and y are those of a plain causal pass. `separate` mode scores the K+1 prefixes as independent
sequences instead and is the reference the packed mode is tested against.
"""

import math
from dataclasses import dataclass

import torch

from answergain.signals.answer_gain import PROBE

PROBE_MODES = ("packed", "separate")
VALUE_FORMS = ("geomean", "raw")


@dataclass
class PackedLayout:
    input_ids: torch.Tensor  # (L,)
    position_ids: torch.Tensor  # (L,)
    allowed: torch.Tensor  # (L, L) bool, True = may attend
    response_logit_rows: torch.Tensor  # rows whose logits predict y_0 .. y_{R-1}
    answer_logit_rows: torch.Tensor  # (K+1, N) rows whose logits predict the answer tokens of block k
    answer_ids: torch.Tensor  # (N,)


def probe_cuts(spans: list[tuple[int, int]]) -> list[int]:
    """Response-relative boundaries after 0, 1, ..., K steps."""
    return [0] + [end for _, end in spans]


def coarsen_spans(spans: list[tuple[int, int]], max_steps: int) -> list[tuple[int, int]]:
    """Merge consecutive steps into at most `max_steps` contiguous groups of near-equal step count.

    Used when x + y + (K+1) probe blocks would exceed the teacher's length budget: the proposal's
    mitigation of probing at coarser boundaries rather than dropping the rollout.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if len(spans) <= max_steps:
        return list(spans)
    groups, n = [], len(spans)
    for g in range(max_steps):
        first, last = g * n // max_steps, (g + 1) * n // max_steps - 1
        groups.append((spans[first][0], spans[last][1]))
    return groups


def fit_spans(spans, n_prompt: int, n_response: int, block: int, max_tokens: int):
    """Spans (possibly coarsened) whose packed sequence fits `max_tokens`, or None if not even one step fits."""
    max_blocks = (max_tokens - n_prompt - n_response) // block
    if max_blocks < 2:  # need V_0 and at least V_K
        return None
    return coarsen_spans(spans, max_blocks - 1)


def packed_layout(
    prompt_ids: list[int], response_ids: list[int], cuts: list[int], probe_ids: list[int], answer_ids: list[int]
) -> PackedLayout:
    P, R, B, N = len(prompt_ids), len(response_ids), len(probe_ids) + len(answer_ids), len(answer_ids)
    if not prompt_ids or not answer_ids or not probe_ids:
        raise ValueError("prompt, probe and answer must be non-empty")
    base = P + R
    L = base + len(cuts) * B
    block = probe_ids + answer_ids

    ids = list(prompt_ids) + list(response_ids) + block * len(cuts)
    positions = list(range(base))
    allowed = torch.zeros(L, L, dtype=torch.bool)
    allowed[:base, :base] = torch.ones(base, base, dtype=torch.bool).tril()
    own = torch.ones(B, B, dtype=torch.bool).tril()
    answer_rows = []
    for k, cut in enumerate(cuts):
        start = base + k * B
        positions.extend(range(P + cut, P + cut + B))
        allowed[start:start + B, : P + cut] = True
        allowed[start:start + B, start:start + B] = own
        # logits at block offset j predict block token j+1; answer token i sits at offset |probe| + i
        answer_rows.append([start + len(probe_ids) - 1 + i for i in range(N)])
    return PackedLayout(
        input_ids=torch.tensor(ids, dtype=torch.long),
        position_ids=torch.tensor(positions, dtype=torch.long),
        allowed=allowed,
        response_logit_rows=torch.arange(P - 1, P + R - 1, dtype=torch.long),
        answer_logit_rows=torch.tensor(answer_rows, dtype=torch.long).view(len(cuts), N),
        answer_ids=torch.tensor(answer_ids, dtype=torch.long),
    )


def additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(1, 1, L, L) additive mask: 0 where allowed, dtype-min elsewhere.

    transformers passes a 4D mask through unchanged; an additive float mask works for both the
    eager and the SDPA attention paths (a bool mask only for SDPA).
    """
    mask = torch.zeros(allowed.shape, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


def _model_dtype(model) -> torch.dtype:
    return next(model.parameters()).dtype


@torch.no_grad()
def score_packed(model, layout: PackedLayout, keep_response_logits: bool = True):
    """One forward pass. Returns (response logits (R, V) or None, answer log-probs (K+1, N))."""
    device = next(model.parameters()).device
    rows = layout.answer_logit_rows.flatten()
    keep = torch.cat([layout.response_logit_rows, rows]) if keep_response_logits else rows
    out = model(
        input_ids=layout.input_ids[None].to(device),
        position_ids=layout.position_ids[None].to(device),
        attention_mask=additive_mask(layout.allowed, _model_dtype(model)).to(device),
        logits_to_keep=keep.to(device),
        use_cache=False,
    )
    logits = out.logits[0]
    n_resp = len(layout.response_logit_rows) if keep_response_logits else 0
    answer_logits = logits[n_resp:].float().view(*layout.answer_logit_rows.shape, -1)
    answer_logps = torch.log_softmax(answer_logits, dim=-1).gather(
        -1, layout.answer_ids.to(device)[None, :, None].expand(answer_logits.shape[0], -1, 1)
    ).squeeze(-1)
    return (logits[:n_resp] if keep_response_logits else None), answer_logps


@torch.no_grad()
def score_separate(model, prompt_ids, response_ids, cuts, probe_ids, answer_ids, keep_response_logits=True):
    """Reference implementation: one causal pass per prefix (plus one for the rollout logits)."""
    device = next(model.parameters()).device
    P, N = len(prompt_ids), len(answer_ids)
    response_logits = None
    if keep_response_logits:
        ids = torch.tensor([list(prompt_ids) + list(response_ids)], device=device)
        response_logits = model(input_ids=ids, use_cache=False).logits[0, P - 1: P + len(response_ids) - 1]
    rows = []
    for cut in cuts:
        ids = list(prompt_ids) + list(response_ids[:cut]) + list(probe_ids) + list(answer_ids)
        logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0, -N - 1: -1].float()
        target = torch.tensor(answer_ids, device=device)
        rows.append(torch.log_softmax(logits, dim=-1).gather(-1, target[:, None]).squeeze(-1))
    return response_logits, torch.stack(rows)


def probe_values(answer_logps: torch.Tensor, form: str = "geomean") -> list[float]:
    """V_k in [0, 1] (Eq. 2): per-token geometric mean, or the raw answer probability (ablation f)."""
    if form not in VALUE_FORMS:
        raise ValueError(f"value form must be one of {VALUE_FORMS}")
    total = answer_logps.double().sum(-1)
    log_v = total / answer_logps.shape[-1] if form == "geomean" else total
    return [math.exp(value) for value in log_v.tolist()]


def encode_probe(tokenizer, answer: str, probe: str = PROBE) -> tuple[list[int], list[int]]:
    """Probe ids and the answer ids scored after it (answer plus the closing brace, as ALERD)."""
    probe_ids = tokenizer(probe, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + "}", add_special_tokens=False)["input_ids"]
    return probe_ids, answer_ids


def score_rollout(model, prompt_ids, response_ids, spans, probe_ids, answer_ids, mode="packed",
                  keep_response_logits=True):
    """(response logits or None, answer log-probs (K+1, N)) for one rollout under `mode`."""
    cuts = probe_cuts(spans)
    if mode == "packed":
        layout = packed_layout(prompt_ids, response_ids, cuts, probe_ids, answer_ids)
        return score_packed(model, layout, keep_response_logits)
    if mode == "separate":
        return score_separate(model, prompt_ids, response_ids, cuts, probe_ids, answer_ids, keep_response_logits)
    raise ValueError(f"probe mode must be one of {PROBE_MODES}")


@torch.no_grad()
def response_logits_only(model, prompt_ids, response_ids) -> torch.Tensor:
    """Plain causal pass, logits predicting the response (OPD without probes)."""
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(response_ids)], device=device)
    keep = torch.arange(len(prompt_ids) - 1, len(prompt_ids) + len(response_ids) - 1, device=device)
    return model(input_ids=ids, logits_to_keep=keep, use_cache=False).logits[0]
