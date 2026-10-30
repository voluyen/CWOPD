"""Per-token divergences, step consequence and the consequence weights (proposal Eqs. 1, 3-5)."""

import math
import random

import torch
import torch.nn.functional as F

from answergain.allocation.weights import step_weights

DIVERGENCES = ("rkl", "fkl", "jsd")
# Step signals in the same weight form w = 1 + s / mean(s) (E4a); `uniform` is plain OPD.
SIGNALS = ("consequence", "permuted", "entropy", "divergence", "uniform")
# Token-selective baselines (SE-KD- and TIP-style): supervise only the top fraction of tokens.
SELECTIONS = ("none", "entropy", "entropy-divergence")
SIGNS = ("abs", "pos")
NORMALIZATIONS = ("running", "rollout", "alerd")


def token_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, kind: str = "rkl",
                     jsd_beta: float = 0.5) -> torch.Tensor:
    """D(p_S, p_T) at every position, (R,). Logits are (R, V) over a shared vocabulary.

    rkl = KL(p_S || p_T) (the estimator OPD uses, Lemma 3); fkl = KL(p_T || p_S);
    jsd = GKD's generalized JSD: beta KL(p_T || M) + (1 - beta) KL(p_S || M), M = beta p_T + (1 - beta) p_S.
    """
    if kind not in DIVERGENCES:
        raise ValueError(f"divergence must be one of {DIVERGENCES}")
    log_s = F.log_softmax(student_logits.float(), dim=-1)
    log_t = F.log_softmax(teacher_logits.float(), dim=-1)
    if kind == "rkl":
        return (log_s.exp() * (log_s - log_t)).sum(-1)
    if kind == "fkl":
        return (log_t.exp() * (log_t - log_s)).sum(-1)
    if not 0.0 < jsd_beta < 1.0:
        raise ValueError("jsd_beta must be in (0, 1)")
    log_m = torch.logsumexp(
        torch.stack([log_t + math.log(jsd_beta), log_s + math.log(1.0 - jsd_beta)]), dim=0
    )
    kl_t = (log_t.exp() * (log_t - log_m)).sum(-1)
    kl_s = (log_s.exp() * (log_s - log_m)).sum(-1)
    return jsd_beta * kl_t + (1.0 - jsd_beta) * kl_s


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=-1)
    return -(log_p.exp() * log_p).sum(-1)


def consequence(values: list[float], sign: str = "abs") -> list[float]:
    """g_k = |V_k - V_{k-1}| (Eq. 3) or max(V_k - V_{k-1}, 0) (ablation E4b), k = 1..K."""
    if sign not in SIGNS:
        raise ValueError(f"sign must be one of {SIGNS}")
    diffs = [after - before for before, after in zip(values, values[1:])]
    return [abs(d) for d in diffs] if sign == "abs" else [max(d, 0.0) for d in diffs]


def step_means(token_values: list[float], spans: list[tuple[int, int]]) -> list[float]:
    return [sum(token_values[start:end]) / (end - start) for start, end in spans]


def permute_within_batch(signals: list[list[float]], rng: random.Random) -> list[list[float]]:
    """E4a control: the batch's step values reassigned to random steps (same distribution, wrong positions)."""
    flat = [value for steps in signals for value in steps]
    rng.shuffle(flat)
    out, index = [], 0
    for steps in signals:
        out.append(flat[index: index + len(steps)])
        index += len(steps)
    return out


class RunningMean:
    """Cumulative token-weighted mean of the step signal over all rollouts seen (g-bar in Eq. 4).

    Updated with the batch before the batch's weights are computed (Algorithm 1, lines 6-7), so the
    first batch initializes it.
    """

    def __init__(self, total: float = 0.0, tokens: float = 0.0):
        self.total, self.tokens = total, tokens

    def update(self, total: float, tokens: float) -> float:
        self.total += total
        self.tokens += tokens
        return self.value

    @property
    def value(self) -> float:
        return self.total / self.tokens if self.tokens > 0 else 0.0

    def state_dict(self) -> dict:
        return {"total": self.total, "tokens": self.tokens}


def batch_signal_mass(signals: list[list[float] | None], lengths: list[list[int]]) -> tuple[float, float]:
    """(sum_k len_k * s_k, sum_k len_k) over the rollouts that carry a signal."""
    total = tokens = 0.0
    for steps, steps_len in zip(signals, lengths):
        if steps is None:
            continue
        total += sum(value * length for value, length in zip(steps, steps_len))
        tokens += sum(steps_len)
    return total, tokens


def step_weights_for(signal: list[float] | None, lengths: list[int], normalization: str, mean: float,
                     floor: bool = True) -> list[float]:
    """Stop-gradient step weights (Eq. 4 and the E4d/E4e ablations).

    running: 1 + s_k / mean(s) with the cumulative mean; rollout: same with this rollout's own
    token-weighted mean; alerd: ALERD's within-trace standardize-clip-exp allocation
    (c = 2, tau = 1, lambda = 0.5). floor=False drops the `1 +` (w proportional to s). A rollout
    without a signal (no teacher answer) gets uniform weights, i.e. plain OPD.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    if signal is None or not signal:
        return [1.0] * len(lengths)
    if normalization == "alerd":
        return step_weights(signal, lengths, temperature=1.0, interpolation=0.5, clip=2.0)
    if normalization == "rollout":
        mass = sum(lengths)
        mean = sum(s * n for s, n in zip(signal, lengths)) / mass if mass else 0.0
    if mean <= 0:
        return [1.0] * len(lengths)  # flat value path everywhere: nothing to reweight
    scaled = [value / mean for value in signal]
    return [1.0 + value for value in scaled] if floor else scaled


def expand_to_tokens(step_values: list[float], spans: list[tuple[int, int]], n_tokens: int) -> list[float]:
    out = [0.0] * n_tokens
    for value, (start, end) in zip(step_values, spans):
        out[start:end] = [value] * (end - start)
    return out


def select_top_fraction(scores: list[torch.Tensor], fraction: float) -> list[torch.Tensor]:
    """0/1 weights keeping the top `fraction` of tokens by score across the given rollouts."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("selection fraction must be in (0, 1]")
    flat = torch.cat([score.float().flatten() for score in scores])
    if flat.numel() == 0:
        return [score.new_zeros(score.shape) for score in scores]
    keep = max(1, math.ceil(fraction * flat.numel()))
    threshold = torch.topk(flat, keep).values.min()
    return [(score >= threshold).float() for score in scores]
