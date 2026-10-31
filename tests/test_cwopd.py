"""CW-OPD: packed probes, rollout step spans, consequence weights, divergences, pilot statistics."""
import math
import random
from pathlib import Path

import pytest
import torch

from answergain.cwopd import lm_eval_suite
from answergain.cwopd.objective import (
    RunningMean,
    consequence,
    expand_to_tokens,
    permute_within_batch,
    select_top_fraction,
    step_weights_for,
    token_divergence,
)
from answergain.cwopd.probes import coarsen_spans, encode_probe, fit_spans, probe_values, score_rollout
from answergain.cwopd.scoring import ScoredRollout, student_logits, student_logits_batch
from answergain.cwopd.stats import jaccard_top, partial, spearman
from answergain.cwopd.steps import decode_with_offsets, rollout_step_spans

TOKENIZER = Path(__file__).resolve().parent / "regression" / "assets" / "tokenizer"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER))


def tiny_model(tokenizer, attn):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                         tie_word_embeddings=False, attn_implementation=attn)
    return Qwen2ForCausalLM(config).eval()


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_packed_probes_match_one_pass_per_prefix(tokenizer, attn):
    model = tiny_model(tokenizer, attn)
    prompt = tokenizer("Solve 2+3.", add_special_tokens=False)["input_ids"]
    response = tokenizer("First add. Then 2 + 3 = 5. So \\boxed{5}.", add_special_tokens=False)["input_ids"]
    spans = rollout_step_spans(tokenizer, response)
    probe_ids, answer_ids = encode_probe(tokenizer, "5")
    packed = score_rollout(model, prompt, response, spans, probe_ids, answer_ids, mode="packed")
    separate = score_rollout(model, prompt, response, spans, probe_ids, answer_ids, mode="separate")
    assert len(spans) == 3 and packed[1].shape == (len(spans) + 1, len(answer_ids))
    torch.testing.assert_close(packed[0], separate[0], atol=1e-5, rtol=1e-5)  # rollout logits untouched
    torch.testing.assert_close(packed[1], separate[1], atol=1e-5, rtol=1e-5)


def test_batched_student_pass_matches_one_pass_per_rollout(tokenizer):
    model = tiny_model(tokenizer, "sdpa")
    texts = [("Solve 2+3.", "So 5."), ("A much longer question about numbers here?", "First, think. Then 7."),
             ("Hi", "Answer: \\boxed{1}.")]
    rollouts = [ScoredRollout(prompt_ids=tokenizer(p, add_special_tokens=False)["input_ids"],
                              response_ids=tokenizer(r, add_special_tokens=False)["input_ids"], answer=None)
                for p, r in texts]
    with torch.no_grad():
        batched = student_logits_batch(model, rollouts)
        for rollout, logits in zip(rollouts, batched):
            single = student_logits(model, rollout.prompt_ids, rollout.response_ids)
            assert logits.shape == single.shape == (len(rollout.response_ids), model.config.vocab_size)
            torch.testing.assert_close(logits, single, atol=1e-5, rtol=1e-5)


def test_rollout_spans_tile_sampled_ids_including_multibyte_and_eos(tokenizer):
    text = "Ünïcode first. Then more — done!"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    decoded, starts = decode_with_offsets(tokenizer, ids)
    assert decoded == text and starts == sorted(starts)
    spans = rollout_step_spans(tokenizer, ids)
    assert spans[0][0] == 0 and spans[-1][1] == len(ids)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_coarsening_keeps_boundaries_and_respects_the_budget():
    spans = [(i, i + 1) for i in range(10)]
    merged = coarsen_spans(spans, 3)
    assert len(merged) == 3 and merged[0][0] == 0 and merged[-1][1] == 10
    assert fit_spans(spans, n_prompt=5, n_response=10, block=4, max_tokens=15 + 4 * 11) == spans
    assert len(fit_spans(spans, 5, 10, 4, 15 + 4 * 4)) == 3  # 4 blocks -> 3 steps
    assert fit_spans(spans, 5, 10, 4, 15 + 4) is None


def test_probe_values_and_consequence():
    logps = torch.log(torch.tensor([[0.25, 0.25], [0.5, 0.5], [0.5, 0.5], [1.0, 1.0]]))
    values = probe_values(logps, "geomean")
    assert values == pytest.approx([0.25, 0.5, 0.5, 1.0])
    assert probe_values(logps, "raw")[0] == pytest.approx(0.0625)
    assert consequence([0.5, 0.2, 0.9], "abs") == pytest.approx([0.3, 0.7])
    assert consequence([0.5, 0.2, 0.9], "pos") == pytest.approx([0.0, 0.7])


def test_weights_floor_running_mean_and_uniform_fallback():
    running = RunningMean()
    mean = running.update(total=2.0, tokens=4.0)
    assert mean == 0.5 and running.update(0.0, 4.0) == 0.25  # cumulative, token-weighted
    assert step_weights_for([0.5, 0.0], [2, 2], "running", 0.5) == [2.0, 1.0]
    assert step_weights_for([0.5, 0.0], [2, 2], "running", 0.5, floor=False) == [1.0, 0.0]
    assert step_weights_for([0.4, 0.0], [1, 3], "rollout", 99.0) == [5.0, 1.0]  # own mean 0.1
    assert step_weights_for(None, [3, 1], "running", 0.5) == [1.0, 1.0]  # no a_T: plain OPD
    assert step_weights_for([0.0, 0.0], [3, 1], "running", 0.0) == [1.0, 1.0]  # flat path
    alerd = step_weights_for([0.1, 0.9], [2, 2], "alerd", 0.0)
    assert sum(w * n for w, n in zip(alerd, [2, 2])) == pytest.approx(4.0)  # ALERD keeps token mass
    assert expand_to_tokens([2.0, 1.0], [(0, 2), (2, 5)], 5) == [2.0, 2.0, 1.0, 1.0, 1.0]


def test_permutation_keeps_the_batch_distribution():
    signals = [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6]]
    permuted = permute_within_batch(signals, random.Random(0))
    assert [len(s) for s in permuted] == [2, 1, 3]
    assert sorted(v for s in permuted for v in s) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_divergences():
    torch.manual_seed(0)
    student, teacher = torch.randn(4, 11), torch.randn(4, 11)
    for kind in ("rkl", "fkl", "jsd"):
        assert torch.allclose(token_divergence(student, student, kind), torch.zeros(4), atol=1e-6)
        assert (token_divergence(student, teacher, kind) > 0).all()
    jsd = token_divergence(student, teacher, "jsd", 0.9)
    assert (jsd <= math.log(2) + 1e-6).all()
    log_s, log_t = student.log_softmax(-1), teacher.log_softmax(-1)
    assert torch.allclose(token_divergence(student, teacher, "rkl"), (log_s.exp() * (log_s - log_t)).sum(-1))


def test_selection_keeps_the_top_fraction():
    scores = [torch.tensor([0.1, 0.9, 0.5]), torch.tensor([0.7, 0.2])]
    kept = select_top_fraction(scores, 0.4)
    assert [k.tolist() for k in kept] == [[0.0, 1.0, 0.0], [1.0, 0.0]]


def test_pilot_statistics():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert partial(0.5, 0.0, 0.0) == pytest.approx(0.5)
    assert jaccard_top([5, 1, 1, 1, 1, 1, 1, 1, 1, 1], [5, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0.1) == 1.0


def test_eval_suite_groups_tasks_by_shot_count():
    args = lm_eval_suite.build_parser().parse_args(
        ["--model", "m", "--tag", "t", "--results-dir", "r", "--lm-eval-python", "py"])
    plan = {cmd[cmd.index("--num_fewshot") + 1]: cmd for cmd, _ in lm_eval_suite.commands(args)}
    assert set(plan) == {"0", "3", "4", "5"}  # TSD-KD Appendix A.10 shots
    assert "gsm8k,gsm_plus,mmlu_stem,mmlu_pro_math" in plan["5"]
    assert "--confirm_run_unsafe_code" in plan["0"] and "--confirm_run_unsafe_code" not in plan["5"]
    metrics = {"gsm8k": {"exact_match,strict-match": 0.5}}
    assert lm_eval_suite.primary("gsm8k", metrics) == (0.5, "exact_match,strict-match")
