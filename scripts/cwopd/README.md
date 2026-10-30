# CW-OPD: consequence-weighted on-policy distillation

Code for `proposal.pdf`, run under the experimental setup of RIVAD (ICLR 2027 submission), which
reuses the TSD-KD setup and numbers. Family `cwopd`, configs under `training_configs/cwopd/`,
code in `src/answergain/cwopd/`.

## Setup on the GPU server (one 60 GB GPU)

```bash
scripts/cwopd/setup.sh          # uv sync + .venv-eval with lm-evaluation-harness 0.4.13 (vLLM)
huggingface-cli login           # Gemma2 checkpoints are gated
scripts/cwopd/smoke.sh          # 2 training steps of CW-OPD on 16 prompts + a tiny eval, ~15 min
```

Local model copies: `QWEN25_14B`, `QWEN25_1P5B`, `GEMMA2_9B`, `GEMMA2_2B` env vars override the HF ids;
`LMEVAL_PYTHON` overrides `.venv-eval/bin/python`.

## Phases (proposal Sec. 9)

| Script | What | Proposal |
|---|---|---|
| `phase0_calibrate.sh` | evaluate untrained teachers/students, compare with RIVAD Table 2 / 6 rows | protocol check |
| `phase0_pilot.sh` | data, a_T, 300 student rollouts, E1 distinctness, E2 Monte Carlo validity | RQ1, RQ2, decision on H1 |
| `phase1_cost.sh` | teacher time / memory with and without packed probes against K | E5, RQ3 |
| `phase2_main.sh` | SeqKD, Supervised KD, OPD-RKL, GKD-JSD(0.9), entropy- and TIP-selective OPD, CW-OPD, CW-GKD | E3 |
| `phase3_ablations.sh` | E4 (a)-(g) + paragraph segmentation, Gemma2 pair, E6 analysis | E4, E6 |

Every script forwards overrides: `scripts/cwopd/phase2_main.sh stages.train.args.teacher-cache=recompute`.
Extra seeds: `SEEDS="11 12" scripts/cwopd/phase2_main.sh` (runs are suffixed `-s11`, `-s12`).
Single runs: `cwopd run qwen25-1.5b/cw-opd`, `cwopd show ...`, `--dry-run`, `--stages train,eval`.

Outputs: `data/cwopd/<pair>/`, `checkpoints/cwopd/<run>/` (`metrics.jsonl` per step, `rollouts/` dumps),
`results/cwopd/<pair>/<run>/summary.json`, `results/cwopd/<pair>/table.md` (with the published rows),
`results/cwopd/<pair>/pilot/{report.md,cost.json}`.

## Setup taken from RIVAD / TSD-KD

| | Value | Source |
|---|---|---|
| Pairs | Qwen2.5-14B-Instruct -> 1.5B-Instruct; Gemma2-9B-it -> 2B-it | RIVAD 4.1 |
| Prompts | `Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen` (79,751 UltraInteract), 1% held out -> ~78K | TSD-KD `train.py` |
| Teacher responses / a_T | `Minsang/TSD-KD-Qwen2.5-14B-Instruct-Gen`; Gemma2-9B generated with vLLM (greedy) | TSD-KD README |
| Schedule | 128 prompts/update, 3 epochs, cosine, warmup 0.1, lr 5e-6 (Qwen) / 1e-7 (Gemma), KD temp 1 | TSD-KD Table 10 |
| Lengths | prompt <= 512, 512 new tokens, teacher sequence <= 2560 | RIVAD Table 5, B.2 |
| Optimizer | AdamW, weight decay 0.01, clip 1.0, seed 10 | RIVAD Table 5 |
| GKD | on-policy, JSD beta = 0.9 | TSD-KD / RIVAD |
| Eval | lm-eval: GSM8K 5-shot, GSM-Plus 5, MATH (minerva) 4, MMLU-Pro-Math 5, MMLU-STEM 5, SciQ 0, MBPP 0, BBH-CoT 3; IFEval extra | TSD-KD A.10 |

## Deviations and open choices (check before reporting)

- **One GPU instead of 4 (RIVAD) / 8 (TSD-KD).** The global batch of 128 is kept, rollouts are processed
  one at a time. The student trains in bf16 weights, as TSD-KD's script does. fp32 master weights do not
  fit next to the 14B teacher in 60 GB.
- **Wall-clock.** One update means 128 HF-generate rollouts, 128 teacher passes and 128 student passes. Read
  `generate_s + score_s + train_s` from the smoke run and multiply by ~617 steps/epoch x 3 epochs. Expect
  days per arm, so prioritise: calibration, pilot, then `cw-opd`, `opd-rkl` and `gkd-jsd`. Run the
  ablations with `stages.train.args.epochs=1` if budget forces it, and say so in the paper.
- **Host RAM.** `teacher-cache: cpu` keeps a batch of teacher logits in RAM, about 20 GB (Qwen) or 34 GB
  (Gemma) per update. With less RAM, pass `stages.train.args.teacher-cache=recompute`, which costs one extra
  teacher pass per rollout.
- **Eval protocol.** The metric key per benchmark (e.g. GSM8K flexible vs strict extract) and chat template
  (off by default) are not stated by TSD-KD. Fix them with `phase0_calibrate.sh` so the base rows match
  57.1 / 38.8 / 16.9 / ... before comparing.
- **Baselines not reimplemented.** MiniLLM, DistiLLM, Speculative KD, TSD-KD, CSD and AMiD come from the
  published rows in `answergain.cwopd.table`. TSD-KD's code is public (github.com/kmswin1/TSD-KD) if a rerun is
  needed. The entropy- and TIP-selective arms and the E4g entropy term are approximations: selection fraction
  0.1, coefficient 0.1, both to tune.
- **No a_T.** Prompts whose teacher response has no `\boxed{}` within 32 tokens (code, truncated) keep
  uniform weights, i.e. plain OPD, as the proposal specifies. `target_rate` in `metrics.jsonl` reports the
  share.
- **Probe length.** If x + y + (K+1) probe blocks exceed 2560 tokens, steps are merged into coarser
  boundaries (`coarsened_rollouts` in the metrics), the mitigation of proposal Sec. 7.
- **Model selection.** TSD-KD keeps the best epoch by held-out GKD loss. Here the final checkpoint is
  evaluated, and per-epoch checkpoints are kept in `checkpoint-*` for the same selection.
