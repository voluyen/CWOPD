# Answer-Gain Allocation

Code for **Decisive, Not Difficult: Allocating Learning Effort across Reasoning Steps in Long-CoT Distillation**.

Standard SFT on long chain-of-thought traces gives every token the same weight, and selective methods
drop steps outright. Answer-gain allocation keeps every step supervised and redistributes each trace's
token budget instead:

1. Split a fixed teacher trace into steps (sentence boundaries).
2. Score each step by its **answer-likelihood gain** `u_k = b_k - b_{k-1}`, where
   `b_k = log q0(a* | x, s_1:k, probe)` under the frozen student `q0` (one offline vLLM pass, no sampling or gradients).
3. Standardize the gains within the trace, clip to `[-c, c]`, apply `exp(v/tau)`, normalize by token mass
   so the trace keeps its total weight, then shrink toward uniform with `lambda` (Eq. 4-8).
4. Train with the weighted SFT loss (Eq. 9).

Defaults: `c = 2`, `lambda = 0.5`, `tau = 1` (main results). On 966 P-ALIGN traces the paper reports
+2.02 avg pass@1 for DeepSeek-R1-Distill-Qwen-1.5B and +3.83 for Qwen3-8B over P-ALIGN.

## Install

```bash
uv sync                 # torch, vLLM, transformers, DeepSpeed, PEFT (pyproject.toml, Python 3.12)
# or: pip install -e .
```

This installs three commands: `answergain` (the method, ablations and the SGL baseline), `palign`
(P-ALIGN baseline) and `ssft` (Segment-Selective SFT baseline). The reference trainers
(LLaMA-Factory for P-ALIGN, Unsloth for Segment-Selective SFT) need their own environments; point a
stage at one with `stages.<stage>.python=/path/to/env/bin/python`.

## Data

Datasets are pulled from the Hugging Face Hub, nothing is stored in the repo:

| Data | Source |
|---|---|
| 966 P-ALIGN traces (main track) | [`qizheyanger/P-ALIGN`](https://huggingface.co/datasets/qizheyanger/P-ALIGN) |
| s1K-1.1 DeepSeek-R1 traces (long-CoT baselines) | [`baesad/s1K-1.1-deepseek-cot`](https://huggingface.co/datasets/baesad/s1K-1.1-deepseek-cot) |
| AIME24, AIME25, AMC12, MATH500 | fetched by the eval stage |

## Quickstart

```bash
answergain list                                           # experiments under training_configs/
answergain show r1-qwen-1.5b-palign/e7-alg-tau1           # fully resolved config
answergain run  r1-qwen-1.5b-palign/e7-alg-tau1 --dry-run # print the commands, run nothing
answergain run  r1-qwen-1.5b-palign/e7-alg-tau1 run.gpus=[0,1]

palign run r1-qwen-1.5b          # P-ALIGN baseline, same student
ssft   run r1-qwen-1.5b          # Segment-Selective SFT baseline
```

* `key.path=value` overrides any config value, for example `train_seed=43` or `allocation.temperature=1.0`.
* `--stages a,b` runs only those stages, `--from b` resumes there, `--force` reruns finished stages.
* A stage is skipped when its `creates` path exists. Logs go to `logs/<run>-<stage>.log`.
* `answergain check` validates every config against the argument parsers of the modules it calls.
* `scripts/<model>/run.sh` runs every recipe of one student (`r1-qwen-1.5b`, `qwen25-7b`, `qwen3-8b`).

Pipeline for the answer-gain arms: `prepare -> answer_gain -> gain_signal -> weights -> train -> eval`.
The two offline steps also run standalone:

```bash
python generate/prepare_data.py --help
python generate/score_answer_gain.py --help
python eval/run_eval.py --help
```

## Reproducing the paper

Evaluation follows P-ALIGN: thinking off, `n = 3` samples, temperature 0.6, top-p 0.9, repetition
penalty 1.05, 4096 tokens, Math-Verify grading.

| Paper | Command |
|---|---|
| Table 2, uniform SFT (P-ALIGN data) | `answergain run r1-qwen-1.5b-palign/e0-uniform` |
| Table 2, answer-gain allocation, 1.5B | `answergain run r1-qwen-1.5b-palign/e7-alg-tau1` |
| Table 2, Qwen2.5-7B / Qwen3-8B (LoRA) | `answergain run qwen25-7b-palign/answer-gain-lora`, `qwen3-8b-palign/answer-gain-lora` |
| Table 2, baselines | `palign run <model>`, `ssft run <model>`, `answergain run r1-qwen-1.5b-palign/answergain-spectral` (SGL) |
| Table 3, step score and normalization | `e2-predictability-easy`, `e3-predictability-hard`, `e4-random-assignment`, `e5-alg-no-answer-upweight`, `e9-alg-global-norm`, `e10-alg-no-norm` |
| Table 4, shrinkage `lambda` | `e1-alg` (`lambda = 0.5`), `e6-alg-lambda1` |
| Table 5, temperature `tau` | `e7-alg-tau1`, `e1-alg` (`tau = 2`), `e8-alg-tau4` |
| Appendix F, DFT variants | `r1-qwen-1.5b-palign/sft-dft`, `dft-*` |

All names are under `training_configs/`. [docs/reproducing.md](docs/reproducing.md) lists the remaining
rows and the multi-GPU and multi-seed options. Note that `allocation.temperature` defaults to `2.0`
in `training_configs/_methods/_allocation.yaml`, while the paper's main setting is `tau = 1`:
`e7-alg-tau1` sets it, but `qwen25-7b-palign/answer-gain-lora` and `qwen3-8b-palign/answer-gain-lora`
do not, so pass `allocation.temperature=1.0` there to match the paper.

## Layout

```
src/answergain/        library
  signals/             answer gain, probe, score transforms
  allocation/          weights (Eq. 4-8), per-trace / global / no normalization
  data/                tokenize + segment, P-ALIGN and s1K loaders
  segment/             sentence segmentation, char -> token spans
  selection/           spectral gate (SGL baseline)
  training/            weighted SFT objective, trainers, Unsloth backend
  eval/                benchmarks, vLLM generation, graders, result tables
  palign/  ssft/       baseline pipelines
  cli.py  config.py    stage runner and YAML config system
training_configs/      one YAML per experiment, plus _common/, _methods/, _tracks/
accelerate_configs/    DeepSpeed ZeRO-2 config
generate/              offline data preparation and answer-gain scoring
eval/                  evaluation and comparison entry points
baselines/             P-ALIGN, Segment-Selective SFT and SGL entry points
scripts/               per-model run.sh and setup.sh
references/            the original P-ALIGN and Segment-Selective SFT code
tests/                 unit tests, equivalence tests against references/, regression tests
results/               summary.json per run, comparison-table.md
```

## Development

```bash
pytest -m "not regression"     # unit and equivalence tests, CPU only
ruff check src tests
```

signal, segmenter or experiment.

## CW-OPD (on-policy follow-up)

`cwopd` runs consequence-weighted on-policy distillation (`proposal.pdf`) under the RIVAD / TSD-KD
setup; see [scripts/cwopd/README.md](scripts/cwopd/README.md).

## Citation

The paper is under anonymous review; a BibTeX entry will be added after the review.
