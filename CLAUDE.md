# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

Research code for distilling long chain-of-thought into small students, packaged as the `answergain`
framework (`src/answergain/`). Four method families share one pipeline and one config system, and each
has its own command and its own run names:

| Command  | Family | Configs | Run name |
|----------|--------|---------|----------|
| `answergain`    | spectral-guided learning, answer-gain allocation and Appendix F DFT comparisons | `training_configs/` | `<arm>-<track>` |
| `palign` | P-ALIGN (adaptive prefix alignment) | `training_configs/baselines/palign/` | `palign-<model>` |
| `ssft`   | Segment-Selective SFT (IG attribution) | `training_configs/baselines/ssft/` | `ssft-<model>` |
| `cwopd`  | consequence-weighted on-policy distillation (proposal.pdf), RIVAD/TSD-KD setup | `training_configs/cwopd/` | `<arm>-<pair>` |

Keep the families separate: never fold a P-ALIGN or SSFT run into an `answergain` config or name, and
keep their data dirs apart (P-ALIGN tokenizes differently from the SGL arms, see below).

`docs/methods.md` (what each family computes, fidelity of each port), `docs/reproducing.md`
(published rows → commands) and `docs/extending.md` (stage and data contracts) are the reference;
read them before changing a method.

## Commands

```bash
pytest                               # everything, ~2.5 min on CPU, no GPU or network needed
pytest -m "not regression"           # unit + equivalence tests only (~30 s)
pytest -m regression                 # golden outputs and CLI end-to-end
ruff check .         # lint (config in pyproject.toml); CI runs both

answergain list | show <cfg> | check | run <cfg> [--dry-run] [--stages a,b] [--from s] [--force] [key.path=value ...]
palign run r1-qwen-1.5b              # same CLI for palign / ssft, each only accepts its own configs
python -m answergain.<module> --help        # every stage module also runs standalone
```

Install: `uv sync` (or `pip install -e .`). `uv.lock` must be regenerated with network access
(`uv lock`) after dependency changes; it pins CUDA wheels (torch cu130, vLLM). For a CPU test env,
install CPU torch + `transformers==5.5.3 peft accelerate datasets pandas pyarrow pyyaml tqdm
matplotlib math-verify==0.8.0 sympy pytest ruff` and `pip install --no-deps -e .` (as CI does).

## Layout

```
src/answergain/
  cli.py, config.py   stage runner (four entry points) and YAML config system
  data/               prepare (tokenize + segment -> train-segmented.jsonl), s1k (trace rows)
  segment/            sentence | paragraph | cue rules, char -> token spans
  signals/            spectral capture (strength + entropy), answer gain
  selection/          spectral gate -> loss_mask          allocation/  answer-gain weights -> loss_weights
  data/dft_sources.py Appendix F token_source labels
  palign/             truncate -> align -> build          ssft/        attribution -> select -> build
  cwopd/              packed answer probes, consequence weights, on-policy trainer, pilots, lm-eval suite
  training/           masked / weighted objectives, HF trainer, Unsloth backend
  eval/               benchmarks, vLLM generation, graders, comparison table
training_configs/_common/       base, models, data blocks, training recipes, eval protocol
training_configs/     experiments; `_methods/`, `_tracks/`, `_common/` are building blocks
  baselines/          palign / ssft configs
accelerate_configs/   DeepSpeed JSON
generate/, eval/      thin scoring, preparation and evaluation entry points
baselines/            P-ALIGN, SSFT and SGL module entry points
references/           upstream P-ALIGN and SegmentSelectiveSFT code — read-only
tests/regression/     golden hashes, CLI end-to-end, offline assets
scripts/<model>/run.sh paper experiments via YAML; scripts/setup.sh installs dependencies
scripts/cwopd/       CW-OPD phases (setup, smoke, phase0-3); README there lists setup and deviations
results*/, logs/      live output dirs written by runs; experiments/ holds old root logs
```

## Conventions

- **Stage modules** expose `build_parser()` and `main(argv=None)`; `answergain check` validates config
  args against `build_parser()`, so keep heavy optional imports (vLLM, Unsloth) inside `main`.
- **Configs**: only `true`/`false` are booleans (`off`, `no` stay strings — `think_prefix: off` is
  meaningful). `${a.b}` interpolation, `${env:VAR,default}`. Paths are relative to the repo root.
  `family:` must match the command. Stage keys: `module|script`, `args`, `positional`, `launcher`
  (python|torchrun), `creates`, `gpus`, `effective_batch`, `env`, `cwd`, `python`, `raw_keys`.
- **Prompts**: SGL trains on the eval prompt `"...\boxed{}.<question>"` (no separator). P-ALIGN's
  LLaMA-Factory training uses `"...\boxed{}.\n<question>"` and, for Qwen, also supervises the `\n`
  after `<|im_end|>`; `training_configs/baselines/palign/_llamafactory-format.yaml` sets this. Evaluation is the same
  for all families (`training_configs/_common/eval/palign-protocol.yaml`: thinking off, n=3, T=0.6,
  top_p 0.9, rep. penalty 1.05, 4096 tokens, math_verify).
- **Schedules differ on purpose**: SGL arms decay lr to 1e-5 (`full-ds-z2.yaml`); P-ALIGN and SSFT
  decay to 0 (`min-learning-rate: 0.0`). SSFT keeps upstream quirks: no EOS appended, first /
  second-to-last / last segments always learned, seed 3407.
- Code and comments are in English, matching the surrounding style; comments explain *why*.
  `references/SegmentSelectiveSFT/CLAUDE.md` (Vietnamese without diacritics) applies only inside
  that directory, which should not be edited anyway.
- Add tests next to the area you change; for a new port, compare against the upstream code in
  `references/` rather than against a reimplementation.

## Environment notes

- GPU-only paths (real training, vLLM eval / judge, the upstream reference trainers) cannot run in
  a CPU sandbox; the end-to-end tests train a tiny random Qwen2 model instead. Say so when a change
  could only be checked on CPU.
- HF Hub may be unreachable: tests use `tests/regression/assets/` (a local BPE tokenizer, P-ALIGN
  samples, P-ALIGN's published generations with labels) and skip dataset-dependent checks when the HF cache is unavailable.
- The upstream reference trainers need their own environments
  (`stages.train_reference.python=/path/to/env/bin/python`).

## Dataset sources

P-ALIGN uses `qizheyanger/P-ALIGN`, file `P-ALIGN.json`, with the alpaca
`instruction` / `input` / `output` schema. `data.palign_download` uses `hf_hub_download`;
`data.prepare` streams that JSON with the existing shuffle seed and buffer.
s1K-1.1 uses `baesad/s1K-1.1-deepseek-cot` through `load_dataset`.
Reference training exports HF data and `dataset_info.json` into `data/palign-reference/`
and supplies `dataset_dir` / `dataset` overrides; upstream files remain read-only.

Resolved configuration snapshots and the audited differences are in `docs/refactor/`.
See `docs/reproducing.md` for retained paper rows and removed drivers.
