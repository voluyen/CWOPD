#!/usr/bin/env bash
# Quick GPU smoke test of the whole CW-OPD path (data -> targets -> train -> eval): two full-size steps
# (128 prompts) so their timings and peak_memory_gb hold for the real run, then a tiny eval,
# in throwaway dirs. Run this first on a new server.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
smoke=(name=smoke-cw-opd data_dir=data/cwopd/smoke results_dir=results/cwopd/smoke
       stages.data.args.limit=600 stages.train.args.max-steps=2 stages.train.args.prompts-per-step=128
       stages.train.args.dump-rollouts-every=1 stages.eval.args.limit=8 eval_benchmarks=gsm8k,math)
cwopd run qwen25-1.5b/cw-opd "${smoke[@]}" "$@"
cwopd run qwen25-1.5b/cw-opd "${smoke[@]}" --stages e6 "$@"
tail -n 2 checkpoints/cwopd/smoke-cw-opd-qwen25-1.5b/metrics.jsonl
