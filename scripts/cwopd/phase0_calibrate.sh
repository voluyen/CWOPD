#!/usr/bin/env bash
# Phase 0a: evaluate the untrained teachers and students and compare with the first rows of
# RIVAD Table 2 / Table 6. If they do not match, fix the protocol (metric key, chat template)
# in answergain/cwopd/lm_eval_suite.py before training anything.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
for config in qwen25-1.5b/base-teacher qwen25-1.5b/base-student gemma2-2b/base-teacher gemma2-2b/base-student; do
  cwopd run "$config" "$@"
done
cwopd run qwen25-1.5b/base-student --stages table "$@"
cwopd run gemma2-2b/base-student --stages table "$@"
