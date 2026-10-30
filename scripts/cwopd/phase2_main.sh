#!/usr/bin/env bash
# Phase 2: E3 main comparison on Qwen2.5-14B -> Qwen2.5-1.5B (RIVAD Table 2 setting). Every
# on-policy arm shares prompts, rollout budget (512 new tokens), teacher and schedule.
# MiniLLM, DistiLLM, Speculative KD, TSD-KD, CSD and AMiD are read from the published rows
# (answergain.cwopd.table) or run with their own code; they are not reimplemented here.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
for config in seqkd supervised-kd opd-rkl gkd-jsd select-entropy select-tip cw-opd cw-gkd-jsd; do
  run_seeds "qwen25-1.5b/$config" "$@"
done
cwopd run qwen25-1.5b/cw-opd --stages e6 "$@"
cwopd run qwen25-1.5b/cw-opd --stages table "$@"
