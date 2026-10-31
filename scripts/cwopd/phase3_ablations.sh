#!/usr/bin/env bash
# Phase 3: E4 ablations (Qwen pair), the second pair (Gemma2-9B -> Gemma2-2B, RIVAD Table 6)
# and the E6 analyses. On Gemma2 only CW-OPD is trained; baselines come from RIVAD Table 6.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
for config in e4a-permuted e4a-entropy e4a-divergence e4b-signed e4c-student-scorer e4d-rollout-mean \
              e4d-alerd e4e-no-floor e4f-raw-value e4g-entropy-min e4h-paragraph; do
  run_seeds "qwen25-1.5b/$config" "$@"
done
run_seeds gemma2-2b/cw-opd "$@"
for config in qwen25-1.5b/e4b-signed qwen25-1.5b/e4d-alerd gemma2-2b/cw-opd; do
  cwopd run "$config" --stages e6 "$@"
done
cwopd run qwen25-1.5b/cw-opd --stages table "$@"
cwopd run gemma2-2b/cw-opd --stages table "$@"
