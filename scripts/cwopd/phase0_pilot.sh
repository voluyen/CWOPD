#!/usr/bin/env bash
# Phase 0b: data, teacher targets, E1 distinctness pilot and E2 probe validity, before training.
# Decision point on H1: read results/cwopd/qwen25-1.5b/pilot/report.md. If `h1_stop_signal` is
# true (consequence overlaps strongly with divergence), stop or reframe (proposal Sec. 7).
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
cwopd run qwen25-1.5b/pilot "$@"
cat results/cwopd/qwen25-1.5b/pilot/report.md
