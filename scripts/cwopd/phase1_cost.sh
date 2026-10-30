#!/usr/bin/env bash
# Phase 1: E5 cost of the packed probes (teacher time and memory against K).
# E2 already ran in phase0_pilot.sh (stage mc). Training-time overhead is also logged per step
# (score_s, probe_overhead_tokens, peak_memory_gb in checkpoints/cwopd/<run>/metrics.jsonl).
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
cwopd run qwen25-1.5b/cost "$@"
