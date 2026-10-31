#!/usr/bin/env bash
# Phase 2: E3 main comparison on Qwen2.5-14B -> Qwen2.5-1.5B (RIVAD Table 2 setting). Only CW-OPD is
# trained; every baseline (SeqKD, Supervised-KD, GKD, MiniLLM, DistiLLM, Speculative KD, TSD-KD, CSD,
# AMiD, RIVAD) is read from the published rows in answergain.cwopd.table. The other arm configs
# (opd-rkl, gkd-jsd, seqkd, supervised-kd, select-*) stay available for single runs.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
run_seeds qwen25-1.5b/cw-opd "$@"
cwopd run qwen25-1.5b/cw-opd --stages e6 "$@"
cwopd run qwen25-1.5b/cw-opd --stages table "$@"
