#!/usr/bin/env bash
# Every phase in the proposal's work plan order (Sec. 9). Overrides are passed to every run, e.g.
#   scripts/cwopd/run_all.sh run.gpus=[1]   (another GPU)  or  stages.train.args.teacher-cache=recompute
set -euo pipefail
here="$(dirname "${BASH_SOURCE[0]}")"
"$here/phase0_calibrate.sh" "$@"
"$here/phase0_pilot.sh" "$@"
"$here/phase1_cost.sh" "$@"
"$here/phase2_main.sh" "$@"
"$here/phase3_ablations.sh" "$@"
