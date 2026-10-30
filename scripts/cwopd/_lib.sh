# Shared helpers for scripts/cwopd/*.sh. Source it; do not run it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Extra seeds besides the default 10 (RIVAD Table 5), e.g. SEEDS="11 12" for mean and spread.
SEEDS=${SEEDS:-}

# run_seeds <config> [overrides...]: the default seed, then every seed in $SEEDS.
run_seeds() {
  local config=$1; shift
  cwopd run "$config" "$@"
  for seed in $SEEDS; do
    cwopd run "$config" "seed=$seed" "seed_suffix=-s$seed" "$@"
  done
}
