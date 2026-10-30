#!/usr/bin/env bash
# Training environment (uv.lock: torch cu130, vLLM, transformers) and a separate lm-eval
# environment in .venv-eval, since lm-evaluation-harness pins its own vLLM.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
uv sync
uv venv .venv-eval --python 3.12
uv pip install --python .venv-eval/bin/python "lm_eval[vllm,math,ifeval]==0.4.13"
.venv-eval/bin/python -m lm_eval ls tasks | grep -E "^\|? *(gsm8k|gsm_plus|minerva_math|mbpp|sciq|mmlu_stem|mmlu_pro_math|bbh_cot_fewshot|ifeval) " || true
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Gemma2 checkpoints are gated: run 'huggingface-cli login' before the gemma2-2b configs."
