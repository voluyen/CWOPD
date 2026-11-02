"""MBPP answer extraction for instruction-tuned models (used by mbpp_cwopd.yaml).

lm-eval's own mbpp_instruct filter prepends ``` to the completion and then matches ```lang\\n...```:
a completion that starts with `def` loses that word to the language tag, and one that never closes
its fence yields nothing. Both are the normal output of Qwen2.5-Instruct, so the stock task scores
~0 for every model. Here the code is the completion up to its first closing fence; then only definitions, imports and plain assignments are kept, so the model's own
trailing asserts, prints and half-finished lines (the 256-token cut) cannot fail the test run.
"""

import ast
import re

import evaluate as hf_evaluate

pass_at_k = hf_evaluate.load("code_eval")
KEEP = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)


def pass_at_1(references, predictions) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return pass_at_k.compute(references=references, predictions=predictions, k=[1])[0]["pass@1"]


def extract_code(text: str) -> str:
    # the opening fence is part of the prompt (gen_prefix), so the code ends at the first fence; a
    # model that repeats the opening fence is handled by stripping it first
    code = re.sub(r"^\s*```(?:python)?[ \t]*\n", "", text).split("```")[0]
    lines = code.replace("\r\n", "\n").split("\n")
    while lines:  # drop trailing lines until the rest parses (a cut-off last statement)
        try:
            tree = ast.parse("\n".join(lines))
            break
        except SyntaxError:
            lines.pop()
    else:
        return ""
    return "\n\n".join(ast.get_source_segment("\n".join(lines), node) or "" for node in tree.body if isinstance(node, KEEP))


def build_predictions(resps, docs):
    return [[extract_code(r) for r in resp] for resp in resps]
