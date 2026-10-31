"""Training rollouts from a vLLM engine colocated with the trainer on the same GPU.

The engine lives in the trainer's process (VLLM_ENABLE_V1_MULTIPROCESSING=0), and the student's current
weights are copied into it before every step. Samples therefore still come from p_S of that step, as with
HF generate, at a fraction of the time. vLLM is imported lazily so the trainer runs without it on CPU.
"""

import os

import torch


class VLLMRollouts:
    def __init__(self, model_path: str, *, stops: list[int], vocab: int, out_vocab: int, max_new_tokens: int,
                 temperature: float, top_p: float, gpu_memory_utilization: float, max_model_len: int, seed: int):
        # in-process engine: load_weights then copies tensors in place, no IPC or serialization
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM

        self.llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, seed=seed, enable_prefix_caching=False)
        self.stops = stops
        # padded embedding rows (Qwen2.5 pads its vocabulary) are not tokens; HF generate suppresses them too.
        # A bias on the few padded ids, not allowed_token_ids over the whole vocabulary: vLLM caps that list
        # at 1024 entries. -100 leaves them a probability of ~e^-100 relative to any real token.
        self.bias = {token: -100.0 for token in range(vocab, out_vocab)} or None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _models(self, fn):
        if hasattr(self.llm, "apply_model"):
            return self.llm.apply_model(fn)
        executor = self.llm.llm_engine.model_executor  # older vLLM without LLM.apply_model
        return [fn(executor.driver_worker.worker.model_runner.model)]

    @torch.no_grad()
    def load(self, module) -> None:
        """Copy the student's trainable weights (HF names, as vLLM's loaders expect) into the engine."""
        weights = [(name, param.detach().to(torch.bfloat16)) for name, param in module.named_parameters()]
        self._models(lambda model: model.load_weights(weights))

    def generate(self, prompts: list[list[int]], seed: int) -> list[tuple[list[int], str]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sampling = {"temperature": self.temperature, "top_p": self.top_p} if self.temperature > 0 else {
            "temperature": 0.0}
        params = [
            SamplingParams(max_tokens=self.max_new_tokens, stop_token_ids=self.stops, seed=seed + index,
                           logit_bias=self.bias, repetition_penalty=1.0, top_k=-1, **sampling)
            for index in range(len(prompts))
        ]
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
        results = []
        for output in outputs:
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            if completion.finish_reason == "stop":
                if not ids or ids[-1] not in self.stops:
                    ids.append(self.stops[0])  # keep the turn-closing token, as HF generate does
                results.append((ids, "stop"))
            else:
                results.append((ids, "length"))
        return results

