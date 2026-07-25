from collections.abc import Callable

import onnxruntime as ort

ModifierFunc = Callable[[float], float]


from collections.abc import Callable
from typing import Protocol


class InferenceEngine(Protocol):
    def infer(self, inputs: list[float]) -> list[float]: ...


class OnnxEngine:
    def __init__(self, policy_path: str, log_function: Callable[[str], None]) -> None:
        self._log_function = log_function
        self._ort_session = ort.InferenceSession(policy_path)
        self._input_name = self._ort_session.get_inputs()[0].name
        self._output_name = self._ort_session.get_outputs()[0].name

        self._log_function(
            f"ONNX model loaded. Input: {self._input_name}, Output: {self._output_name}"
        )

    def infer(self, inputs: list[float]) -> list[float]:
        return self._ort_session.run([self._output_name], {self._input_name: [inputs]})[
            0
        ][0]


class InferencePipeline:
    def __init__(
        self,
        inference_engine: InferenceEngine,
        input_names: list[str],
        output_names: list[str],
        observation_modifiers: dict[str, ModifierFunc],
        output_modifiers: dict[str, ModifierFunc],
        log_function: Callable[[str], None],
    ) -> None:
        self._inference_engine = inference_engine
        self.log_function = log_function

        self._input_keys = tuple(input_names)
        self._output_keys = tuple(output_names)

        self._input_modifiers = tuple(
            observation_modifiers.get(key, lambda x: x) for key in self._input_keys
        )
        self._output_modifiers = tuple(
            output_modifiers.get(key, lambda x: x) for key in self._output_keys
        )

    def execute(self, inputs: dict[str, float]) -> dict[str, float]:
        modified_inputs = [
            mod(inputs[key])
            for key, mod in zip(self._input_keys, self._input_modifiers)
        ]

        raw_outputs = self._inference_engine.infer(modified_inputs)

        return {
            key: mod(out)
            for key, mod, out in zip(
                self._output_keys, self._output_modifiers, raw_outputs
            )
        }
