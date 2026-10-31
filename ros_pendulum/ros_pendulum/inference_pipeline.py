from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import onnxruntime as ort

# from leapp import InferenceManager


@dataclass
class PipelineInput(ABC):
    pass


@dataclass
class PipelineOutput(ABC):
    @classmethod
    @abstractmethod
    def from_action(cls, action: 'Action') -> 'PipelineOutput':
        pass


@dataclass
class Action(ABC):
    @classmethod
    @abstractmethod
    def from_vector(cls, vector: list[float]) -> 'Action':
        pass


@dataclass
class Observation(ABC):
    
    @classmethod
    @abstractmethod
    def create(cls) -> 'Observation':
        pass

    @abstractmethod
    def update(self, previous_action: Action | None, pipeline_input: PipelineInput) -> None:
        """Refresh from the latest sample. ``previous_action`` is None on the first tick."""

    @abstractmethod
    def to_vector(self) -> list[float]:
        pass


class InferenceEngine(Protocol):
    def infer(self, inputs: list[float]) -> list[float]: ...


class FakeEngine:
    def __init__(self, log_function: Callable[[str], None]) -> None:
        self._log_function = log_function
        self._log_function("FakeInference initialized.")

    def infer(self, inputs: list[float]) -> list[float]:
        input_summary = "Values received for inference:\n" + "\n".join(
            f" - {v}" for v in inputs
        )
        self._log_function(input_summary)
        return [0.0]


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
        raw = self._ort_session.run([self._output_name], {self._input_name: [inputs]})[0][0]
        return [float(v) for v in raw]


class LeappEngine:
    pass


class InferencePipeline:
    def __init__(
        self,
        inference_engine: InferenceEngine,
        observation_cls: type[Observation],
        output_cls: type[PipelineOutput],
        action_cls: type[Action],
        log_function: Callable[[str], None],
    ) -> None:
        self._inference_engine = inference_engine
        self._output_cls = output_cls
        self._action_cls = action_cls
        self.log_function = log_function

        self.previous_action: Action | None = None
        self.observation: Observation = observation_cls.create()

    def infer(self, pipeline_input: PipelineInput) -> PipelineOutput:
        self.observation.update(
            previous_action=self.previous_action,
            pipeline_input=pipeline_input,
        )

        output_vector = self._inference_engine.infer(self.observation.to_vector())
        action = self._action_cls.from_vector(output_vector)
        self.previous_action = action

        return self._output_cls.from_action(action)
