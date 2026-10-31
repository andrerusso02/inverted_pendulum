from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ros_pendulum.inference_pipeline import (
    Action,
    FakeEngine,
    InferencePipeline,
    Observation,
    OnnxEngine,
    PipelineInput,
    PipelineOutput,
)


@dataclass
class PendulumPipelineInput(PipelineInput):
    gim_pos: float
    gim_vel: float
    free_pos: float
    free_vel: float


@dataclass
class PendulumPipelineOutput(PipelineOutput):
    gim_pos: float

    @classmethod
    def from_action(cls, action: 'PendulumAction') -> 'PendulumPipelineOutput':
        return cls(gim_pos=action.gim_pos)


@dataclass
class PendulumAction(Action):
    gim_pos: float

    @classmethod
    def from_vector(cls, vector: list[float]) -> 'PendulumAction':
        return cls(gim_pos=vector[0])


@dataclass
class PendulumObservation(Observation):
    # Order mirrors ObservationsCfg.PolicyCfg. Do not reorder.
    gim_pos: float
    gim_vel: float
    free_pos: float
    free_vel: float
    gim_pos_action_history: deque[float]
    gim_pos_observation_history: deque[float]
    gim_vel_observation_history: deque[float]
    free_pos_observation_history: deque[float]
    free_vel_observation_history: deque[float]

    @classmethod
    def create(cls) -> 'PendulumObservation':
        return cls(
            gim_pos=0.,
            gim_vel=0.,
            free_pos=0.,
            free_vel=0.,
            gim_pos_action_history=deque([0.0] * 3, maxlen=3),
            gim_pos_observation_history=deque([0.0] * 1, maxlen=1),
            gim_vel_observation_history=deque([0.0] * 1, maxlen=1),
            free_pos_observation_history=deque([0.0] * 1, maxlen=1),
            free_vel_observation_history=deque([0.0] * 1, maxlen=1),
        )

    def update(
        self,
        previous_action: PendulumAction | None,
        pipeline_input: PendulumPipelineInput
    ) -> None:
        self.gim_pos = pipeline_input.gim_pos
        self.gim_vel = pipeline_input.gim_vel
        self.free_pos = pipeline_input.free_pos
        self.free_vel = pipeline_input.free_vel

        self.gim_pos_observation_history.append(pipeline_input.gim_pos)
        self.gim_vel_observation_history.append(pipeline_input.gim_vel)
        self.free_pos_observation_history.append(pipeline_input.free_pos)
        self.free_vel_observation_history.append(pipeline_input.free_vel)

        self.gim_pos_action_history.append(
            previous_action.gim_pos if previous_action is not None else 0.0
        )

    def to_vector(self) -> list[float]:
        return [
            item
            for value in self.__dict__.values()
            for item in (value if isinstance(value, Iterable) else (value,))
        ]


def build_pendulum_inference_pipeline(
    policy_path: str, log_function: Callable[[str], None], use_fake: bool = False
) -> InferencePipeline:
    if use_fake:
        engine = FakeEngine(log_function)
    else:
        engine = OnnxEngine(policy_path, log_function)

    return InferencePipeline(
        inference_engine=engine,
        observation_cls=PendulumObservation,
        output_cls=PendulumPipelineOutput,
        action_cls=PendulumAction,
        log_function=log_function,
    )
