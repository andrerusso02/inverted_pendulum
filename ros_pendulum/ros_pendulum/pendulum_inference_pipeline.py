from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from math import cos, sin

from ros_pendulum.inference_pipeline import (
    Action,
    FakeEngine,
    InferencePipeline,
    Observation,
    OnnxEngine,
    PipelineInput,
    PipelineOutput,
)
from scipy.signal import lfilter

# Mirror of ActionsCfg.motor_position in pendulum_env_cfg.py:
# EMAJointPositionToLimitsActionCfg(scale=2.0, rescale_to_limits=False,
# clip={"gim6010": (-2.0, 2.0)}, alpha=0.15).
ACTION_SCALE = 2.0
ACTION_EMA_ALPHA = 0.15
# Applied AFTER scale, in rad, exactly like the action term's clip. Without it the real
# controller tracks unbounded EMA(2*raw) targets into full rotations (sim only reset the
# episode at +-2 rad via too_many_rotations, which has no real-world counterpart).
ACTION_CLIP_RAD = 2.0


class EMAJointPositionFilter:

    def __init__(self, alpha: float, scale: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Moving average weight must be in the range [0, 1]. Got {alpha}.')
        self._alpha = alpha
        self._scale = scale
        # H(z) = alpha / (1 - (1 - alpha) z^-1). lfilter's transposed direct-form-II state for a
        # first-order section is z1 = -a[1] * y[k-1] = (1 - alpha) * y[k-1], which is exactly the
        # "previous applied action" the recursion carries.
        self._b = [alpha]
        self._a = [1.0, alpha - 1.0]
        self._zi = [0.0]

    def reset(self, joint_pos: float) -> None:
        """Seed the state from the measured joint position, as the term does on env reset."""
        self._zi = [(1.0 - self._alpha) * joint_pos]

    def process(self, raw_action: float) -> float:
        """Scale and clip one raw policy output, then return the filtered position command."""
        scaled = max(-ACTION_CLIP_RAD, min(ACTION_CLIP_RAD, self._scale * raw_action))
        applied, self._zi = lfilter(self._b, self._a, [scaled], zi=self._zi)
        return float(applied[0])


@dataclass
class PendulumPipelineInput(PipelineInput):
    gim_pos: float
    gim_vel: float
    free_pos: float


@dataclass
class PendulumPipelineOutput(PipelineOutput):
    gim_pos: float
    action_filter: EMAJointPositionFilter = field(
        default_factory=lambda: EMAJointPositionFilter(alpha=ACTION_EMA_ALPHA, scale=ACTION_SCALE)
    )

    def update(self, action: 'PendulumAction') -> None:
        self.gim_pos = self.action_filter.process(action.gim_pos)

    def reset(self, joint_pos: float) -> None:
        """Reseed the filter from the measured joint position (mirrors the sim reset and
        avoids slewing from 0 when the motor does not happen to sit at zero)."""
        self.gim_pos = joint_pos
        self.action_filter.reset(joint_pos)


@dataclass
class PendulumAction(Action):
    gim_pos: float

    @classmethod
    def from_vector(cls, vector: list[float]) -> 'PendulumAction':
        return cls(gim_pos=vector[0])


@dataclass
class PendulumObservation(Observation):
    # Order mirrors ObservationsCfg.PolicyCfg. Do not reorder.
    gim_pos_observation_history: deque[float]
    gim_vel_observation_history: deque[float]
    free_pos_observation_history: deque[tuple[float, float]]
    gim_pos_action_history: deque[float]

    @classmethod
    def create(cls) -> 'PendulumObservation':
        return cls(
            gim_pos_observation_history=deque([0.0] * 2, maxlen=2),
            gim_vel_observation_history=deque([0.0] * 2, maxlen=2),
            free_pos_observation_history=deque([(0.0, 1.0)] * 22, maxlen=22),
            gim_pos_action_history=deque([0.0] * 5, maxlen=5),
        )

    def update(
        self,
        previous_action: PendulumAction | None,
        pipeline_input: PendulumPipelineInput
    ) -> None:

        self.gim_pos_observation_history.append(pipeline_input.gim_pos)
        self.gim_vel_observation_history.append(pipeline_input.gim_vel)
        self.free_pos_observation_history.append(
            (sin(pipeline_input.free_pos), cos(pipeline_input.free_pos))
        )

        self.gim_pos_action_history.append(
            previous_action.gim_pos if previous_action is not None else 0.0
        )

    def to_vector(self) -> list[float]:
        # Isaac Lab flattens each term's history oldest first, with the sine/cosine
        # pair adjacent within each sample. Total observation size is 33.
        return [
            *self.gim_pos_observation_history,
            *self.gim_vel_observation_history,
            *(component for pair in self.free_pos_observation_history for component in pair),
            *self.gim_pos_action_history,
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
        # gim_pos=0.0 matches the filter's initial zi=[0.0], i.e. IsaacLab's reset() seeding
        # when the gim is at its zero. Call action_filter.reset(measured_pos) if it is not.
        output=PendulumPipelineOutput(gim_pos=0.0),
        action_cls=PendulumAction,
        log_function=log_function,
    )
