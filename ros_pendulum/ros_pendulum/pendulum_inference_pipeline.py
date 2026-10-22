from collections.abc import Callable

import numpy as np

from ros_pendulum import inference_pipeline


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


def warp_angle(angle: float) -> float:
    """Wraps an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def build_pendulum_inference_pipeline(
    policy_path: str, log_function: Callable[[str], None], use_fake: bool = False
) -> inference_pipeline.InferencePipeline:
    if use_fake:
        engine = FakeEngine(log_function)
    else:
        engine = inference_pipeline.OnnxEngine(policy_path, log_function)

    return inference_pipeline.InferencePipeline(
        inference_engine=engine,
        input_names=["gim_pos", "gim_vel", "free_pos", "free_vel"],
        output_names=["gim_pos"],
        observation_modifiers={},
        output_modifiers={},
        log_function=log_function,
    )
