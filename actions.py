from __future__ import annotations

from typing import Mapping

from config import CLASSES as FINE_GRAINED_CLASSES
from config import CLASS_TO_IDX, IDX_TO_CLASS, MODEL_CLASSES

CLASSES = list(MODEL_CLASSES)

LEFT_DIRECTIONS = {"left", "light_left", "pivot_left", "sharp_left"}
RIGHT_DIRECTIONS = {"right", "light_right", "pivot_right", "sharp_right"}
FORWARD_DIRECTIONS = {"forward"}
IGNORED_DIRECTIONS = {"backward", "stop", "other", "ignored"}


def normalize_model_action(label: str | None) -> str | None:
    """Collapse raw labels into the 3-class space used by the model."""
    if label is None:
        return None

    normalized = str(label).strip().lower()

    if normalized in CLASSES:
        return normalized
    if normalized in LEFT_DIRECTIONS:
        return "left"
    if normalized in RIGHT_DIRECTIONS:
        return "right"
    if normalized in FORWARD_DIRECTIONS:
        return "forward"
    if normalized in IGNORED_DIRECTIONS:
        return None

    return None


def motor_direction(gpio_a: int, gpio_b: int, side: str) -> str:
    """Decode one motor state from the GPIO pair that drives it.

    The left and right motors use mirrored GPIO conventions because of the
    physical wiring, so the mapping intentionally depends on ``side``.
    """
    if side == "left":
        if gpio_a == 0 and gpio_b == 0:
            return "stop"
        if gpio_a == 1 and gpio_b == 0:
            return "forward"
        if gpio_a == 0 and gpio_b == 1:
            return "backward"
    elif side == "right":
        if gpio_a == 0 and gpio_b == 0:
            return "stop"
        if gpio_a == 0 and gpio_b == 1:
            return "forward"
        if gpio_a == 1 and gpio_b == 0:
            return "backward"

    return "unknown"


def classify_direction(row: Mapping[str, int | float]) -> str:
    """Infer the fine-grained driving action from one telemetry row."""
    speed_a = float(row["speedA"])
    speed_b = float(row["speedB"])

    # The GPIO decoding is asymmetric because each side is wired differently.
    left_dir = motor_direction(int(row["GPIO1"]), int(row["GPIO2"]), "left")
    right_dir = motor_direction(int(row["GPIO3"]), int(row["GPIO4"]), "right")

    if left_dir == "stop" and right_dir == "stop":
        return "stop"
    if left_dir == "backward" and right_dir == "backward":
        return "backward"
    if left_dir == "forward" and right_dir == "backward":
        return "sharp_right"
    if left_dir == "backward" and right_dir == "forward":
        return "sharp_left"
    if left_dir == "forward" and right_dir == "stop":
        return "pivot_right"
    if left_dir == "stop" and right_dir == "forward":
        return "pivot_left"

    if left_dir == "forward" and right_dir == "forward":
        # Once both wheels go forward, relative speed tells us whether the car
        # is drifting gently left or right.
        if speed_a == speed_b:
            return "forward"
        if speed_a > speed_b:
            return "light_right"
        return "light_left"

    return "other"


def row_to_model_action(row: Mapping[str, int | float]) -> str | None:
    """Map one raw CSV row directly to the simplified model label."""
    return normalize_model_action(classify_direction(row))


def motor_to_action(
    speed_a: float,
    speed_b: float,
    eps: float = 5.0,
    min_speed: float = 5.0,
    motor_a_is_left: bool = True,
) -> str:
    """Infer a coarse action from signed motor speeds only.

    This heuristic is useful when GPIO states are unavailable and the sign of
    each speed already encodes forward vs backward motion.
    """
    speed_a = float(speed_a)
    speed_b = float(speed_b)

    if abs(speed_a) < min_speed and abs(speed_b) < min_speed:
        return "stop"

    if abs(speed_a - speed_b) < eps:
        if speed_a > min_speed and speed_b > min_speed:
            return "forward"
        if speed_a < -min_speed and speed_b < -min_speed:
            return "backward"
        return "stop"

    # ``motor_a_is_left`` keeps the helper reusable if the wiring order changes.
    if motor_a_is_left:
        if speed_b > speed_a + eps:
            return "left"
        if speed_a > speed_b + eps:
            return "right"
    else:
        if speed_a > speed_b + eps:
            return "left"
        if speed_b > speed_a + eps:
            return "right"

    return "other"


__all__ = [
    "CLASS_TO_IDX",
    "CLASSES",
    "FINE_GRAINED_CLASSES",
    "IDX_TO_CLASS",
    "classify_direction",
    "motor_direction",
    "motor_to_action",
    "normalize_model_action",
    "row_to_model_action",
]
