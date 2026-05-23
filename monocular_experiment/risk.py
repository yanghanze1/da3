from __future__ import annotations

import numpy as np

from .models import RiskAssessment, TrackedObject


def safe_braking_distance_m(speed_mps: float, reaction_time_s: float, max_decel_mps2: float) -> float:
    """论文 Eq.38：理论安全刹车距离 d_brake = v·τ + v²/(2·a_max)。"""

    return float(speed_mps) * float(reaction_time_s) + (float(speed_mps) ** 2) / (2.0 * float(max_decel_mps2))


def estimate_clear_path_width(
    road_world_points: np.ndarray,
    anchor_z_m: float,
    config: dict[str, float],
) -> float:
    """论文 Eq.32 / 5.3.4：估计同纵深处可通行横向净宽 W_clear。

    在锚点纵深附近取道路面点，计算其在横向(Y轴)上的跨度。
    """
    if len(road_world_points) == 0:
        return float(config["default_clear_path_width_m"])
    z_window_m = float(config["z_window_m"])
    near_mask = np.abs(road_world_points[:, 2] - float(anchor_z_m)) <= z_window_m
    if int(np.count_nonzero(near_mask)) < int(config["min_free_space_points"]):
        return float(config["default_clear_path_width_m"])
    ys = road_world_points[near_mask, 1]
    width = float(np.max(ys) - np.min(ys))
    return max(width, float(config["default_clear_path_width_m"]))


def compute_lateral_occupancy_ratio(width_obs_m: float, w_clear_m: float) -> float:
    """横向占用比例：w_obs / W_clear。论文 Eq.37 第三项条件。"""
    if w_clear_m <= 0.0:
        return 1.0
    return float(width_obs_m) / float(w_clear_m)


def _height_thresholds(obstacle_type: str, config: dict[str, float]) -> tuple[float, float]:
    """返回 (delta_h_danger, delta_h_warning) 落差门坎。"""
    if obstacle_type == "negative":
        return float(config["negative_limit_height_m"]), float(config["negative_safe_height_m"])
    return float(config["positive_limit_height_m"]), float(config["positive_safe_height_m"])


def _bypass_capacity_m(w_clear_m: float, config: dict[str, float]) -> float:
    usable = float(w_clear_m) - float(config["vehicle_width_m"]) - 2.0 * float(config["side_clearance_m"])
    return max(0.0, usable)


def _optional_float(config: dict[str, float], key: str) -> float | None:
    value = config.get(key)
    return None if value is None else float(value)


def _valid_dimension(value: float, max_value: float | None) -> bool:
    return np.isfinite(value) and value >= 0.0 and (max_value is None or value <= max_value)


def _valid_object_geometry(object_state: TrackedObject, config: dict[str, float]) -> bool:
    max_valid_height_m = _optional_float(config, "max_valid_height_m")
    max_valid_width_m = _optional_float(config, "max_valid_width_m")
    return _valid_dimension(float(object_state.height_m), max_valid_height_m) and _valid_dimension(
        float(object_state.width_m),
        max_valid_width_m,
    )


def classify_risk_weight(
    object_state: TrackedObject,
    w_clear_m: float,
    config: dict[str, float],
) -> tuple[str, bool]:
    """论文：依 H_obs 与 W_obs/W_clear 判定 omega0/1/2 风险层级。"""
    delta_h_danger, delta_h_warning = _height_thresholds(object_state.obstacle_type, config)
    bypass_capacity_m = _bypass_capacity_m(w_clear_m, config)
    bypassable = float(object_state.width_m) < bypass_capacity_m
    h_obs = float(object_state.height_m)

    if h_obs <= delta_h_warning:
        return "omega0", bypassable
    if h_obs <= delta_h_danger and bypassable:
        return "omega1", bypassable
    return "omega2", bypassable


def decision_for_object(
    object_state: TrackedObject,
    w_clear_m: float,
    distance_m: float,
    d_brake_m: float,
    config: dict[str, float],
) -> str:
    """论文 Eq.37 / Table 5-2：三维条件组合判定 Safe / Warning / Danger。

    Danger 条件（三者之一）：
      1. 落差过大：height_m > delta_h_danger
      2. 距离过近：distance_m <= d_brake_m
      3. 横向占用过高：w_obs / W_clear > width_occupancy_threshold

    Warning 条件：
      距离已进入警告范围，或落差/横向占用已接近危险水平

    否则为 Safe。
    """
    delta_h_danger, delta_h_warning = _height_thresholds(object_state.obstacle_type, config)
    height_occ = float(object_state.height_m) > delta_h_danger
    dist_occ = distance_m <= d_brake_m
    width_occ_thresh = float(config.get("width_occupancy_threshold", 0.85))
    lateral_ratio = compute_lateral_occupancy_ratio(float(object_state.width_m), w_clear_m)
    width_occ = lateral_ratio > width_occ_thresh

    # Danger: 任意一项满足
    if height_occ or dist_occ or width_occ:
        return "danger"

    # Warning: 落差接近危险 或 距离在警告范围内
    warning_scale = float(config.get("warning_distance_scale", 1.5))
    warning_dist = warning_scale * d_brake_m
    height_near_danger = float(object_state.height_m) > delta_h_warning
    dist_in_warning = distance_m <= warning_dist
    width_near_limit = lateral_ratio > (width_occ_thresh * 0.7)

    if height_near_danger or dist_in_warning or width_near_limit:
        return "warning"

    return "safe"


def assess_tracked_objects(
    tracked_objects: list[TrackedObject],
    road_world_points: np.ndarray,
    speed_mps: float,
    config: dict[str, float],
) -> list[RiskAssessment]:
    """对追踪物件做论文 Table 5-2 风险矩阵与刹车决策。"""

    d_brake_m = safe_braking_distance_m(
        speed_mps=speed_mps,
        reaction_time_s=float(config["reaction_time_s"]),
        max_decel_mps2=float(config["max_decel_mps2"]),
    )
    assessments: list[RiskAssessment] = []
    min_forward_distance = float(config.get("min_forward_distance_m", 0.0))
    for object_state in tracked_objects:
        if object_state.state == "retired":
            continue
        if not np.isfinite(float(object_state.distance_m)) or float(object_state.distance_m) <= min_forward_distance:
            continue
        if not _valid_object_geometry(object_state, config):
            continue
        w_clear_m = estimate_clear_path_width(road_world_points, object_state.distance_m, config)
        lateral_ratio = compute_lateral_occupancy_ratio(float(object_state.width_m), w_clear_m)
        risk_weight, bypassable = classify_risk_weight(object_state, w_clear_m, config)
        decision = decision_for_object(object_state, w_clear_m, object_state.distance_m, d_brake_m, config)
        assessments.append(
            RiskAssessment(
                object_id=object_state.object_id,
                risk_weight=risk_weight,
                decision=decision,
                bypassable=bypassable,
                w_clear_m=w_clear_m,
                d_brake_m=d_brake_m,
                metadata={
                    "state": object_state.state,
                    "height_m": float(object_state.height_m),
                    "width_m": float(object_state.width_m),
                    "distance_m": float(object_state.distance_m),
                    "lateral_occupancy_ratio": lateral_ratio,
                    "bypass_capacity_m": _bypass_capacity_m(w_clear_m, config),
                },
            )
        )
    return assessments


def attach_risk_to_objects(
    tracked_objects: list[TrackedObject],
    risk_assessments: list[RiskAssessment],
) -> list[TrackedObject]:
    by_id = {item.object_id: item for item in risk_assessments}
    for object_state in tracked_objects:
        risk = by_id.get(object_state.object_id)
        object_state.risk = risk.to_dict() if risk is not None else None
    return tracked_objects
