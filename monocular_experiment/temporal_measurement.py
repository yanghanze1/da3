from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import bbox_iou, project_world_points
from .models import CandidateObservation


@dataclass
class TemporalFrameSnapshot:
    frame_id: str
    frame_index: int
    timestamp_s: float
    cumulative_forward_m: float
    candidates: list[CandidateObservation]
    image_gray: np.ndarray | None = None


@dataclass
class TemporalKeyframeSelection:
    snapshot: TemporalFrameSnapshot
    baseline_m: float
    time_gap_s: float
    score: float
    reject_reasons: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.reject_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.snapshot.frame_id,
            "frame_index": int(self.snapshot.frame_index),
            "baseline_m": float(self.baseline_m),
            "time_gap_s": float(self.time_gap_s),
            "score": float(self.score),
            "reject_reasons": list(self.reject_reasons),
        }


def _cfg_float(config: dict[str, Any], key: str, default: float) -> float:
    return float(config.get(key, default))


def _cfg_int(config: dict[str, Any], key: str, default: int) -> int:
    return int(config.get(key, default))


def _cfg_optional_float(config: dict[str, Any], key: str) -> float | None:
    value = config.get(key)
    return None if value is None else float(value)


def _valid_dimension(value: float, max_value: float | None) -> bool:
    return np.isfinite(value) and value >= 0.0 and (max_value is None or value <= max_value)


def transform_history_points_to_current(points: np.ndarray | list[list[float]], baseline_m: float) -> np.ndarray:
    transformed = np.asarray(points, dtype=np.float64).copy()
    if transformed.size == 0:
        return transformed.reshape(0, 3)
    transformed = transformed.reshape(-1, 3)
    transformed[:, 2] = transformed[:, 2] - float(baseline_m)
    return transformed


def _transform_anchor_to_current(anchor: list[float], baseline_m: float) -> np.ndarray:
    transformed = np.asarray(anchor, dtype=np.float64).copy()
    if transformed.size < 3:
        return np.full(3, np.nan, dtype=np.float64)
    transformed[2] = transformed[2] - float(baseline_m)
    return transformed


def _transform_z_range_to_current(z_range: list[float], baseline_m: float) -> np.ndarray:
    values = np.asarray(z_range, dtype=np.float64).copy()
    if values.size != 2:
        return np.full(2, np.nan, dtype=np.float64)
    values -= float(baseline_m)
    return np.sort(values)


def _z_range_overlap(a: list[float], b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    if a_arr.size != 2 or b.size != 2:
        return 0.0
    a0, a1 = float(np.min(a_arr)), float(np.max(a_arr))
    b0, b1 = float(np.min(b)), float(np.max(b))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-9, max(a1, b1) - min(a0, b0))
    return float(inter / union)


def _candidate_object_id(candidate: CandidateObservation) -> str | None:
    object_id = getattr(candidate, "object_id", None) or candidate.metadata.get("object_id")
    return str(object_id) if object_id else None


def select_keyframes(
    current_snapshot: TemporalFrameSnapshot,
    history: list[TemporalFrameSnapshot],
    config: dict[str, Any],
) -> list[TemporalKeyframeSelection]:
    if not bool(config.get("enabled", False)):
        return []

    min_baseline = _cfg_float(config, "min_baseline_m", 0.20)
    max_baseline = _cfg_float(config, "max_baseline_m", 1.20)
    target_baseline = _cfg_float(config, "target_baseline_m", (min_baseline + max_baseline) / 2.0)
    min_time_gap = _cfg_float(config, "min_time_gap_s", 0.20)
    max_time_gap = _cfg_float(config, "max_time_gap_s", 4.00)
    max_count = max(1, _cfg_int(config, "max_keyframes_per_candidate", 3))

    selections: list[TemporalKeyframeSelection] = []
    for snapshot in history:
        baseline = abs(float(current_snapshot.cumulative_forward_m) - float(snapshot.cumulative_forward_m))
        time_gap = abs(float(current_snapshot.timestamp_s) - float(snapshot.timestamp_s))
        reject_reasons: list[str] = []
        if baseline < min_baseline:
            reject_reasons.append("baseline_too_small")
        if baseline > max_baseline:
            reject_reasons.append("baseline_too_large")
        if time_gap < min_time_gap:
            reject_reasons.append("time_gap_too_small")
        if time_gap > max_time_gap:
            reject_reasons.append("time_gap_too_large")
        baseline_score = 1.0 - min(1.0, abs(baseline - target_baseline) / max(target_baseline, 1e-6))
        time_score = 1.0 - min(1.0, max(0.0, time_gap - min_time_gap) / max(max_time_gap - min_time_gap, 1e-6))
        score = max(0.0, 0.75 * baseline_score + 0.25 * time_score)
        selections.append(
            TemporalKeyframeSelection(
                snapshot=snapshot,
                baseline_m=baseline,
                time_gap_s=time_gap,
                score=score,
                reject_reasons=reject_reasons,
            )
        )

    valid = [item for item in selections if item.valid]
    valid.sort(key=lambda item: item.score, reverse=True)
    return valid[:max_count]


def _projected_bbox_and_centroid(
    candidate: CandidateObservation,
    baseline_m: float,
    intrinsics: dict[str, Any] | None,
    extrinsics: dict[str, Any] | None,
) -> tuple[list[int], np.ndarray] | None:
    if intrinsics is None or extrinsics is None:
        return None
    points = transform_history_points_to_current(candidate.support_points, baseline_m)
    if len(points) == 0:
        return None
    image_points = project_world_points(points, intrinsics, extrinsics)
    valid = np.all(np.isfinite(image_points), axis=1)
    if not np.any(valid):
        return None
    valid_points = image_points[valid]
    min_xy = np.floor(valid_points.min(axis=0)).astype(int)
    max_xy = np.ceil(valid_points.max(axis=0)).astype(int)
    wh = np.maximum(1, max_xy - min_xy)
    bbox = [int(min_xy[0]), int(min_xy[1]), int(wh[0]), int(wh[1])]
    return bbox, valid_points.mean(axis=0)


def match_history_candidate(
    current_candidate: CandidateObservation,
    history_candidates: list[CandidateObservation],
    baseline_m: float,
    config: dict[str, Any],
    intrinsics: dict[str, Any] | None = None,
    extrinsics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    max_anchor_distance = _cfg_float(config, "max_anchor_distance_m", 0.50)
    max_centroid_shift = _cfg_float(config, "max_centroid_shift_px", 120.0)
    min_match_score = _cfg_float(config, "min_match_score", 0.05)
    allow_known_id_geometry_fallback = bool(config.get("allow_known_object_id_geometry_fallback", False))
    current_anchor = np.asarray(current_candidate.anchor, dtype=np.float64)
    current_centroid = np.asarray(current_candidate.centroid_2d, dtype=np.float64)
    current_object_id = _candidate_object_id(current_candidate)

    def _metrics(candidate: CandidateObservation) -> dict[str, Any] | None:
        if candidate.obstacle_type != current_candidate.obstacle_type:
            type_score = 0.0
        else:
            type_score = 1.0
        transformed_anchor = _transform_anchor_to_current(candidate.anchor, baseline_m)
        if not np.all(np.isfinite(transformed_anchor)):
            return None
        min_forward_distance = _cfg_float(config, "min_forward_distance_m", 0.0)
        if transformed_anchor[2] <= min_forward_distance:
            return None
        anchor_dist = float(np.linalg.norm(current_anchor - transformed_anchor))
        projected = _projected_bbox_and_centroid(candidate, baseline_m, intrinsics, extrinsics)
        if projected is None:
            projected_bbox = candidate.bbox
            centroid = np.asarray(candidate.centroid_2d, dtype=np.float64)
            projection_used = False
        else:
            projected_bbox, centroid = projected
            projection_used = True
        centroid_shift = float(np.linalg.norm(current_centroid - centroid))
        iou = bbox_iou(current_candidate.bbox, projected_bbox)
        z_overlap = _z_range_overlap(current_candidate.z_range_m, _transform_z_range_to_current(candidate.z_range_m, baseline_m))
        anchor_score = max(0.0, 1.0 - anchor_dist / max(max_anchor_distance, 1e-6))
        centroid_score = max(0.0, 1.0 - centroid_shift / max(max_centroid_shift, 1e-6))
        score = 0.45 * anchor_score + 0.20 * centroid_score + 0.15 * float(iou) + 0.15 * z_overlap + 0.05 * type_score
        return {
            "candidate": candidate,
            "score": float(score),
            "anchor_distance_m": anchor_dist,
            "centroid_shift_px": centroid_shift,
            "bbox_iou": float(iou),
            "z_range_overlap": float(z_overlap),
            "object_id": _candidate_object_id(candidate),
            "projection_used": projection_used,
        }

    if current_object_id:
        id_matches = [candidate for candidate in history_candidates if _candidate_object_id(candidate) == current_object_id]
        best_id: dict[str, Any] | None = None
        for candidate in id_matches:
            item = _metrics(candidate)
            if item is None:
                continue
            item["match_method"] = "object_id"
            item["object_id_match"] = True
            if best_id is None or item["score"] > best_id["score"]:
                best_id = item
        if best_id is not None and best_id["anchor_distance_m"] <= max_anchor_distance and best_id["centroid_shift_px"] <= max_centroid_shift and best_id["score"] >= min_match_score:
            return best_id

    best: dict[str, Any] | None = None
    for candidate in history_candidates:
        history_object_id = _candidate_object_id(candidate)
        if current_object_id and history_object_id and history_object_id != current_object_id and not allow_known_id_geometry_fallback:
            continue
        item = _metrics(candidate)
        if item is None:
            continue
        if item["anchor_distance_m"] > max_anchor_distance:
            continue
        if item["centroid_shift_px"] > max_centroid_shift:
            continue
        if item["score"] < min_match_score:
            continue
        item["match_method"] = "geometry"
        item["object_id_match"] = bool(current_object_id and history_object_id and current_object_id == history_object_id)
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def _support_points(candidate: CandidateObservation) -> np.ndarray:
    points = np.asarray(candidate.support_points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return points.reshape(-1, 3)


def _robust_range(values: np.ndarray, low_pct: float, high_pct: float) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    low = float(np.percentile(values, low_pct))
    high = float(np.percentile(values, high_pct))
    if high < low:
        low, high = high, low
    return low, high


def fuse_candidate_measurement(
    current_candidate: CandidateObservation,
    matched_history: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    low_pct = _cfg_float(config, "robust_percentile_low", 5.0)
    high_pct = _cfg_float(config, "robust_percentile_high", 95.0)
    min_support = _cfg_int(config, "min_support_points", 6)
    min_history_support = _cfg_int(config, "min_history_support_points", 4)

    min_forward_distance = _cfg_float(config, "min_forward_distance_m", 0.0)
    current_points = _support_points(current_candidate)
    current_points = current_points[np.isfinite(current_points[:, 2]) & (current_points[:, 2] > min_forward_distance)]
    if len(current_points) < min_support:
        return {
            "enabled": True,
            "status": "insufficient_current_support",
            "method": "support_point_fusion",
            "quality": 0.0,
            "current_support_count": int(len(current_points)),
            "history_support_count": 0,
            "fused_support_count": int(len(current_points)),
            "selected_keyframes": [],
            "reject_reasons": ["insufficient_current_support"],
        }

    history_points: list[np.ndarray] = []
    selected_keyframes: list[dict[str, Any]] = []
    match_scores: list[float] = []
    for item in matched_history:
        candidate = item["candidate"]
        baseline = float(item["baseline_m"])
        transformed = transform_history_points_to_current(candidate.support_points, baseline)
        if len(transformed) == 0:
            continue
        transformed = transformed[np.isfinite(transformed[:, 2]) & (transformed[:, 2] > min_forward_distance)]
        if len(transformed) == 0:
            continue
        history_points.append(transformed)
        match_scores.append(float(item["match_score"]))
        selected_keyframes.append(
            {
                "frame_id": str(item["frame_id"]),
                "frame_index": int(item["frame_index"]),
                "candidate_id": str(candidate.candidate_id),
                "object_id": _candidate_object_id(candidate),
                "baseline_m": baseline,
                "time_gap_s": float(item["time_gap_s"]),
                "selector_score": float(item["selector_score"]),
                "match_score": float(item["match_score"]),
                "match_method": str(item.get("match_method", "geometry")),
                "object_id_match": bool(item.get("object_id_match", False)),
                "projection_used": bool(item.get("projection_used", False)),
            }
        )

    history_support_count = int(sum(len(points) for points in history_points))
    if history_support_count < min_history_support:
        return {
            "enabled": True,
            "status": "insufficient_history_support",
            "method": "support_point_fusion",
            "quality": 0.0,
            "current_support_count": int(len(current_points)),
            "history_support_count": history_support_count,
            "fused_support_count": int(len(current_points) + history_support_count),
            "selected_keyframes": selected_keyframes,
            "reject_reasons": ["insufficient_history_support"],
        }

    fused_points = np.vstack([current_points, *history_points])
    x0, x1 = _robust_range(fused_points[:, 0], low_pct, high_pct)
    y0, y1 = _robust_range(fused_points[:, 1], low_pct, high_pct)
    z0, z1 = _robust_range(fused_points[:, 2], low_pct, high_pct)
    height_m = max(float(current_candidate.height_m), float(abs(x1 - x0)))
    width_m = max(0.0, float(y1 - y0))
    distance_m = float(z0)
    z_range_m = [float(z0), float(z1)]
    if not np.isfinite(distance_m) or distance_m <= min_forward_distance:
        return {
            "enabled": True,
            "status": "invalid_forward_distance",
            "method": "support_point_fusion",
            "quality": 0.0,
            "current_support_count": int(len(current_points)),
            "history_support_count": history_support_count,
            "fused_support_count": int(len(fused_points)),
            "selected_keyframes": selected_keyframes,
            "reject_reasons": ["invalid_forward_distance"],
        }
    max_valid_height_m = _cfg_optional_float(config, "max_valid_height_m")
    max_valid_width_m = _cfg_optional_float(config, "max_valid_width_m")
    if not _valid_dimension(height_m, max_valid_height_m) or not _valid_dimension(width_m, max_valid_width_m):
        return {
            "enabled": True,
            "status": "invalid_geometry",
            "method": "support_point_fusion",
            "quality": 0.0,
            "current_support_count": int(len(current_points)),
            "history_support_count": history_support_count,
            "fused_support_count": int(len(fused_points)),
            "selected_keyframes": selected_keyframes,
            "reject_reasons": ["invalid_geometry"],
            "height_m": float(height_m) if np.isfinite(height_m) else None,
            "width_m": float(width_m) if np.isfinite(width_m) else None,
            "single_frame_height_m": float(current_candidate.height_m),
            "single_frame_width_m": float(current_candidate.width_m),
            "single_frame_distance_m": float(current_candidate.distance_m),
        }

    support_score = min(1.0, len(fused_points) / max(float(min_support + min_history_support), 1.0))
    history_score = min(1.0, history_support_count / max(float(min_history_support), 1.0))
    match_score = float(np.mean(match_scores)) if match_scores else 0.0
    keyframe_score = min(1.0, len(selected_keyframes) / max(1.0, float(_cfg_int(config, "max_keyframes_per_candidate", 3))))
    quality = float(np.clip(0.35 * support_score + 0.30 * history_score + 0.25 * match_score + 0.10 * keyframe_score, 0.0, 1.0))

    return {
        "enabled": True,
        "status": "ok",
        "method": "support_point_fusion",
        "quality": quality,
        "height_m": height_m,
        "width_m": width_m,
        "distance_m": distance_m,
        "z_range_m": z_range_m,
        "current_support_count": int(len(current_points)),
        "history_support_count": history_support_count,
        "fused_support_count": int(len(fused_points)),
        "selected_keyframes": selected_keyframes,
        "reject_reasons": [],
        "single_frame_height_m": float(current_candidate.height_m),
        "single_frame_width_m": float(current_candidate.width_m),
        "single_frame_distance_m": float(current_candidate.distance_m),
        "single_frame_z_range_m": [float(value) for value in current_candidate.z_range_m],
    }


def apply_temporal_measurement(candidate: CandidateObservation, measurement: dict[str, Any], config: dict[str, Any]) -> bool:
    measurement = dict(measurement)
    min_quality = _cfg_float(config, "min_quality_to_apply", 0.60)
    should_apply = (
        bool(config.get("apply_to_candidate_metrics", True))
        and measurement.get("status") == "ok"
        and float(measurement.get("quality", 0.0)) >= min_quality
    )
    measurement["applied"] = bool(should_apply)
    candidate.metadata["temporal_measurement"] = measurement
    if not should_apply:
        return False

    candidate.height_m = float(measurement["height_m"])
    candidate.width_m = float(measurement["width_m"])
    candidate.distance_m = float(measurement["distance_m"])
    candidate.z_range_m = [float(value) for value in measurement["z_range_m"]]
    anchor = np.asarray(candidate.anchor, dtype=np.float64).copy()
    if anchor.size >= 3:
        anchor[2] = candidate.distance_m
        candidate.anchor = anchor.round(6).tolist()
    return True


class TemporalMeasurementManager:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.history: list[TemporalFrameSnapshot] = []

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    def keyframe_count(self) -> int:
        return len(self.history)

    def measure_candidates(
        self,
        *,
        frame_bgr: np.ndarray | None,
        frame_id: str,
        frame_index: int,
        timestamp_s: float,
        cumulative_forward_m: float,
        candidates: list[CandidateObservation],
        config: dict[str, Any] | None = None,
        intrinsics: dict[str, Any] | None = None,
        extrinsics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cfg = dict(self.config)
        if config:
            cfg.update(config)
        if not bool(cfg.get("enabled", False)):
            return []

        current_snapshot = TemporalFrameSnapshot(
            frame_id=frame_id,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            cumulative_forward_m=cumulative_forward_m,
            candidates=candidates,
            image_gray=None,
        )
        selections = select_keyframes(current_snapshot, self.history, cfg)
        measurements: list[dict[str, Any]] = []
        if not selections:
            for candidate in candidates:
                measurement = {
                    "enabled": True,
                    "candidate_id": candidate.candidate_id,
                    "roi_id": candidate.roi_id,
                    "object_id": _candidate_object_id(candidate),
                    "status": "no_keyframe",
                    "method": "support_point_fusion",
                    "quality": 0.0,
                    "selected_keyframes": [],
                    "reject_reasons": ["no_keyframe"],
                    "applied": False,
                }
                candidate.metadata["temporal_measurement"] = measurement
                measurements.append(measurement)
            return measurements

        for candidate in candidates:
            matched_history: list[dict[str, Any]] = []
            for selection in selections:
                match = match_history_candidate(
                    candidate,
                    selection.snapshot.candidates,
                    selection.baseline_m,
                    cfg,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                )
                if match is None:
                    continue
                matched_history.append(
                    {
                        "candidate": match["candidate"],
                        "frame_id": selection.snapshot.frame_id,
                        "frame_index": selection.snapshot.frame_index,
                        "baseline_m": selection.baseline_m,
                        "time_gap_s": selection.time_gap_s,
                        "selector_score": selection.score,
                        "match_score": match["score"],
                        "match_method": match.get("match_method", "geometry"),
                        "object_id_match": bool(match.get("object_id_match", False)),
                        "anchor_distance_m": match["anchor_distance_m"],
                        "centroid_shift_px": match["centroid_shift_px"],
                        "bbox_iou": match["bbox_iou"],
                        "z_range_overlap": match["z_range_overlap"],
                        "projection_used": bool(match.get("projection_used", False)),
                    }
                )
            measurement = fuse_candidate_measurement(candidate, matched_history, cfg)
            measurement["candidate_id"] = candidate.candidate_id
            measurement["roi_id"] = candidate.roi_id
            measurement["object_id"] = _candidate_object_id(candidate)
            if not matched_history and measurement.get("status") != "ok":
                measurement["status"] = "no_candidate_match"
                measurement["reject_reasons"] = ["no_candidate_match"]
            applied = apply_temporal_measurement(candidate, measurement, cfg)
            measurement = candidate.metadata["temporal_measurement"]
            measurement["applied"] = bool(applied)
            measurements.append(measurement)
        return measurements

    def add_snapshot(
        self,
        *,
        frame_bgr: np.ndarray | None,
        frame_id: str,
        frame_index: int,
        timestamp_s: float,
        cumulative_forward_m: float,
        candidates: list[CandidateObservation],
    ) -> None:
        if not self.enabled:
            return
        self.history.append(
            TemporalFrameSnapshot(
                frame_id=frame_id,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                cumulative_forward_m=cumulative_forward_m,
                candidates=list(candidates),
                image_gray=None,
            )
        )
        history_size = max(1, _cfg_int(self.config, "history_size", 30))
        if len(self.history) > history_size:
            self.history = self.history[-history_size:]
