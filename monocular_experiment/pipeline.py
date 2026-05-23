from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import resolve_path
from .control import ControlEvent, LogControlBackend, NoopControlBackend
from .depth_model import infer_relative_depth
from .geometry import (
    bbox_iou,
    estimate_scale_factor_from_relative_depth_plane,
    infer_forward_displacement_m,
    infer_speed_mps,
    sample_points_from_mask,
)
from .ground_plane import estimate_ground_plane
from .io_utils import ensure_dir, list_frame_paths, load_yaml, read_image, read_motion_csv, save_json, write_jsonl
from .models import FrameState, ScaleAlignmentResult
from .obstacles import (
    build_candidate_observations,
    build_depth_residual_roi_candidates,
    classify_all_candidates,
    summarize_cross_frame_matches,
)
from .risk import assess_tracked_objects, attach_risk_to_objects
from .segmentation import segment_frame
from .temporal_measurement import TemporalMeasurementManager
from .tracking import ObjectTracker
from .visualization import save_overlay


def _backend_for_run(output_dir: Path, backend_name: str):
    if backend_name == "log":
        return LogControlBackend(output_dir / "risk_events.csv")
    return NoopControlBackend()


def _risk_assessment_by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["object_id"]: item for item in items}


_DECISION_RANK = {"safe": 0, "warning": 1, "danger": 2}


class _RiskEventEmitter:
    def __init__(self, config: dict[str, Any]):
        self.min_hits = int(config.get("emit_min_consecutive_hits", config.get("min_consecutive_hits", 3)) or 0)
        self.emit_predicted = bool(config.get("emit_predicted_objects", True))
        self.min_support_points = int(config.get("emit_min_support_points", 0) or 0)
        self.cooldown_frames = int(config.get("emit_cooldown_frames", 0) or 0)
        self.safe_clear_frames = int(config.get("emit_safe_clear_frames", 0) or 0)
        self.require_temporal_for_fallback = bool(config.get("require_temporal_for_fallback", False))
        self._last_emitted: dict[str, dict[str, Any]] = {}
        self._safe_counts: dict[str, int] = {}
        self._frame_index = 0

    def _is_fallback_object(self, obj: Any) -> bool:
        source = str((getattr(obj, "metadata", {}) or {}).get("roi_source", ""))
        return source.startswith("fallback")

    def _temporal_applied(self, obj: Any) -> bool:
        temporal = (getattr(obj, "metadata", {}) or {}).get("temporal_measurement") or {}
        return bool(temporal.get("applied"))

    def _eligible(self, obj: Any, risk: dict[str, Any]) -> bool:
        if risk.get("decision") == "safe":
            return False
        if int(getattr(obj, "hit_count", 0)) < self.min_hits:
            return False
        if not self.emit_predicted and getattr(obj, "state", "") == "predicted":
            return False
        if len(getattr(obj, "support_points", []) or []) < self.min_support_points:
            return False
        if self.require_temporal_for_fallback and self._is_fallback_object(obj) and not self._temporal_applied(obj):
            return False
        return True

    def _allowed_by_cooldown(self, object_id: str, decision: str) -> bool:
        previous = self._last_emitted.get(object_id)
        if previous is None:
            return True
        previous_decision = str(previous.get("decision", "safe"))
        previous_frame = int(previous.get("frame_index", -10**9))
        if _DECISION_RANK.get(decision, 0) > _DECISION_RANK.get(previous_decision, 0):
            return True
        return self._frame_index - previous_frame > self.cooldown_frames

    def filter_events(self, tracked_objects: list[Any], risks_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        current_ids = {obj.object_id for obj in tracked_objects}
        for obj in tracked_objects:
            risk = risks_by_id.get(obj.object_id)
            if risk is None:
                continue
            if risk.get("decision") == "safe":
                self._safe_counts[obj.object_id] = self._safe_counts.get(obj.object_id, 0) + 1
                if self._safe_counts[obj.object_id] >= self.safe_clear_frames:
                    self._last_emitted.pop(obj.object_id, None)
                continue
            self._safe_counts[obj.object_id] = 0
            if not self._eligible(obj, risk):
                continue
            decision = str(risk.get("decision", "safe"))
            if not self._allowed_by_cooldown(obj.object_id, decision):
                continue
            event = dict(risk)
            event["emitted"] = True
            event["metadata"] = {
                **dict(event.get("metadata") or {}),
                "hit_count": int(getattr(obj, "hit_count", 0)),
                "state": str(getattr(obj, "state", "")),
                "support_point_count": int(len(getattr(obj, "support_points", []) or [])),
            }
            emitted.append(event)
            self._last_emitted[obj.object_id] = {"decision": decision, "frame_index": self._frame_index}
        stale_ids = set(self._last_emitted) - current_ids
        for object_id in stale_ids:
            self._last_emitted.pop(object_id, None)
            self._safe_counts.pop(object_id, None)
        self._frame_index += 1
        return emitted


def _depth_stats(depth_map: np.ndarray, confidence: np.ndarray | None, backend: str) -> dict[str, Any]:
    valid = np.isfinite(depth_map) & (depth_map > 0.0)
    return {
        "backend": backend,
        "shape": [int(depth_map.shape[0]), int(depth_map.shape[1])],
        "valid_count": int(np.count_nonzero(valid)),
        "relative_depth_min": float(np.min(depth_map[valid])) if np.any(valid) else 0.0,
        "relative_depth_max": float(np.max(depth_map[valid])) if np.any(valid) else 0.0,
        "relative_depth_mean": float(np.mean(depth_map[valid])) if np.any(valid) else 0.0,
        "confidence_mean": float(np.mean(confidence)) if confidence is not None and confidence.size else None,
    }


def _resolve_obstacle_analysis_mask(frontend: Any) -> tuple[np.ndarray, str]:
    obstacle_mask = getattr(frontend, "obstacle_analysis_mask", None)
    if obstacle_mask is not None:
        return np.asarray(obstacle_mask, dtype=bool), "processing_roi"
    return np.asarray(frontend.analysis_mask, dtype=bool), "frontend_analysis_mask"


def _bbox_area_px(bbox: list[int]) -> int:
    return int(max(0, int(bbox[2]) * int(bbox[3])))


def _bbox_intersection_area_px(a: list[int], b: list[int]) -> int:
    ax, ay, aw, ah = [int(value) for value in a]
    bx, by, bw, bh = [int(value) for value in b]
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    return int(max(0, x1 - x0) * max(0, y1 - y0))


def _intersection_over_smaller_area(a: list[int], b: list[int]) -> float:
    smaller = min(_bbox_area_px(a), _bbox_area_px(b))
    if smaller <= 0:
        return 0.0
    return float(_bbox_intersection_area_px(a, b) / smaller)


def _roi_source_priority(candidate: Any, depth_roi_cfg: dict[str, Any]) -> int:
    source = str(getattr(candidate, "source", ""))
    priority = depth_roi_cfg.get("source_priority") or [
        "depth_residual_contour",
        "hybrid",
        "negative_space",
        "non_road_near_road",
        "fallback",
    ]
    for index, name in enumerate(priority):
        value = str(name)
        if source == value or (value == "fallback" and source.startswith("fallback")):
            return index
    return len(priority)


def _roi_sort_key(candidate: Any, depth_roi_cfg: dict[str, Any]) -> tuple[Any, ...]:
    metadata = getattr(candidate, "metadata", {}) or {}
    point_count = int(metadata.get("point_count", metadata.get("support_count", 0)) or 0)
    mask_fill_ratio = float(metadata.get("mask_fill_ratio", 0.0) or 0.0)
    return (
        _roi_source_priority(candidate, depth_roi_cfg),
        -point_count,
        -mask_fill_ratio,
        bool(getattr(candidate, "touch_border", False)),
        _bbox_area_px(candidate.bbox),
        str(getattr(candidate, "roi_id", "")),
    )


def _filter_roi_candidates(
    candidates: list[Any],
    depth_roi_cfg: dict[str, Any],
    frame_shape: tuple[int, int, int] | tuple[int, int] | None,
) -> list[Any]:
    if frame_shape is None:
        return list(candidates)
    frame_h = int(frame_shape[0])
    frame_w = int(frame_shape[1])
    max_area_ratio = depth_roi_cfg.get("max_roi_area_ratio")
    max_area = None if max_area_ratio is None else float(max_area_ratio) * float(frame_h * frame_w)
    max_bottom_ratio = depth_roi_cfg.get("max_roi_bottom_ratio")
    max_bottom_px = None if max_bottom_ratio is None else float(max_bottom_ratio) * float(frame_h)
    filtered: list[Any] = []
    for candidate in candidates:
        x, y, w, h = [int(value) for value in candidate.bbox]
        if max_area is not None and _bbox_area_px(candidate.bbox) > max_area:
            continue
        if max_bottom_px is not None and float(y + h) > max_bottom_px:
            continue
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            continue
        filtered.append(candidate)
    return filtered


def _limit_roi_candidates(candidates: list[Any], depth_roi_cfg: dict[str, Any]) -> list[Any]:
    max_selected = int(depth_roi_cfg.get("max_selected_rois_per_frame", 0) or 0)
    if max_selected <= 0 or len(candidates) <= max_selected:
        return candidates
    return sorted(candidates, key=lambda item: _roi_sort_key(item, depth_roi_cfg))[:max_selected]


def _deduplicate_roi_candidates(
    candidates: list[Any],
    iou_threshold: float,
    depth_roi_cfg: dict[str, Any] | None = None,
) -> list[Any]:
    cfg = depth_roi_cfg or {}
    containment_threshold = float(cfg.get("containment_threshold", 0.90))
    selected: list[Any] = []
    for candidate in sorted(candidates, key=lambda item: _roi_sort_key(item, cfg)):
        candidate_area = _bbox_area_px(candidate.bbox)
        suppressed = False
        for kept in selected:
            kept_area = _bbox_area_px(kept.bbox)
            if bbox_iou(candidate.bbox, kept.bbox) >= iou_threshold:
                suppressed = True
                break
            if _intersection_over_smaller_area(candidate.bbox, kept.bbox) >= containment_threshold:
                if candidate_area >= kept_area or _roi_source_priority(candidate, cfg) >= _roi_source_priority(kept, cfg):
                    suppressed = True
                    break
        if not suppressed:
            selected.append(candidate)
    return selected


def _select_roi_candidates_for_obstacles(
    frontend_rois: list[Any],
    depth_roi_candidates: list[Any],
    candidate_cfg: dict[str, Any],
    frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    frontend_backend: str | None = None,
) -> list[Any]:
    depth_roi_cfg = candidate_cfg.get("depth_contour_roi") or {}
    if not bool(depth_roi_cfg.get("enabled", False)):
        return frontend_rois

    threshold = float(depth_roi_cfg.get("deduplicate_iou_threshold", 0.7))
    filtered_depth = _filter_roi_candidates(depth_roi_candidates, depth_roi_cfg, frame_shape)
    filtered_frontend = _filter_roi_candidates(frontend_rois, depth_roi_cfg, frame_shape)
    selection_mode = str(depth_roi_cfg.get("selection_mode", "depth_first")).lower()
    if selection_mode == "union":
        combined = [*filtered_depth, *filtered_frontend]
        return _limit_roi_candidates(_deduplicate_roi_candidates(combined, threshold, depth_roi_cfg), depth_roi_cfg)

    if selection_mode in {"depth_plus_verified_frontend", "source_aware"}:
        backend_is_fallback = str(frontend_backend or "").startswith("fallback")
        verified_frontend = [item for item in filtered_frontend if not backend_is_fallback and not str(item.source).startswith("fallback")]
        fallback_frontend = [item for item in filtered_frontend if backend_is_fallback or str(item.source).startswith("fallback")]
        deduplicated = _deduplicate_roi_candidates([*filtered_depth, *verified_frontend], threshold, depth_roi_cfg)
        verified_ids = {id(item) for item in verified_frontend}
        selected_frontend = [item for item in deduplicated if id(item) in verified_ids]
        selected_depth = [item for item in deduplicated if id(item) not in verified_ids]
        max_selected = int(depth_roi_cfg.get("max_selected_rois_per_frame", 0) or 0)
        if max_selected > 0:
            depth_budget = max(0, max_selected - len(selected_frontend))
            selected_depth = sorted(selected_depth, key=lambda item: _roi_sort_key(item, depth_roi_cfg))[:depth_budget]
        combined = [*selected_frontend, *selected_depth]
        if not combined and bool(depth_roi_cfg.get("fallback_to_frontend_rois", True)):
            combined = fallback_frontend
        return sorted(combined, key=lambda item: _roi_sort_key(item, depth_roi_cfg))

    if filtered_depth:
        return _limit_roi_candidates(filtered_depth, depth_roi_cfg)
    if bool(depth_roi_cfg.get("fallback_to_frontend_rois", True)):
        return _limit_roi_candidates(filtered_frontend, depth_roi_cfg)
    return []


def _export_roi_diagnostics(candidates: list[Any]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", {}) or {}
        exported.append(
            {
                "roi_id": str(getattr(candidate, "roi_id", "")),
                "bbox": [int(value) for value in getattr(candidate, "bbox", [0, 0, 0, 0])],
                "source": str(getattr(candidate, "source", "")),
                "area_px": int(getattr(candidate, "area_px", 0) or 0),
                "touch_border": bool(getattr(candidate, "touch_border", False)),
                "metadata": {
                    key: metadata.get(key)
                    for key in ["sign", "raw_bbox", "point_count", "mask_fill_ratio", "border_touch"]
                    if key in metadata
                },
            }
        )
    return exported


def _export_obstacle_rois(
    candidate_clusters: list[Any],
    frame_shape: tuple[int, int, int],
) -> list[dict[str, Any]]:
    frame_h, frame_w = frame_shape[:2]
    exported: list[dict[str, Any]] = []
    for candidate in candidate_clusters:
        x, y, w, h = [int(value) for value in candidate.bbox]
        touch_border = bool(x <= 0 or y <= 0 or (x + w) >= frame_w or (y + h) >= frame_h)
        exported.append(
            {
                "roi_id": str(candidate.candidate_id),
                "candidate_id": str(candidate.candidate_id),
                "object_id": candidate.object_id,
                "bbox": [x, y, w, h],
                "area_px": int(max(1, w * h)),
                "touch_border": touch_border,
                "source": "candidate_cluster",
                "candidate_type": str(getattr(candidate, "candidate_type", "")),
                "metadata": {
                    "candidate_id": str(candidate.candidate_id),
                    "object_id": candidate.object_id,
                    "obstacle_type": str(candidate.obstacle_type),
                    "candidate_type": str(getattr(candidate, "candidate_type", "")),
                    "distance_m": float(candidate.distance_m),
                    "height_m": float(candidate.height_m),
                    "width_m": float(candidate.width_m),
                    "z_range_m": [float(candidate.z_range_m[0]), float(candidate.z_range_m[1])],
                },
            }
        )
    return exported


def run_pipeline(
    config: dict[str, Any],
    dataset_root: str | Path,
    sequence_id: str,
    output_dir: str | Path,
    control_backend: str = "log",
) -> dict[str, Any]:
    """執行新版論文對齊單眼管線。"""

    dataset_dir = Path(dataset_root) / sequence_id
    frames_dir = dataset_dir / "frames"
    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise ValueError(f"Sequence {sequence_id} must contain at least one frame.")

    motion = read_motion_csv(dataset_dir / "motion.csv")
    if len(motion) != len(frame_paths):
        raise ValueError("motion.csv row count must match the number of frames.")

    intrinsics_path = resolve_path(config, config["calibration"]["intrinsics_path"])
    extrinsics_path = resolve_path(config, config["calibration"]["extrinsics_path"])
    intrinsics = load_yaml(intrinsics_path)
    extrinsics = load_yaml(extrinsics_path)

    output_dir = ensure_dir(output_dir)
    overlays_dir = ensure_dir(output_dir / "overlays")
    backend = _backend_for_run(Path(output_dir), control_backend)
    tracker = ObjectTracker(config["tracking"])
    temporal_cfg = dict(config.get("temporal_measurement", {}))
    temporal_manager = TemporalMeasurementManager(temporal_cfg)
    cumulative_forward_m = 0.0
    tracking_cfg = dict(config["tracking"])
    min_consecutive_hits = int(tracking_cfg.get("min_consecutive_hits", 3))

    results: list[dict[str, Any]] = []
    frame_latency_ms: list[float] = []
    risk_event_count = 0
    risk_emitter = _RiskEventEmitter(config["risk"])

    roi_cfg = dict(config["roi"])
    roi_cfg["_config_dir"] = config.get("_config_dir")
    depth_cfg = dict(config["depth_model"])
    depth_cfg["_config_dir"] = config.get("_config_dir")
    scale_cfg = dict(config["scale_alignment"])
    plane_cfg = config["plane"]
    candidate_cfg = config["candidate_generation"]
    risk_cfg = config["risk"]

    prev_candidates = []

    for index, frame_path in enumerate(tqdm(frame_paths, desc=f"run-pipeline:{sequence_id}")):
        frame_id = frame_path.stem
        frame = read_image(frame_path)
        motion_curr = motion.iloc[index]
        motion_prev = motion.iloc[index - 1] if index > 0 else None
        timestamp_curr_s = float(motion_curr["timestamp_s"])
        timestamp_prev_s = float(motion_prev["timestamp_s"]) if motion_prev is not None else None
        speed_value = (
            float(motion_curr["speed_mps"])
            if "speed_mps" in motion_curr.index and not np.isnan(motion_curr["speed_mps"])
            else None
        )
        forward_displacement_value = (
            float(motion_curr["forward_displacement_m"])
            if "forward_displacement_m" in motion_curr.index and not np.isnan(motion_curr["forward_displacement_m"])
            else None
        )
        speed_mps = infer_speed_mps(
            timestamp_prev_s=timestamp_prev_s,
            timestamp_curr_s=timestamp_curr_s,
            speed_value=speed_value,
            forward_displacement_m=forward_displacement_value,
        )
        forward_displacement_m = infer_forward_displacement_m(
            timestamp_prev_s=timestamp_prev_s,
            timestamp_curr_s=timestamp_curr_s,
            speed_value=speed_value,
            forward_displacement_m=forward_displacement_value,
        )
        if "cumulative_distance_m" in motion_curr.index and not np.isnan(motion_curr["cumulative_distance_m"]):
            cumulative_forward_m = float(motion_curr["cumulative_distance_m"])
        else:
            cumulative_forward_m += max(0.0, float(forward_displacement_m))

        start_total = time.perf_counter()

        start = time.perf_counter()
        frontend = segment_frame(frame, roi_cfg)
        frontend_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth_output = infer_relative_depth(frame, depth_cfg)
        relative_depth = np.asarray(depth_output["relative_depth"], dtype=np.float32)
        confidence = (
            np.asarray(depth_output["confidence"], dtype=np.float32)
            if depth_output.get("confidence") is not None
            else None
        )
        depth_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        scale_result_rel = estimate_scale_factor_from_relative_depth_plane(
            relative_depth_map=relative_depth,
            road_mask=frontend.road_mask,
            intrinsics=intrinsics,
            camera_height_m=float(scale_cfg["camera_height_m"]),
            stride=int(scale_cfg["sampling_stride"]),
            min_rel_depth=float(scale_cfg["min_relative_depth"]),
            trim_percentile=float(scale_cfg["trim_percentile"]),
            min_candidates=int(scale_cfg["min_candidates"]),
            ransac_iterations=int(plane_cfg.get("ransac_iterations", 80)),
            ransac_threshold_rel=float(plane_cfg.get("ransac_distance_threshold_m", 0.05)),
            ransac_min_inliers=int(plane_cfg.get("ransac_min_inliers", 20)),
        )
        scale_factor = scale_result_rel.scale_factor
        absolute_depth = relative_depth * float(scale_factor)
        scale_result = ScaleAlignmentResult(
            scale_factor=scale_factor,
            candidate_count=int(scale_result_rel.metadata.get("candidate_count", 0)),
            selected_count=int(scale_result_rel.metadata.get("inlier_count", 0)),
            status=scale_result_rel.status,
            metadata={
                **scale_result_rel.metadata,
                "h_hat_cam": scale_result_rel.h_hat_cam,
                "plane_normal_rel": scale_result_rel.plane_normal_rel,
                "plane_offset_rel": scale_result_rel.plane_offset_rel,
            },
        )
        scale_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        road_sample = sample_points_from_mask(
            depth_map_m=absolute_depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            mask=frontend.road_mask,
            stride=int(plane_cfg["sampling_stride"]),
            max_points=int(plane_cfg.get("max_points", 0)),
        )
        plane = estimate_ground_plane(road_sample["world_points"], plane_cfg)
        plane_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        obstacle_analysis_mask, obstacle_analysis_mask_source = _resolve_obstacle_analysis_mask(frontend)
        candidate_sample = sample_points_from_mask(
            depth_map_m=absolute_depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            mask=obstacle_analysis_mask,
            stride=int(candidate_cfg["sampling_stride"]),
            max_points=int(candidate_cfg.get("max_points", 0)),
        )
        depth_roi_candidates = build_depth_residual_roi_candidates(
            candidate_image_points=candidate_sample["image_points"],
            candidate_depths_m=candidate_sample["depths_m"],
            plane=plane,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            frame_shape=frame.shape,
            analysis_mask=obstacle_analysis_mask,
            config=candidate_cfg,
        )
        roi_candidates_for_candidates = _select_roi_candidates_for_obstacles(
            frontend.roi_candidates,
            depth_roi_candidates,
            candidate_cfg,
            frame_shape=frame.shape,
            frontend_backend=frontend.backend,
        )
        candidate_clusters = build_candidate_observations(
            roi_candidates=roi_candidates_for_candidates,
            candidate_image_points=candidate_sample["image_points"],
            candidate_world_points=candidate_sample["world_points"],
            candidate_depths_m=candidate_sample["depths_m"],
            plane=plane,
            config=candidate_cfg,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
        candidate_clusters = classify_all_candidates(
            candidates=candidate_clusters,
            road_mask=frontend.road_mask,
            config=candidate_cfg,
        )
        temporal_measurements: list[dict[str, Any]] = []
        temporal_ms = 0.0
        cross_frame_matches = summarize_cross_frame_matches(
            previous_candidates=prev_candidates,
            current_candidates=candidate_clusters,
            config=config["tracking"],
            forward_displacement_m=forward_displacement_m,
        )
        candidate_ms = (time.perf_counter() - start) * 1000.0

        def _measure_temporal_after_assignment(assigned_candidates: list[Any]) -> None:
            nonlocal temporal_measurements, temporal_ms
            start_temporal = time.perf_counter()
            temporal_measurements = temporal_manager.measure_candidates(
                frame_bgr=frame,
                frame_id=frame_id,
                frame_index=index,
                timestamp_s=timestamp_curr_s,
                cumulative_forward_m=cumulative_forward_m,
                candidates=assigned_candidates,
                config=temporal_cfg,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
            temporal_ms = (time.perf_counter() - start_temporal) * 1000.0

        start = time.perf_counter()
        tracked_objects = tracker.update(
            frame_id=frame_id,
            candidates=candidate_clusters,
            cross_frame_matches=cross_frame_matches,
            forward_displacement_m=forward_displacement_m,
            post_assignment_callback=_measure_temporal_after_assignment,
        )
        exported_roi_candidates = _export_obstacle_rois(candidate_clusters, frame.shape)
        risk_assessments = assess_tracked_objects(
            tracked_objects=tracked_objects,
            road_world_points=road_sample["world_points"],
            speed_mps=speed_mps,
            config=risk_cfg,
        )
        tracked_objects = attach_risk_to_objects(tracked_objects, risk_assessments)
        raw_risk_events = [item.to_dict() for item in risk_assessments]
        risk_by_id = _risk_assessment_by_id(raw_risk_events)
        risk_events = risk_emitter.filter_events(tracked_objects, risk_by_id)
        for risk in risk_events:
            risk_event_count += 1
            backend.emit(ControlEvent(frame_id=frame_id, decision=risk["decision"], object_id=risk["object_id"]))
        tracking_risk_ms = (time.perf_counter() - start) * 1000.0

        timing_ms = {
            "frontend": frontend_ms,
            "depth_inference": depth_ms,
            "scale_alignment": scale_ms,
            "plane_fit": plane_ms,
            "candidate_generation": candidate_ms,
            "temporal_measurement": temporal_ms,
            "tracking_and_risk": tracking_risk_ms,
            "total": (time.perf_counter() - start_total) * 1000.0,
        }
        frame_latency_ms.append(timing_ms["total"])

        road_mask_stats = frontend.road_mask_stats()
        road_mask_stats["roi_count"] = len(exported_roi_candidates)
        road_mask_stats["processing_roi"] = frontend.processing_roi
        road_mask_stats["obstacle_analysis_mask_source"] = obstacle_analysis_mask_source
        road_mask_stats["obstacle_analysis_area_px"] = int(np.count_nonzero(obstacle_analysis_mask))
        road_mask_stats["frontend_roi_count"] = len(frontend.roi_candidates)
        road_mask_stats["depth_roi_count"] = len(depth_roi_candidates)
        road_mask_stats["selected_roi_count"] = len(roi_candidates_for_candidates)
        road_mask_stats["candidate_roi_source_counts"] = {
            source: sum(1 for item in roi_candidates_for_candidates if item.source == source)
            for source in sorted({item.source for item in roi_candidates_for_candidates})
        }
        road_mask_stats["candidate_roi_source"] = (
            roi_candidates_for_candidates[0].source if roi_candidates_for_candidates else "none"
        )
        road_mask_stats["frontend_roi_diagnostics"] = _export_roi_diagnostics(frontend.roi_candidates)
        road_mask_stats["depth_roi_diagnostics"] = _export_roi_diagnostics(depth_roi_candidates)
        road_mask_stats["selected_roi_diagnostics"] = _export_roi_diagnostics(roi_candidates_for_candidates)
        road_mask_stats["candidate_object_id_count"] = sum(1 for item in candidate_clusters if item.object_id)
        road_mask_stats["stable_object_id_count"] = len({item.object_id for item in candidate_clusters if item.object_id})
        road_mask_stats["temporal_keyframe_count"] = temporal_manager.keyframe_count()
        road_mask_stats["temporal_measurement_attempted_count"] = len(temporal_measurements)
        road_mask_stats["temporal_measurement_ok_count"] = sum(
            1 for item in temporal_measurements if item.get("status") == "ok"
        )
        road_mask_stats["temporal_measurement_applied_count"] = sum(
            1 for item in temporal_measurements if item.get("applied")
        )

        frame_state = FrameState(
            frame_id=frame_id,
            depth_stats=_depth_stats(relative_depth, confidence, str(depth_output["backend"])),
            scale_alignment=scale_result.to_dict(),
            road_mask_stats=road_mask_stats,
            plane_model=plane.to_dict(),
            roi_candidates=exported_roi_candidates,
            candidate_clusters=[item.to_dict() for item in candidate_clusters],
            cross_frame_matches=[item.to_dict() for item in cross_frame_matches],
            tracked_objects=[item.to_dict() for item in tracked_objects],
            risk_events=risk_events,
            timing_ms=timing_ms,
            temporal_measurements=temporal_measurements,
            raw_risk_assessments=raw_risk_events,
        )
        results.append(frame_state.to_dict())

        if config.get("visualization", {}).get("save_overlays", True):
            save_overlay(
                frame_bgr=frame,
                frame_id=frame_id,
                frontend=frontend,
                road_points=road_sample["image_points"],
                plane=plane.to_dict(),
                tracked_objects=[item.to_dict() for item in tracked_objects],
                output_path=overlays_dir / f"{frame_id}.png",
                candidate_rois=roi_candidates_for_candidates,
            )

        temporal_manager.add_snapshot(
            frame_bgr=frame,
            frame_id=frame_id,
            frame_index=index,
            timestamp_s=timestamp_curr_s,
            cumulative_forward_m=cumulative_forward_m,
            candidates=candidate_clusters,
        )
        prev_candidates = candidate_clusters

    write_jsonl(output_dir / "frame_states.jsonl", results)
    summary = {
        "sequence_id": sequence_id,
        "frames_processed": len(results),
        "mean_latency_ms": float(np.mean(frame_latency_ms)) if frame_latency_ms else 0.0,
        "max_latency_ms": float(np.max(frame_latency_ms)) if frame_latency_ms else 0.0,
        "risk_event_count": int(risk_event_count),
    }
    save_json(output_dir / "pipeline_summary.json", summary)
    return summary
