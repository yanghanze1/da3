from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from .geometry import bbox_iou, points_in_bbox, signed_plane_distance
from .models import CandidateObservation, CrossFrameMatch, PlaneEstimate, RoiCandidate


class _CandidateObservationList(list):
    pass


def _cluster_bbox(points_2d: np.ndarray) -> list[int]:
    min_xy = np.floor(points_2d.min(axis=0)).astype(int)
    max_xy = np.ceil(points_2d.max(axis=0)).astype(int)
    wh = np.maximum(1, max_xy - min_xy)
    return [int(min_xy[0]), int(min_xy[1]), int(wh[0]), int(wh[1])]


def _pad_bbox_within_bounds(bbox: list[int], bounds: list[int], config: dict[str, Any]) -> list[int]:
    default_padding = int(config.get("cluster_bbox_padding_px", 0))
    pad_x = int(config.get("cluster_bbox_padding_x_px", default_padding))
    pad_y = int(config.get("cluster_bbox_padding_y_px", default_padding))
    pad_top = int(config.get("cluster_bbox_padding_top_px", pad_y))
    pad_bottom = int(config.get("cluster_bbox_padding_bottom_px", pad_y))
    if pad_x <= 0 and pad_top <= 0 and pad_bottom <= 0:
        return list(bbox)
    x, y, w, h = [int(value) for value in bbox]
    bx, by, bw, bh = [int(value) for value in bounds]
    x0 = max(bx, x - pad_x)
    y0 = max(by, y - pad_top)
    x1 = min(bx + bw, x + w + pad_x)
    y1 = min(by + bh, y + h + pad_bottom)
    return [int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))]


def _optional_float(config: dict[str, Any], key: str) -> float | None:
    value = config.get(key)
    return None if value is None else float(value)


def _valid_dimension(value: float, max_value: float | None) -> bool:
    return np.isfinite(value) and value >= 0.0 and (max_value is None or value <= max_value)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _int_bbox(value: Any) -> list[int]:
    bbox = list(value or [0, 0, 0, 0])[:4]
    while len(bbox) < 4:
        bbox.append(0)
    return [int(v) for v in bbox]


def _finite_range(values: np.ndarray) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None
    return [float(np.min(finite)), float(np.max(finite))]


def _sample_image_points(points_2d: np.ndarray, max_points: int) -> list[list[int]]:
    if len(points_2d) == 0 or max_points == 0:
        return []
    points = np.asarray(points_2d[:, :2], dtype=np.float64)
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
        points = points[indices]
    return np.rint(points).astype(int).tolist()


def _new_roi_cluster_trace(
    roi: RoiCandidate,
    roi_image: np.ndarray,
    roi_world: np.ndarray,
    delta_i: np.ndarray | None,
    positive_mask: np.ndarray | None,
    negative_mask: np.ndarray | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    metadata = roi.metadata or {}
    max_points = int(config.get("trace_max_points_per_sign", 1000))
    valid_delta = np.isfinite(delta_i) if delta_i is not None else np.zeros((len(roi_world),), dtype=bool)
    positive = positive_mask if positive_mask is not None else np.zeros((len(roi_world),), dtype=bool)
    negative = negative_mask if negative_mask is not None else np.zeros((len(roi_world),), dtype=bool)
    return {
        "roi_id": str(roi.roi_id),
        "source": str(roi.source),
        "bbox": _int_bbox(roi.bbox),
        "raw_bbox": _int_bbox(metadata.get("raw_bbox", roi.bbox)),
        "area_px": int(roi.area_px),
        "mask_fill_ratio": _finite_float(metadata.get("mask_fill_ratio")),
        "point_count": int(metadata.get("point_count", len(roi_world)) or 0),
        "total_point_count": int(len(roi_world)),
        "valid_residual_point_count": int(np.count_nonzero(valid_delta)),
        "positive_abnormal_point_count": int(np.count_nonzero(positive)),
        "negative_abnormal_point_count": int(np.count_nonzero(negative)),
        "positive_abnormal_image_points_sample": _sample_image_points(roi_image[positive], max_points),
        "negative_abnormal_image_points_sample": _sample_image_points(roi_image[negative], max_points),
        "abnormal_point_sample_limit": max_points,
        "candidate_generated": False,
        "candidate_count": 0,
        "candidate_ids": [],
        "cluster_count": 0,
        "dbscan_input_point_count": 0,
        "dbscan_noise_point_count": 0,
        "primary_reject_reason": "accepted",
        "cluster_reject_reason_counts": {},
        "sign_traces": [],
    }


def _record_reject(sign_trace: dict[str, Any], reason: str) -> None:
    sign_trace.setdefault("reject_reasons", []).append(reason)


def _configured_sources(config: dict[str, Any], key: str) -> set[str]:
    values = config.get(key) or []
    return {str(value) for value in values}


def _depth_supported_relaxation_enabled(
    *,
    roi: RoiCandidate,
    config: dict[str, Any],
    reference_metric: str,
    cluster_expected_depths: np.ndarray | None,
) -> bool:
    relaxed_ratio = config.get("min_abnormal_ratio_depth_supported")
    if relaxed_ratio is None:
        return False
    if reference_metric != "observed_minus_plane_depth_m":
        return False
    allowed_sources = _configured_sources(config, "depth_supported_relaxed_sources")
    if str(roi.source) not in allowed_sources:
        return False
    return cluster_expected_depths is not None and bool(np.any(np.isfinite(cluster_expected_depths)))


def _finalize_roi_cluster_trace(trace: dict[str, Any]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    cluster_count = 0
    noise_count = 0
    dbscan_input_count = 0
    for sign_trace in trace.get("sign_traces", []):
        dbscan = sign_trace.get("dbscan") or {}
        cluster_count += int(dbscan.get("cluster_count", 0) or 0)
        noise_count += int(dbscan.get("noise_point_count", 0) or 0)
        dbscan_input_count += int(dbscan.get("input_point_count", 0) or 0)
        for reason in sign_trace.get("reject_reasons", []):
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    if int(trace.get("total_point_count", 0) or 0) == 0:
        reason_counts["no_roi_points"] = reason_counts.get("no_roi_points", 0) + 1

    trace["cluster_count"] = int(cluster_count)
    trace["dbscan_input_point_count"] = int(dbscan_input_count)
    trace["dbscan_noise_point_count"] = int(noise_count)
    trace["candidate_generated"] = bool(trace.get("candidate_count", 0))
    trace["cluster_reject_reason_counts"] = reason_counts
    if trace["candidate_generated"]:
        trace["primary_reject_reason"] = "accepted"
    else:
        priority = [
            "no_roi_points",
            "too_few_signed_points",
            "too_few_forward_points",
            "dbscan_noise_only",
            "cluster_too_few_forward_points",
            "abnormal_ratio_too_low",
            "bbox_area_too_small",
            "invalid_height_or_width",
        ]
        trace["primary_reject_reason"] = next((reason for reason in priority if reason_counts.get(reason, 0) > 0), "too_few_signed_points")
    return trace


def _slice_points_in_roi(
    roi: RoiCandidate,
    image_points: np.ndarray,
    world_points: np.ndarray,
    depths_m: np.ndarray,
    expected_depths_m: np.ndarray | None = None,
    residuals_m: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    roi_mask = points_in_bbox(image_points, roi.bbox)
    sliced: dict[str, np.ndarray] = {
        "image_points": image_points[roi_mask],
        "world_points": world_points[roi_mask],
        "depths_m": depths_m[roi_mask],
    }
    if expected_depths_m is not None:
        sliced["expected_depths_m"] = expected_depths_m[roi_mask]
    if residuals_m is not None:
        sliced["residuals_m"] = residuals_m[roi_mask]
    return sliced


def _observed_minus_plane_depth_residual(
    image_points: np.ndarray,
    observed_depths_m: np.ndarray,
    intrinsics: dict[str, Any],
    extrinsics: dict[str, Any],
    plane: PlaneEstimate,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute residual = observed depth - plane-expected depth per sampled point."""

    if len(image_points) == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])

    rotation = np.asarray(extrinsics["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(extrinsics["translation_vector"], dtype=np.float64).reshape(3)
    inv_rotation = np.linalg.inv(rotation)
    camera_origin_world = -(inv_rotation @ translation)

    u = image_points[:, 0].astype(np.float64)
    v = image_points[:, 1].astype(np.float64)
    rays_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
    rays_world = (inv_rotation @ rays_cam.T).T

    normal = np.asarray(plane.normal, dtype=np.float64)
    offset = float(plane.offset)
    denom = rays_world @ normal
    numer = -(camera_origin_world @ normal + offset)
    lam = np.divide(
        numer,
        denom,
        out=np.full_like(denom, np.nan, dtype=np.float64),
        where=np.abs(denom) > 1e-9,
    )

    expected_depths = np.full_like(observed_depths_m, np.nan, dtype=np.float64)
    valid = np.isfinite(lam) & (lam > 0.0)
    if np.any(valid):
        intersections_world = camera_origin_world[None, :] + rays_world * lam[:, None]
        intersections_cam = (rotation @ intersections_world.T).T + translation
        z_plane = intersections_cam[:, 2]
        valid &= np.isfinite(z_plane) & (z_plane > 0.0)
        expected_depths[valid] = z_plane[valid]

    residuals = np.full_like(observed_depths_m, np.nan, dtype=np.float64)
    valid_depth = np.isfinite(observed_depths_m) & (observed_depths_m > 0.0)
    valid_residual = valid & valid_depth & np.isfinite(expected_depths)
    residuals[valid_residual] = observed_depths_m[valid_residual] - expected_depths[valid_residual]
    return residuals, expected_depths


def _append_clustered_candidates(
    *,
    observations: list[CandidateObservation],
    next_id: int,
    roi: RoiCandidate,
    roi_world: np.ndarray,
    roi_depths: np.ndarray,
    signed_mask: np.ndarray,
    roi_image: np.ndarray,
    delta_i: np.ndarray,
    obstacle_type: str,
    config: dict[str, Any],
    min_cluster_points: int,
    min_abnormal_ratio: float,
    expected_depths_i: np.ndarray | None,
    reference_metric: str,
    positive_is_negative_delta: bool,
    trace: dict[str, Any] | None = None,
) -> int:
    signed_point_count = int(np.count_nonzero(signed_mask))
    min_forward_distance_m = float(config.get("min_forward_distance_m", 0.0))
    dbscan_eps_m = float(config["dbscan_eps_m"])
    dbscan_min_samples = int(config["dbscan_min_samples"])
    sign_trace: dict[str, Any] | None = None
    if trace is not None:
        sign_trace = {
            "sign": obstacle_type,
            "min_cluster_points": int(min_cluster_points),
            "min_abnormal_ratio": float(min_abnormal_ratio),
            "signed_point_count_before_forward_filter": signed_point_count,
            "forward_distance_filter": {
                "min_forward_distance_m": min_forward_distance_m,
                "before_count": signed_point_count,
                "after_count": 0,
            },
            "dbscan": {
                "input_point_count": 0,
                "eps_m": dbscan_eps_m,
                "min_samples": dbscan_min_samples,
                "cluster_count": 0,
                "noise_point_count": 0,
            },
            "clusters": [],
            "reject_reasons": [],
        }
        trace.setdefault("sign_traces", []).append(sign_trace)

    if signed_point_count < min_cluster_points:
        if sign_trace is not None:
            _record_reject(sign_trace, "too_few_signed_points")
        return next_id

    signed_world = roi_world[signed_mask]
    signed_image = roi_image[signed_mask]
    signed_depths = roi_depths[signed_mask]
    signed_delta = delta_i[signed_mask]
    signed_expected_depths = expected_depths_i[signed_mask] if expected_depths_i is not None else None

    forward_mask = np.isfinite(signed_world[:, 2]) & (signed_world[:, 2] > min_forward_distance_m)
    signed_world = signed_world[forward_mask]
    signed_image = signed_image[forward_mask]
    signed_depths = signed_depths[forward_mask]
    signed_delta = signed_delta[forward_mask]
    signed_expected_depths = signed_expected_depths[forward_mask] if signed_expected_depths is not None else None
    if sign_trace is not None:
        sign_trace["forward_distance_filter"]["after_count"] = int(len(signed_world))
    if len(signed_world) < min_cluster_points:
        if sign_trace is not None:
            _record_reject(sign_trace, "too_few_forward_points")
        return next_id

    labels = DBSCAN(
        eps=dbscan_eps_m,
        min_samples=dbscan_min_samples,
    ).fit(signed_world[:, [1, 2]]).labels_
    unique_labels = sorted(set(int(label) for label in labels if int(label) >= 0))
    noise_point_count = int(np.count_nonzero(labels < 0))
    if sign_trace is not None:
        sign_trace["dbscan"] = {
            "input_point_count": int(len(signed_world)),
            "eps_m": dbscan_eps_m,
            "min_samples": dbscan_min_samples,
            "cluster_count": int(len(unique_labels)),
            "noise_point_count": noise_point_count,
        }
    if not unique_labels:
        if sign_trace is not None:
            _record_reject(sign_trace, "dbscan_noise_only")
        return next_id

    min_bbox_area_px = int(config.get("min_bbox_area_px", 0))

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_world = signed_world[cluster_mask]
        cluster_image = signed_image[cluster_mask]
        cluster_depths = signed_depths[cluster_mask]
        cluster_delta = signed_delta[cluster_mask]
        cluster_expected_depths = signed_expected_depths[cluster_mask] if signed_expected_depths is not None else None
        min_valid_forward_points = int(config.get("min_valid_forward_points", min_cluster_points))
        valid_forward = np.isfinite(cluster_world[:, 2]) & (cluster_world[:, 2] > min_forward_distance_m)
        cluster_record: dict[str, Any] | None = None
        if sign_trace is not None:
            cluster_record = {
                "label": int(label),
                "point_count": int(np.count_nonzero(cluster_mask)),
                "valid_forward_point_count": int(np.count_nonzero(valid_forward)),
                "image_bbox": _cluster_bbox(cluster_image) if len(cluster_image) else None,
                "padded_bbox": None,
                "world_y_range_m": _finite_range(cluster_world[:, 1]) if len(cluster_world) else None,
                "world_z_range_m": _finite_range(cluster_world[:, 2]) if len(cluster_world) else None,
                "abnormal_ratio": None,
                "height_m": None,
                "width_m": None,
                "bbox_area_px": None,
                "bbox_decision": None,
                "accepted": False,
                "candidate_generated": False,
                "candidate_id": None,
                "reject_reason": None,
            }
            sign_trace.setdefault("clusters", []).append(cluster_record)
        if int(np.count_nonzero(valid_forward)) < min_valid_forward_points:
            if cluster_record is not None:
                cluster_record["reject_reason"] = "cluster_too_few_forward_points"
            if sign_trace is not None:
                _record_reject(sign_trace, "cluster_too_few_forward_points")
            continue
        cluster_world = cluster_world[valid_forward]
        cluster_image = cluster_image[valid_forward]
        cluster_depths = cluster_depths[valid_forward]
        cluster_delta = cluster_delta[valid_forward]
        cluster_expected_depths = cluster_expected_depths[valid_forward] if cluster_expected_depths is not None else None
        if len(cluster_world) < min_cluster_points:
            if cluster_record is not None:
                cluster_record["reject_reason"] = "cluster_too_few_forward_points"
                cluster_record["valid_forward_point_count"] = int(len(cluster_world))
            if sign_trace is not None:
                _record_reject(sign_trace, "cluster_too_few_forward_points")
            continue

        abnormal_ratio = float(len(cluster_world) / max(1, len(roi_world)))
        effective_min_abnormal_ratio = float(min_abnormal_ratio)
        depth_supported_relaxed = _depth_supported_relaxation_enabled(
            roi=roi,
            config=config,
            reference_metric=reference_metric,
            cluster_expected_depths=cluster_expected_depths,
        )
        if depth_supported_relaxed:
            effective_min_abnormal_ratio = min(
                effective_min_abnormal_ratio,
                float(config.get("min_abnormal_ratio_depth_supported", effective_min_abnormal_ratio)),
            )
        if cluster_record is not None:
            cluster_record["abnormal_ratio"] = abnormal_ratio
            cluster_record["min_abnormal_ratio"] = float(min_abnormal_ratio)
            cluster_record["effective_min_abnormal_ratio"] = effective_min_abnormal_ratio
            cluster_record["depth_supported_relaxed"] = bool(depth_supported_relaxed)
        if abnormal_ratio < effective_min_abnormal_ratio:
            if cluster_record is not None:
                cluster_record["reject_reason"] = "abnormal_ratio_too_low"
            if sign_trace is not None:
                _record_reject(sign_trace, "abnormal_ratio_too_low")
            continue

        if obstacle_type == "positive":
            height_m = float(abs(np.min(cluster_delta))) if positive_is_negative_delta else float(np.max(cluster_delta))
        else:
            height_m = float(np.max(cluster_delta)) if positive_is_negative_delta else float(abs(np.min(cluster_delta)))

        distance_percentile = float(config.get("distance_percentile", 5.0))
        distance_m = float(np.percentile(cluster_world[:, 2], distance_percentile))
        anchor_idx = int(np.argmin(np.abs(cluster_world[:, 2] - distance_m)))
        anchor = cluster_world[anchor_idx].copy()
        anchor[2] = distance_m
        cluster_bbox = _cluster_bbox(cluster_image)
        padded_cluster_bbox = _pad_bbox_within_bounds(cluster_bbox, roi.bbox, config)
        use_roi_bbox = bool(config.get("use_roi_bbox_for_depth_roi", False)) and roi.source == "depth_residual_contour"
        bbox = list(roi.bbox) if use_roi_bbox else padded_cluster_bbox
        bbox_area = int(max(1, bbox[2] * bbox[3]))
        if cluster_record is not None:
            cluster_record["image_bbox"] = list(cluster_bbox)
            cluster_record["padded_bbox"] = list(padded_cluster_bbox)
            cluster_record["bbox_area_px"] = bbox_area
            cluster_record["bbox_decision"] = {
                "cluster_bbox": list(cluster_bbox),
                "padded_cluster_bbox": list(padded_cluster_bbox),
                "roi_bbox": list(roi.bbox),
                "final_bbox": list(bbox),
                "use_roi_bbox_for_depth_roi": bool(use_roi_bbox),
            }
        if bbox_area < min_bbox_area_px:
            if cluster_record is not None:
                cluster_record["reject_reason"] = "bbox_area_too_small"
            if sign_trace is not None:
                _record_reject(sign_trace, "bbox_area_too_small")
            continue
        z_range = [float(np.min(cluster_world[:, 2])), float(np.max(cluster_world[:, 2]))]
        width_m = float(np.max(cluster_world[:, 1]) - np.min(cluster_world[:, 1])) if len(cluster_world) else 0.0
        if cluster_record is not None:
            cluster_record["height_m"] = height_m
            cluster_record["width_m"] = width_m
            cluster_record["world_y_range_m"] = _finite_range(cluster_world[:, 1])
            cluster_record["world_z_range_m"] = z_range
        max_valid_height_m = _optional_float(config, "max_valid_height_m")
        max_valid_width_m = _optional_float(config, "max_valid_width_m")
        if not _valid_dimension(height_m, max_valid_height_m) or not _valid_dimension(width_m, max_valid_width_m):
            if cluster_record is not None:
                cluster_record["reject_reason"] = "invalid_height_or_width"
            if sign_trace is not None:
                _record_reject(sign_trace, "invalid_height_or_width")
            continue
        centroid_2d = cluster_image.mean(axis=0).round(3).tolist()
        candidate_id = f"cand_{next_id:03d}"
        if cluster_record is not None:
            cluster_record["accepted"] = True
            cluster_record["candidate_generated"] = True
            cluster_record["candidate_id"] = candidate_id
            cluster_record["reject_reason"] = "accepted"
        if trace is not None:
            trace["candidate_count"] = int(trace.get("candidate_count", 0)) + 1
            trace.setdefault("candidate_ids", []).append(candidate_id)

        observations.append(
            CandidateObservation(
                candidate_id=candidate_id,
                roi_id=roi.roi_id,
                obstacle_type=obstacle_type,
                bbox=bbox,
                centroid_2d=centroid_2d,
                anchor=anchor.round(6).tolist(),
                height_m=height_m,
                width_m=width_m,
                distance_m=distance_m,
                z_range_m=z_range,
                point_count=int(len(cluster_world)),
                abnormal_count=int(len(cluster_world)),
                support_points=cluster_world.round(6).tolist(),
                metadata={
                    "bbox": bbox,
                    "roi_bbox": list(roi.bbox),
                    "raw_bbox": list((roi.metadata or {}).get("raw_bbox", roi.bbox)),
                    "cluster_bbox": list(cluster_bbox),
                    "padded_cluster_bbox": list(padded_cluster_bbox),
                    "final_bbox": list(bbox),
                    "use_roi_bbox_for_depth_roi": bool(use_roi_bbox),
                    "roi_source": roi.source,
                    "roi_area_px": roi.area_px,
                    "bbox_area_px": bbox_area,
                    "delta_mean_m": float(np.mean(cluster_delta)),
                    "delta_min_m": float(np.min(cluster_delta)),
                    "delta_max_m": float(np.max(cluster_delta)),
                    "depth_mean_m": float(np.mean(cluster_depths)),
                    "expected_depth_mean_m": (
                        float(np.nanmean(cluster_expected_depths))
                        if cluster_expected_depths is not None and np.any(np.isfinite(cluster_expected_depths))
                        else None
                    ),
                    "abnormal_ratio": abnormal_ratio,
                    "cluster_sign": obstacle_type,
                    "reference_metric": reference_metric,
                    "positive_is_negative_delta": positive_is_negative_delta,
                    "min_forward_distance_m": min_forward_distance_m,
                    "distance_percentile": distance_percentile,
                    "valid_forward_point_count": int(len(cluster_world)),
                },
            )
        )
        next_id += 1

    return next_id


def _resolve_depth_roi_threshold(value: Any, fallback: float) -> float:
    if value is None:
        return float(fallback)
    return float(value)


def _rasterize_points_mask(points_2d: np.ndarray, selector: np.ndarray, frame_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((h, w), dtype=bool)
    if len(points_2d) == 0 or not np.any(selector):
        return mask
    points = np.rint(points_2d[selector]).astype(int)
    inside = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    if np.any(inside):
        valid = points[inside]
        mask[valid[:, 1], valid[:, 0]] = True
    return mask


def _clean_depth_roi_mask(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cleaned = mask.astype(np.uint8)
    dilate_kernel_px = int(config.get("dilate_kernel_px", 3))
    if dilate_kernel_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_px, dilate_kernel_px))
        cleaned = cv2.dilate(cleaned, kernel)
    close_kernel_px = int(config.get("close_kernel_px", 5))
    if close_kernel_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_px, close_kernel_px))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned.astype(bool)


def _expand_bbox_xywh(
    bbox: list[int],
    frame_shape: tuple[int, int] | tuple[int, int, int],
    config: dict[str, Any],
) -> list[int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x, y, bw, bh = [int(value) for value in bbox]
    default_expand = int(config.get("bbox_expand_px", 0))
    expand_x = int(config.get("bbox_expand_x_px", default_expand))
    expand_y = int(config.get("bbox_expand_y_px", default_expand))
    expand_top = int(config.get("bbox_expand_top_px", expand_y))
    expand_bottom = int(config.get("bbox_expand_bottom_px", expand_y))
    x0 = max(0, x - expand_x)
    y0 = max(0, y - expand_top)
    x1 = min(w, x + bw + expand_x)
    y1 = min(h, y + bh + expand_bottom)
    return [int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))]


def _depth_roi_components(
    mask: np.ndarray,
    residuals_by_pixel: np.ndarray,
    sign: str,
    config: dict[str, Any],
    start_id: int,
) -> list[RoiCandidate]:
    min_area_px = int(config.get("min_area_px", 40))
    min_component_points = int(config.get("min_component_points", 0))
    min_mask_fill_ratio = float(config.get("min_mask_fill_ratio", 0.0))
    reject_boundary_touching = bool(config.get("reject_boundary_touching", False))
    boundary_margin_px = int(config.get("boundary_margin_px", 1))
    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    h, w = mask.shape
    candidates: list[RoiCandidate] = []

    for label in range(1, num_labels):
        component = label_img == label
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue
        border_touch = {
            "top": bool(component[: max(1, boundary_margin_px), :].any()),
            "bottom": bool(component[max(0, h - max(1, boundary_margin_px)) :, :].any()),
            "left": bool(component[:, : max(1, boundary_margin_px)].any()),
            "right": bool(component[:, max(0, w - max(1, boundary_margin_px)) :].any()),
        }
        touches_border = any(border_touch.values())
        if reject_boundary_touching and touches_border:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        raw_bbox = [x, y, bw, bh]
        expanded_bbox = _expand_bbox_xywh(raw_bbox, mask.shape, config)
        component_residuals = residuals_by_pixel[component]
        component_residuals = component_residuals[np.isfinite(component_residuals)]
        if len(component_residuals) < min_component_points:
            continue
        mask_fill_ratio = float(area_px / max(1, bw * bh))
        if mask_fill_ratio < min_mask_fill_ratio:
            continue
        candidates.append(
            RoiCandidate(
                roi_id=f"depth_roi_{start_id + len(candidates):03d}",
                bbox=expanded_bbox,
                area_px=area_px,
                touch_border=touches_border,
                source="depth_residual_contour",
                metadata={
                    "sign": sign,
                    "mask_fill_ratio": mask_fill_ratio,
                    "raw_bbox": raw_bbox,
                    "point_count": int(len(component_residuals)),
                    "residual_mean_m": float(np.mean(component_residuals)) if len(component_residuals) else None,
                    "residual_min_m": float(np.min(component_residuals)) if len(component_residuals) else None,
                    "residual_max_m": float(np.max(component_residuals)) if len(component_residuals) else None,
                    "border_touch": border_touch,
                },
            )
        )
    return candidates


def build_depth_residual_roi_candidates(
    *,
    candidate_image_points: np.ndarray,
    candidate_depths_m: np.ndarray,
    plane: PlaneEstimate,
    intrinsics: dict[str, Any],
    extrinsics: dict[str, Any],
    frame_shape: tuple[int, int] | tuple[int, int, int],
    analysis_mask: np.ndarray,
    config: dict[str, Any],
) -> list[RoiCandidate]:
    contour_cfg = config.get("depth_contour_roi") or {}
    if not bool(contour_cfg.get("enabled", False)) or plane.status != "ok" or len(candidate_image_points) == 0:
        return []

    residuals, _ = _observed_minus_plane_depth_residual(
        image_points=candidate_image_points,
        observed_depths_m=candidate_depths_m,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        plane=plane,
    )
    valid = np.isfinite(residuals)
    positive_threshold = _resolve_depth_roi_threshold(contour_cfg.get("positive_residual_threshold_m"), float(config["h_pos_m"]))
    negative_threshold = _resolve_depth_roi_threshold(contour_cfg.get("negative_residual_threshold_m"), float(config["h_neg_m"]))
    positive_selector = valid & (residuals < -positive_threshold)
    negative_selector = valid & (residuals > negative_threshold)

    h, w = int(frame_shape[0]), int(frame_shape[1])
    residuals_by_pixel = np.full((h, w), np.nan, dtype=np.float64)
    points = np.rint(candidate_image_points[valid]).astype(int)
    inside = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    if np.any(inside):
        valid_residuals = residuals[valid]
        valid_points = points[inside]
        residuals_by_pixel[valid_points[:, 1], valid_points[:, 0]] = valid_residuals[inside]

    analysis = analysis_mask.astype(bool)
    positive_mask = _clean_depth_roi_mask(
        _rasterize_points_mask(candidate_image_points, positive_selector, frame_shape) & analysis,
        contour_cfg,
    ) & analysis
    negative_mask = _clean_depth_roi_mask(
        _rasterize_points_mask(candidate_image_points, negative_selector, frame_shape) & analysis,
        contour_cfg,
    ) & analysis

    candidates = _depth_roi_components(positive_mask, residuals_by_pixel, "positive", contour_cfg, start_id=0)
    candidates.extend(
        _depth_roi_components(negative_mask, residuals_by_pixel, "negative", contour_cfg, start_id=len(candidates))
    )
    return candidates


def build_candidate_observations(
    roi_candidates: list[RoiCandidate],
    candidate_image_points: np.ndarray,
    candidate_world_points: np.ndarray,
    candidate_depths_m: np.ndarray,
    plane: PlaneEstimate,
    config: dict[str, Any],
    intrinsics: dict[str, Any] | None = None,
    extrinsics: dict[str, Any] | None = None,
) -> list[CandidateObservation]:
    """Build obstacle clusters from ROI points with configurable sign semantics."""

    if plane.status != "ok":
        return []

    normal = np.asarray(plane.normal, dtype=np.float64)
    h_pos_m = float(config["h_pos_m"])
    h_neg_m = float(config["h_neg_m"])
    min_cluster_points_default = int(config["min_cluster_points"])
    min_cluster_points_pos = int(config.get("min_cluster_points_pos", min_cluster_points_default))
    min_cluster_points_neg = int(config.get("min_cluster_points_neg", min_cluster_points_default))
    min_abnormal_ratio_default = float(config.get("min_abnormal_ratio", 0.4))
    min_abnormal_ratio_pos = float(config.get("min_abnormal_ratio_pos", min_abnormal_ratio_default))
    min_abnormal_ratio_neg = float(config.get("min_abnormal_ratio_neg", min_abnormal_ratio_default))

    observations: list[CandidateObservation] = _CandidateObservationList()
    trace_enabled = bool(config.get("trace_roi_clusters", True))
    roi_cluster_traces: list[dict[str, Any]] = []
    next_id = 0

    height_reference = str(config.get("height_reference", "depth_residual_to_plane")).lower()
    use_depth_residual = (
        intrinsics is not None
        and extrinsics is not None
        and height_reference in {"depth_residual_to_plane", "depth_residual", "plane_depth_residual"}
    )

    if use_depth_residual:
        all_delta, expected_depths = _observed_minus_plane_depth_residual(
            image_points=candidate_image_points,
            observed_depths_m=candidate_depths_m,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            plane=plane,
        )
        # For residual = observed - plane_expected:
        # negative -> closer than plane (positive obstacle), positive -> farther (negative obstacle).
        positive_is_negative_delta = True
        reference_metric = "observed_minus_plane_depth_m"
    else:
        all_delta = signed_plane_distance(candidate_world_points, normal, plane.offset)
        expected_depths = None
        positive_is_negative_delta = False
        reference_metric = "signed_plane_distance_m"

    for roi in roi_candidates:
        sliced = _slice_points_in_roi(
            roi=roi,
            image_points=candidate_image_points,
            world_points=candidate_world_points,
            depths_m=candidate_depths_m,
            expected_depths_m=expected_depths,
            residuals_m=all_delta,
        )
        roi_image = sliced["image_points"]
        roi_world = sliced["world_points"]
        roi_depths = sliced["depths_m"]
        roi_expected_depths = sliced.get("expected_depths_m")
        delta_i = sliced.get("residuals_m")
        valid_delta = np.isfinite(delta_i) if delta_i is not None else np.zeros((len(roi_world),), dtype=bool)
        if delta_i is not None and positive_is_negative_delta:
            positive_mask = valid_delta & (delta_i < -h_pos_m)
            negative_mask = valid_delta & (delta_i > h_neg_m)
        elif delta_i is not None:
            positive_mask = valid_delta & (delta_i > h_pos_m)
            negative_mask = valid_delta & (delta_i < -h_neg_m)
        else:
            positive_mask = np.zeros((len(roi_world),), dtype=bool)
            negative_mask = np.zeros((len(roi_world),), dtype=bool)

        trace = (
            _new_roi_cluster_trace(roi, roi_image, roi_world, delta_i, positive_mask, negative_mask, config)
            if trace_enabled
            else None
        )
        if len(roi_world) == 0 or delta_i is None:
            if trace is not None:
                roi_cluster_traces.append(_finalize_roi_cluster_trace(trace))
            continue

        next_id = _append_clustered_candidates(
            observations=observations,
            next_id=next_id,
            roi=roi,
            roi_world=roi_world,
            roi_depths=roi_depths,
            signed_mask=positive_mask,
            roi_image=roi_image,
            delta_i=delta_i,
            obstacle_type="positive",
            config=config,
            min_cluster_points=min_cluster_points_pos,
            min_abnormal_ratio=min_abnormal_ratio_pos,
            expected_depths_i=roi_expected_depths,
            reference_metric=reference_metric,
            positive_is_negative_delta=positive_is_negative_delta,
            trace=trace,
        )
        next_id = _append_clustered_candidates(
            observations=observations,
            next_id=next_id,
            roi=roi,
            roi_world=roi_world,
            roi_depths=roi_depths,
            signed_mask=negative_mask,
            roi_image=roi_image,
            delta_i=delta_i,
            obstacle_type="negative",
            config=config,
            min_cluster_points=min_cluster_points_neg,
            min_abnormal_ratio=min_abnormal_ratio_neg,
            expected_depths_i=roi_expected_depths,
            reference_metric=reference_metric,
            positive_is_negative_delta=positive_is_negative_delta,
            trace=trace,
        )
        if trace is not None:
            roi_cluster_traces.append(_finalize_roi_cluster_trace(trace))

    if trace_enabled:
        setattr(observations, "roi_cluster_traces", roi_cluster_traces)
    return observations


def summarize_cross_frame_matches(
    previous_candidates: list[CandidateObservation],
    current_candidates: list[CandidateObservation],
    config: dict[str, Any],
    forward_displacement_m: float = 0.0,
) -> list[CrossFrameMatch]:
    """寤虹珛鐩搁劙鍏╁箑鐨勫€欓伕寤剁簩鎽樿 M_t銆?"""

    if not previous_candidates or not current_candidates:
        return []

    max_anchor_distance = float(config.get("max_anchor_distance_m", 0.6))
    max_centroid_shift_px = float(config.get("max_centroid_shift_px", 60.0))
    min_score = float(config.get("min_match_score", 0.05))
    matches: list[CrossFrameMatch] = []

    for prev_item in previous_candidates:
        best: CrossFrameMatch | None = None
        prev_anchor = np.asarray(prev_item.anchor, dtype=float)
        prev_anchor = prev_anchor.copy()
        prev_anchor[2] -= float(forward_displacement_m)
        prev_centroid = np.asarray(prev_item.centroid_2d, dtype=float)
        prev_z_range = np.asarray(prev_item.z_range_m, dtype=float) - float(forward_displacement_m)

        for curr_item in current_candidates:
            curr_anchor = np.asarray(curr_item.anchor, dtype=float)
            curr_centroid = np.asarray(curr_item.centroid_2d, dtype=float)
            curr_z_range = np.asarray(curr_item.z_range_m, dtype=float)
            anchor_distance = float(np.linalg.norm(prev_anchor - curr_anchor))
            if anchor_distance > max_anchor_distance:
                continue
            centroid_shift = float(np.linalg.norm(prev_centroid - curr_centroid))
            if centroid_shift > max_centroid_shift_px:
                continue

            iou = bbox_iou(prev_item.bbox, curr_item.bbox)
            z_overlap = max(0.0, min(prev_z_range[1], curr_z_range[1]) - max(prev_z_range[0], curr_z_range[0]))
            z_union = max(prev_z_range[1], curr_z_range[1]) - min(prev_z_range[0], curr_z_range[0])
            z_overlap_ratio = float(z_overlap / z_union) if z_union > 0 else 0.0
            type_bonus = 0.15 if prev_item.obstacle_type == curr_item.obstacle_type else -0.10
            score = (
                0.55 * iou
                + 0.25 * max(0.0, 1.0 - anchor_distance / max_anchor_distance)
                + 0.20 * max(0.0, 1.0 - centroid_shift / max_centroid_shift_px)
                + 0.20 * z_overlap_ratio
                + type_bonus
            )
            if score < min_score:
                continue
            candidate = CrossFrameMatch(
                prev_candidate_id=prev_item.candidate_id,
                curr_candidate_id=curr_item.candidate_id,
                score=float(score),
                metadata={
                    "bbox_iou": float(iou),
                    "anchor_distance_m": anchor_distance,
                    "centroid_shift_px": centroid_shift,
                    "z_overlap_ratio": z_overlap_ratio,
                    "type_consistent": prev_item.obstacle_type == curr_item.obstacle_type,
                },
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best is not None:
            matches.append(best)

    return matches


def classify_candidate_type(
    candidate: CandidateObservation,
    road_mask: np.ndarray,
    boundary_contact_ratio_threshold: float = 0.10,
    elongated_aspect_ratio: float = 2.5,
) -> str:
    """论文 Section 4.3.3：判断候选为路缘/阶差(curb/step)还是坑洞/凹陷(pothole)。

    判断依据（Table 4-2）：
    - 路缘/阶差：边界接触比例高（候选区域与道路边界重叠多）+ 形态细长
    - 坑洞/凹陷：内部负向点比例高 + 面状下沉分布

    实现策略：
    1. 取道路遮罩的边界（膨胀后与原始的差值）
    2. 计算候选 bbox 内有多少比例的像素落在道路边界上
    3. 若比例 > 阈值，或 bbox 细长（aspect ratio > 2.5），则判定为路缘/阶差
    4. 否则判定为坑洞/凹陷

    Args:
        candidate: 候选观测结果
        road_mask: PIDNet 输出的二值道路遮罩（H×W）
        boundary_contact_ratio_threshold: 边界接触比例门坎（默认 0.10）
        elongated_aspect_ratio: 用于判断细长形态的长宽比门坎（默认 2.5）

    Returns:
        "curb/step" | "pothole/pothole" | ""
    """
    x, y, w, h = candidate.bbox
    if w <= 0 or h <= 0:
        return ""

    # 形态判断：bbox 的长宽比（论文 Table 4-2）
    aspect_ratio = float(w) / max(1, float(h))
    if aspect_ratio >= elongated_aspect_ratio:
        return "curb/step"

    # 道路边界检测：用 3-pixel 膨胀的结构元素求差值
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    road_dilated = cv2.dilate(road_mask.astype(np.uint8), kernel)
    road_boundary = road_dilated.astype(bool) ^ road_mask.astype(bool)  # 膨胀后新增的区域 = 边界

    # 取候选区域
    x1 = max(0, x - 1)
    y1 = max(0, y - 1)
    x2 = min(road_mask.shape[1], x + w + 1)
    y2 = min(road_mask.shape[0], y + h + 1)
    roi_boundary = road_boundary[y1:y2, x1:x2]
    roi_area = float(w * h)
    boundary_pixels = float(np.count_nonzero(roi_boundary))
    boundary_ratio = boundary_pixels / max(1.0, roi_area)

    if boundary_ratio >= boundary_contact_ratio_threshold:
        return "curb/step"
    else:
        return "pothole/pothole"

    # 综合判断
    if boundary_contact_ratio >= boundary_contact_ratio_threshold or aspect_ratio >= elongated_aspect_ratio:
        return "curb/step"
    else:
        return "pothole/pothole"


def classify_all_candidates(
    candidates: list[CandidateObservation],
    road_mask: np.ndarray,
    config: dict[str, Any],
) -> list[CandidateObservation]:
    """对所有候选运行类型分类，并更新 candidate_type 字段。"""
    bdry_thresh = float(config.get("boundary_contact_ratio_threshold", 0.15))
    elong_thresh = float(config.get("elongated_aspect_ratio", 2.5))
    for cand in candidates:
        cand.candidate_type = classify_candidate_type(
            candidate=cand,
            road_mask=road_mask,
            boundary_contact_ratio_threshold=bdry_thresh,
            elongated_aspect_ratio=elong_thresh,
        )
    return candidates
