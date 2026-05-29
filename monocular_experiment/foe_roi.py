from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _as_gray(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr.astype(np.uint8)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _clip_point(point: np.ndarray | list[float] | tuple[float, float], frame_shape: tuple[int, int] | tuple[int, int, int]) -> list[int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    return [max(0, min(w - 1, x)), max(0, min(h - 1, y))]


def polygon_bbox(polygon: list[list[int]]) -> list[int]:
    points = np.asarray(polygon, dtype=int).reshape((-1, 2))
    x0 = int(np.min(points[:, 0]))
    y0 = int(np.min(points[:, 1]))
    x1 = int(np.max(points[:, 0]))
    y1 = int(np.max(points[:, 1]))
    return [x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)]


def estimate_foe_from_flow_lines(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    h, w = int(frame_shape[0]), int(frame_shape[1])
    prev = np.asarray(previous_points, dtype=float).reshape((-1, 2))
    curr = np.asarray(current_points, dtype=float).reshape((-1, 2))
    if len(prev) != len(curr):
        raise ValueError("previous_points and current_points must have the same length")

    vectors = curr - prev
    lengths = np.linalg.norm(vectors, axis=1)
    min_length = float(cfg.get("flow_min_length_px", 2.0))
    max_length = float(cfg.get("flow_max_length_px", 120.0))
    valid = np.isfinite(prev).all(axis=1) & np.isfinite(curr).all(axis=1)
    valid &= (lengths >= min_length) & (lengths <= max_length)
    prev = prev[valid]
    vectors = vectors[valid]
    lengths = lengths[valid]

    min_points = int(cfg.get("min_flow_points", 40))
    if len(prev) < min_points:
        return {"status": "fallback", "reason": "insufficient_flow_points", "flow_count": int(len(prev))}

    normals = np.column_stack((-vectors[:, 1], vectors[:, 0]))
    norms = np.linalg.norm(normals, axis=1)
    line_valid = norms > 1e-6
    normals = normals[line_valid] / norms[line_valid, None]
    prev = prev[line_valid]
    lengths = lengths[line_valid]
    if len(normals) < min_points or np.linalg.matrix_rank(normals) < 2:
        return {"status": "fallback", "reason": "degenerate_flow_lines", "flow_count": int(len(normals))}

    rhs = np.sum(normals * prev, axis=1)
    foe, *_ = np.linalg.lstsq(normals, rhs, rcond=None)
    residuals = np.abs(normals @ foe - rhs)
    base_threshold = float(cfg.get("foe_inlier_threshold_px", 12.0))
    adaptive_threshold = max(base_threshold, float(np.percentile(residuals, 65)))
    inliers = residuals <= adaptive_threshold
    if int(np.count_nonzero(inliers)) >= min_points and np.linalg.matrix_rank(normals[inliers]) >= 2:
        foe, *_ = np.linalg.lstsq(normals[inliers], rhs[inliers], rcond=None)
        residuals = np.abs(normals @ foe - rhs)
        inliers = residuals <= adaptive_threshold

    x_min = -float(cfg.get("foe_margin_x_ratio", 0.35)) * w
    x_max = (1.0 + float(cfg.get("foe_margin_x_ratio", 0.35))) * w
    y_min = -float(cfg.get("foe_margin_y_ratio", 0.60)) * h
    y_max = float(cfg.get("foe_max_y_ratio", 0.95)) * h
    if not np.isfinite(foe).all() or not (x_min <= foe[0] <= x_max and y_min <= foe[1] <= y_max):
        return {
            "status": "fallback",
            "reason": "foe_out_of_bounds",
            "flow_count": int(len(normals)),
            "foe": [float(foe[0]), float(foe[1])] if np.isfinite(foe).all() else None,
        }

    return {
        "status": "ok",
        "foe": [float(foe[0]), float(foe[1])],
        "flow_count": int(len(normals)),
        "inlier_count": int(np.count_nonzero(inliers)),
        "mean_residual_px": float(np.mean(residuals[inliers])) if np.any(inliers) else float(np.mean(residuals)),
        "mean_flow_length_px": float(np.mean(lengths)) if lengths.size else 0.0,
    }


def _feature_tracking_mask(
    frame_shape: tuple[int, int] | tuple[int, int, int],
    road_mask: np.ndarray | None,
    config: dict[str, Any],
) -> np.ndarray:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    lower_ratio = float(config.get("flow_lower_half_ratio", config.get("road_edge_search_bottom_ratio", 0.35)))
    mask[int(h * lower_ratio) :, :] = 255
    if road_mask is not None and np.asarray(road_mask).shape == (h, w) and np.any(road_mask):
        dilate_px = max(1, int(config.get("flow_road_mask_dilate_px", 41)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        road_support = cv2.dilate(np.asarray(road_mask, dtype=np.uint8), kernel) > 0
        mask = np.where(road_support, mask, 0).astype(np.uint8)
        if int(np.count_nonzero(mask)) < int(config.get("min_feature_mask_area_px", 500)):
            mask[int(h * lower_ratio) :, :] = 255
    return mask


def track_feature_flow(
    previous_frame_bgr: np.ndarray,
    current_frame_bgr: np.ndarray,
    road_mask: np.ndarray | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    prev_gray = _as_gray(previous_frame_bgr)
    curr_gray = _as_gray(current_frame_bgr)
    feature_mask = _feature_tracking_mask(prev_gray.shape, road_mask, cfg)
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=int(cfg.get("max_flow_corners", 800)),
        qualityLevel=float(cfg.get("flow_track_quality", 0.01)),
        minDistance=float(cfg.get("flow_min_distance_px", 7.0)),
        mask=feature_mask,
        blockSize=int(cfg.get("flow_block_size", 7)),
    )
    if prev_points is None or len(prev_points) == 0:
        return {"status": "fallback", "reason": "no_trackable_features", "previous_points": [], "current_points": []}

    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_points,
        None,
        winSize=(int(cfg.get("lk_window_px", 21)), int(cfg.get("lk_window_px", 21))),
        maxLevel=int(cfg.get("lk_max_level", 3)),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if curr_points is None or status is None:
        return {"status": "fallback", "reason": "optical_flow_failed", "previous_points": [], "current_points": []}

    prev = prev_points.reshape((-1, 2))
    curr = curr_points.reshape((-1, 2))
    valid = status.reshape(-1).astype(bool)
    prev = prev[valid]
    curr = curr[valid]
    if len(prev) == 0:
        return {"status": "fallback", "reason": "no_valid_flow", "previous_points": [], "current_points": []}
    return {"status": "ok", "previous_points": prev, "current_points": curr, "tracked_count": int(len(prev))}


def _fit_x_as_function_of_y(points: list[list[float]], config: dict[str, Any]) -> dict[str, Any] | None:
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    min_points = int(config.get("min_edge_points", 12))
    if len(pts) < min_points:
        return None
    keep = np.ones(len(pts), dtype=bool)
    coeff = np.array([0.0, 0.0], dtype=float)
    for _ in range(3):
        y = pts[keep, 1]
        x = pts[keep, 0]
        if len(x) < min_points or np.ptp(y) < 1.0:
            return None
        coeff = np.polyfit(y, x, 1)
        residual = np.abs((coeff[0] * pts[:, 1] + coeff[1]) - pts[:, 0])
        threshold = max(float(config.get("edge_fit_residual_px", 16.0)), float(np.percentile(residual[keep], 80)))
        keep = residual <= threshold
    return {"slope": float(coeff[0]), "intercept": float(coeff[1]), "support_count": int(np.count_nonzero(keep))}


def _line_x_at_y(line: dict[str, Any], y: float) -> float:
    return float(line["slope"]) * float(y) + float(line["intercept"])


def _scaled_corridor_pair(
    left_point: list[int],
    right_point: list[int],
    frame_shape: tuple[int, int] | tuple[int, int, int],
    width_scale: float,
) -> tuple[list[int], list[int]]:
    scale = max(0.05, min(1.0, float(width_scale)))
    if scale >= 0.999:
        return left_point, right_point
    center_x = 0.5 * (float(left_point[0]) + float(right_point[0]))
    left = [center_x + (float(left_point[0]) - center_x) * scale, float(left_point[1])]
    right = [center_x + (float(right_point[0]) - center_x) * scale, float(right_point[1])]
    return _clip_point(left, frame_shape), _clip_point(right, frame_shape)


def estimate_road_edge_lines(road_mask: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or {}
    mask = np.asarray(road_mask, dtype=bool)
    h, w = mask.shape[:2]
    start_row = max(0, min(h - 1, int(h * float(cfg.get("road_edge_search_bottom_ratio", 0.55)))))
    min_width = int(cfg.get("min_road_row_width_px", max(8, int(w * 0.05))))
    left_points: list[list[float]] = []
    right_points: list[list[float]] = []
    center_x = float(w) * float(cfg.get("road_center_x_ratio", 0.5))
    max_half_width = float(w) * float(cfg.get("road_edge_max_half_width_ratio", 0.42))
    step = max(1, int(cfg.get("edge_row_stride", 2)))
    for y in range(start_row, h, step):
        xs = np.flatnonzero(mask[y])
        if len(xs) < min_width:
            continue
        left_candidates = xs[xs <= center_x]
        right_candidates = xs[xs >= center_x]
        left_x = float(left_candidates[0]) if len(left_candidates) else float(xs[0])
        right_x = float(right_candidates[-1]) if len(right_candidates) else float(xs[-1])
        left_x = max(left_x, center_x - max_half_width)
        right_x = min(right_x, center_x + max_half_width)
        if right_x - left_x < min_width:
            continue
        left_points.append([left_x, float(y)])
        right_points.append([right_x, float(y)])

    left_line = _fit_x_as_function_of_y(left_points, cfg)
    right_line = _fit_x_as_function_of_y(right_points, cfg)
    if left_line is None or right_line is None:
        return {"status": "fallback", "reason": "insufficient_road_edge_points"}

    bottom_y = float(h - 1)
    top_y = float(start_row)
    left_bottom_x = _line_x_at_y(left_line, bottom_y)
    right_bottom_x = _line_x_at_y(right_line, bottom_y)
    if right_bottom_x - left_bottom_x < float(cfg.get("min_bottom_width_ratio", 0.15)) * w:
        return {"status": "fallback", "reason": "road_edges_too_narrow"}

    left_line["endpoints"] = [_clip_point([_line_x_at_y(left_line, top_y), top_y], mask.shape), _clip_point([left_bottom_x, bottom_y], mask.shape)]
    right_line["endpoints"] = [_clip_point([_line_x_at_y(right_line, top_y), top_y], mask.shape), _clip_point([right_bottom_x, bottom_y], mask.shape)]
    return {"status": "ok", "source": "road_mask_boundary", "left": left_line, "right": right_line, "start_row": int(start_row)}


def build_road_corridor_from_foe_and_edges(
    foe_xy: list[float] | tuple[float, float],
    road_edges: dict[str, Any],
    frame_shape: tuple[int, int] | tuple[int, int, int],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    h, w = int(frame_shape[0]), int(frame_shape[1])
    left = road_edges.get("left") or {}
    right = road_edges.get("right") or {}
    if not left or not right:
        return {"status": "fallback", "reason": "missing_road_edges"}

    bottom_y = float(h - 1)
    top_y = max(0.0, min(bottom_y - 1.0, float(foe_xy[1])))
    min_top_y = float(h) * float(cfg.get("corridor_min_top_y_ratio", 0.30))
    top_y = max(top_y, min_top_y)
    top_y = min(top_y, float(h) * float(cfg.get("corridor_max_top_y_ratio", 0.65)))
    top_y = min(top_y, bottom_y - max(1.0, float(h) * float(cfg.get("corridor_min_height_ratio", 0.0))))
    top_y = max(0.0, top_y)

    left_top = _clip_point([_line_x_at_y(left, top_y), top_y], frame_shape)
    right_top = _clip_point([_line_x_at_y(right, top_y), top_y], frame_shape)
    left_bottom = _clip_point([_line_x_at_y(left, bottom_y), bottom_y], frame_shape)
    right_bottom = _clip_point([_line_x_at_y(right, bottom_y), bottom_y], frame_shape)
    left_top, right_top = _scaled_corridor_pair(
        left_top,
        right_top,
        frame_shape,
        float(cfg.get("corridor_top_width_scale", cfg.get("corridor_width_scale", 1.0))),
    )
    left_bottom, right_bottom = _scaled_corridor_pair(
        left_bottom,
        right_bottom,
        frame_shape,
        float(cfg.get("corridor_bottom_width_scale", cfg.get("corridor_width_scale", 1.0))),
    )
    if right_top[0] <= left_top[0] or right_bottom[0] <= left_bottom[0]:
        return {"status": "fallback", "reason": "road_edges_cross"}

    polygon = [left_top, right_top, right_bottom, left_bottom]
    area = float(abs(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))
    area_ratio = area / float(max(1, h * w))
    if area_ratio < float(cfg.get("min_corridor_area_ratio", cfg.get("min_triangle_area_ratio", 0.08))):
        return {"status": "fallback", "reason": "corridor_area_too_small", "area_ratio": area_ratio}

    return {
        "status": "ok",
        "polygon": [[int(x), int(y)] for x, y in polygon],
        "bbox": polygon_bbox(polygon),
        "area_ratio": area_ratio,
        "left_top": left_top,
        "right_top": right_top,
        "left_bottom": left_bottom,
        "right_bottom": right_bottom,
        "foe": _clip_point(foe_xy, frame_shape),
    }


def build_triangle_from_foe_and_edges(
    foe_xy: list[float] | tuple[float, float],
    road_edges: dict[str, Any],
    frame_shape: tuple[int, int] | tuple[int, int, int],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    if str(cfg.get("roi_shape", "road_corridor")).lower() in {"road_corridor", "corridor", "rectangle", "quadrilateral"}:
        return build_road_corridor_from_foe_and_edges(foe_xy, road_edges, frame_shape, cfg)

    h, w = int(frame_shape[0]), int(frame_shape[1])
    left = road_edges.get("left") or {}
    right = road_edges.get("right") or {}
    if not left or not right:
        return {"status": "fallback", "reason": "missing_road_edges"}

    bottom_y = h - 1
    left_bottom = _clip_point([_line_x_at_y(left, bottom_y), bottom_y], frame_shape)
    right_bottom = _clip_point([_line_x_at_y(right, bottom_y), bottom_y], frame_shape)
    if right_bottom[0] <= left_bottom[0]:
        return {"status": "fallback", "reason": "road_edges_cross_at_bottom"}

    apex = _clip_point(foe_xy, frame_shape)
    max_apex_y = int(h * float(cfg.get("triangle_max_apex_y_ratio", 0.95)))
    if apex[1] >= max_apex_y:
        return {"status": "fallback", "reason": "foe_too_low"}

    polygon = [apex, right_bottom, left_bottom]
    area = float(abs(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))
    area_ratio = area / float(max(1, h * w))
    if area_ratio < float(cfg.get("min_triangle_area_ratio", 0.08)):
        return {"status": "fallback", "reason": "triangle_area_too_small", "area_ratio": area_ratio}

    return {
        "status": "ok",
        "polygon": [[int(x), int(y)] for x, y in polygon],
        "bbox": polygon_bbox(polygon),
        "area_ratio": area_ratio,
        "left_bottom": left_bottom,
        "right_bottom": right_bottom,
        "foe": apex,
    }


def build_foe_road_triangle_roi(
    previous_frame_bgr: np.ndarray | None,
    current_frame_bgr: np.ndarray,
    road_mask: np.ndarray,
    config: dict[str, Any] | None = None,
    previous_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    if previous_frame_bgr is None:
        return {"status": "fallback", "reason": "no_previous_frame", "state": previous_state or {}}

    flow = track_feature_flow(previous_frame_bgr, current_frame_bgr, road_mask, cfg)
    if flow.get("status") != "ok":
        return {"status": "fallback", "reason": flow.get("reason", "flow_failed"), "flow": flow, "state": previous_state or {}}

    foe = estimate_foe_from_flow_lines(flow["previous_points"], flow["current_points"], current_frame_bgr.shape, cfg)
    if foe.get("status") != "ok":
        return {"status": "fallback", "reason": foe.get("reason", "foe_failed"), "flow": {"tracked_count": flow.get("tracked_count", 0)}, "foe": foe, "state": previous_state or {}}

    raw_foe = np.asarray(foe["foe"], dtype=float)
    smoothed_foe = raw_foe
    previous_foe = (previous_state or {}).get("foe")
    if previous_foe is not None:
        prev = np.asarray(previous_foe, dtype=float)
        if prev.shape == (2,) and np.isfinite(prev).all():
            alpha = float(cfg.get("foe_smoothing_alpha", 0.6))
            alpha = max(0.0, min(1.0, alpha))
            smoothed_foe = alpha * prev + (1.0 - alpha) * raw_foe
    max_apex_y = float(current_frame_bgr.shape[0]) * float(
        cfg.get("foe_max_apex_y_ratio", cfg.get("road_edge_search_bottom_ratio", 0.58))
    )
    if smoothed_foe[1] > max_apex_y:
        smoothed_foe = smoothed_foe.copy()
        smoothed_foe[1] = max_apex_y

    edges = estimate_road_edge_lines(road_mask, cfg)
    if edges.get("status") != "ok":
        return {"status": "fallback", "reason": edges.get("reason", "road_edges_failed"), "flow": {"tracked_count": flow.get("tracked_count", 0)}, "foe": foe, "road_edges": edges, "state": {"foe": smoothed_foe.tolist()}}

    triangle = build_triangle_from_foe_and_edges(smoothed_foe.tolist(), edges, current_frame_bgr.shape, cfg)
    if triangle.get("status") != "ok":
        return {"status": "fallback", "reason": triangle.get("reason", "triangle_failed"), "flow": {"tracked_count": flow.get("tracked_count", 0)}, "foe": {**foe, "smoothed": smoothed_foe.tolist()}, "road_edges": edges, "triangle": triangle, "state": {"foe": smoothed_foe.tolist()}}

    return {
        "status": "ok",
        "mode": "foe_road_triangle",
        "source": "foe_road_triangle",
        "polygon": triangle["polygon"],
        "bbox": triangle["bbox"],
        "foe": {**foe, "raw": raw_foe.tolist(), "smoothed": smoothed_foe.tolist()},
        "road_edges": {
            "left": {"endpoints": edges["left"]["endpoints"], "support_count": int(edges["left"].get("support_count", 0))},
            "right": {"endpoints": edges["right"]["endpoints"], "support_count": int(edges["right"].get("support_count", 0))},
        },
        "triangle": {
            "area_ratio": float(triangle["area_ratio"]),
            "left_top": triangle.get("left_top"),
            "right_top": triangle.get("right_top"),
            "left_bottom": triangle.get("left_bottom"),
            "right_bottom": triangle.get("right_bottom"),
            "foe": triangle.get("foe"),
            "roi_shape": str(cfg.get("roi_shape", "road_corridor")),
        },
        "state": {"foe": smoothed_foe.tolist()},
    }
