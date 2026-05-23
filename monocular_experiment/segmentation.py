from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import RoiCandidate
from .pidnet_official import infer_official_pidnet, infer_onnx_pidnet, infer_tensorrt_pidnet


@dataclass
class FrontendResult:
    road_mask: np.ndarray
    road_probability: np.ndarray
    analysis_mask: np.ndarray
    roi_candidates: list[RoiCandidate]
    backend: str
    processing_roi: dict[str, Any] | None = None
    obstacle_analysis_mask: np.ndarray | None = None
    backend_attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def roi_mask(self) -> np.ndarray:
        """維持舊欄位名稱相容，實際上回傳 analysis mask。"""

        return self.analysis_mask

    @property
    def obstacle_mask(self) -> np.ndarray:
        return self.obstacle_analysis_mask if self.obstacle_analysis_mask is not None else self.analysis_mask

    def road_mask_stats(self) -> dict[str, Any]:
        area = int(np.count_nonzero(self.road_mask))
        analysis_area = int(np.count_nonzero(self.analysis_mask))
        return {
            "backend": self.backend,
            "backend_attempts": list(self.backend_attempts),
            "road_area_px": area,
            "road_ratio": float(area / self.road_mask.size) if self.road_mask.size else 0.0,
            "analysis_area_px": analysis_area,
            "analysis_ratio": float(analysis_area / self.analysis_mask.size) if self.analysis_mask.size else 0.0,
            "roi_count": len(self.roi_candidates),
            "road_probability_mean": float(np.mean(self.road_probability)) if self.road_probability.size else 0.0,
        }


def _resolve_local_path(path_value: str | Path, config: dict[str, Any]) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    config_dir = config.get("_config_dir")
    if config_dir:
        return (Path(config_dir).parent / candidate).resolve()
    return candidate.resolve()


def _softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    shift = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shift)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _remove_small_regions(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 1:
        return mask.astype(bool)
    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            keep[label_img == label] = True
    return keep


def _border_touch_flags(mask: np.ndarray, margin_px: int) -> dict[str, bool]:
    margin = max(0, int(margin_px))
    if margin == 0:
        return {
            "top": bool(mask[0, :].any()),
            "bottom": bool(mask[-1, :].any()),
            "left": bool(mask[:, 0].any()),
            "right": bool(mask[:, -1].any()),
        }
    top = mask[:margin, :]
    bottom = mask[-margin:, :]
    left = mask[:, :margin]
    right = mask[:, -margin:]
    return {
        "top": bool(top.any()),
        "bottom": bool(bottom.any()),
        "left": bool(left.any()),
        "right": bool(right.any()),
    }


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = (~mask).astype(np.uint8) * 255
    flood_canvas = flood.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood_canvas, flood_mask, seedPoint=(0, 0), newVal=128)
    holes = flood_canvas == 255
    return mask | holes


def _build_negative_space_mask(road_mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    close_kernel_px = max(1, int(config.get("hole_close_kernel_px", 11)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_px, close_kernel_px))
    road_closed = cv2.morphologyEx(road_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    road_filled = _fill_holes(road_closed)
    return road_filled & (~road_mask)


def _build_nonroad_near_road_mask(road_mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    h, w = road_mask.shape
    lower_half_ratio = float(config.get("nonroad_lower_half_ratio", 0.35))
    start_row = min(h, max(0, int(h * lower_half_ratio)))

    lower_prior = np.zeros((h, w), dtype=bool)
    lower_prior[start_row:, :] = True

    dilation_px = max(1, int(config.get("nonroad_road_dilation_px", 31)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
    dilated_road_mask = cv2.dilate(road_mask.astype(np.uint8), kernel).astype(bool)

    non_road = ~road_mask
    near_road_nonroad = non_road & lower_prior & dilated_road_mask
    return _remove_small_regions(near_road_nonroad, int(config.get("nonroad_min_area_px", 80)))


def _component_sources(component: np.ndarray, source_masks: dict[str, np.ndarray] | None, default_source: str) -> list[str]:
    if not source_masks:
        return [default_source]
    sources = [name for name, mask in source_masks.items() if np.any(component & mask)]
    return sources or [default_source]


def _bbox_intersects(a: list[int], b: list[int]) -> bool:
    ax, ay, aw, ah = [int(value) for value in a]
    bx, by, bw, bh = [int(value) for value in b]
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def bbox_to_mask(frame_shape: tuple[int, int] | tuple[int, int, int], bbox: list[int]) -> np.ndarray:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x, y, bw, bh = [int(value) for value in bbox]
    x0 = max(0, min(w, x))
    y0 = max(0, min(h, y))
    x1 = max(x0, min(w, x + bw))
    y1 = max(y0, min(h, y + bh))
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def polygon_to_mask(frame_shape: tuple[int, int] | tuple[int, int, int], polygon_xy: list[list[int]]) -> np.ndarray:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    if not polygon_xy:
        return mask.astype(bool)
    points = np.asarray(polygon_xy, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _resolve_processing_bbox(
    frame_shape: tuple[int, int] | tuple[int, int, int],
    rect_cfg: dict[str, Any],
) -> tuple[list[int], str] | None:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    bbox = rect_cfg.get("bbox_xywh")
    source = "manual"
    if bbox is None:
        normalized = rect_cfg.get("normalized_bbox_xywh")
        if normalized is None:
            return None
        nx, ny, nw, nh = [float(value) for value in normalized]
        bbox = [round(nx * w), round(ny * h), round(nw * w), round(nh * h)]
        source = "manual_normalized"

    x, y, bw, bh = [int(round(float(value))) for value in bbox]
    x0 = max(0, min(w, x))
    y0 = max(0, min(h, y))
    x1 = max(x0, min(w, x + max(0, bw)))
    y1 = max(y0, min(h, y + max(0, bh)))
    clipped = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
    if clipped[2] <= 0 or clipped[3] <= 0:
        return None
    return clipped, source


def resolve_processing_roi(frame_shape: tuple[int, int] | tuple[int, int, int], config: dict[str, Any]) -> dict[str, Any] | None:
    rect_cfg = config.get("processing_rect") or {}
    if not rect_cfg or not bool(rect_cfg.get("enabled", False)):
        return None

    resolved_bbox = _resolve_processing_bbox(frame_shape, rect_cfg)
    if resolved_bbox is None:
        return None
    clipped, source = resolved_bbox
    h, w = int(frame_shape[0]), int(frame_shape[1])
    mode = str(rect_cfg.get("mode", "manual")).lower()
    obstacle_mask_source = str(rect_cfg.get("obstacle_analysis_mask", "legacy_analysis_mask"))
    if mode == "trapezoid" and "obstacle_analysis_mask" not in rect_cfg:
        obstacle_mask_source = "processing_roi"

    processing_roi: dict[str, Any] = {
        "enabled": True,
        "mode": mode,
        "source": source,
        "bbox": clipped,
        "restrict_analysis_mask": bool(rect_cfg.get("restrict_analysis_mask", True)),
        "restrict_road_mask": bool(rect_cfg.get("restrict_road_mask", False)),
        "draw_overlay": bool(rect_cfg.get("draw_overlay", True)),
        "obstacle_analysis_mask": obstacle_mask_source,
    }

    if mode == "trapezoid":
        x, y, bw, _ = clipped
        top_right_x = min(w - 1, x + max(0, bw))
        bottom_full_width = bool(rect_cfg.get("trapezoid_bottom_full_width", True))
        if bottom_full_width:
            bottom_left = [0, h - 1]
            bottom_right = [w - 1, h - 1]
        else:
            _, _, _, bh = clipped
            bottom_y = min(h - 1, y + max(0, bh))
            bottom_left = [x, bottom_y]
            bottom_right = [top_right_x, bottom_y]
        polygon = [[x, y], [top_right_x, y], bottom_right, bottom_left]
        processing_roi["polygon"] = [[int(px), int(py)] for px, py in polygon]

    return processing_roi


def resolve_processing_rect(frame_shape: tuple[int, int] | tuple[int, int, int], config: dict[str, Any]) -> dict[str, Any] | None:
    return resolve_processing_roi(frame_shape, config)


def _apply_processing_rect(result: FrontendResult, frame_shape: tuple[int, int, int], config: dict[str, Any]) -> FrontendResult:
    processing_roi = resolve_processing_roi(frame_shape, config)
    if processing_roi is None:
        return result

    roi_mask = (
        polygon_to_mask(frame_shape, processing_roi["polygon"])
        if processing_roi.get("polygon")
        else bbox_to_mask(frame_shape, processing_roi["bbox"])
    )
    if processing_roi["restrict_analysis_mask"]:
        result.analysis_mask = result.analysis_mask.astype(bool) & roi_mask
        result.roi_candidates = [
            candidate for candidate in result.roi_candidates if _bbox_intersects(candidate.bbox, processing_roi["bbox"])
        ]
    if processing_roi["restrict_road_mask"]:
        result.road_mask = result.road_mask.astype(bool) & roi_mask
        result.road_probability = np.where(roi_mask, result.road_probability, 0.0).astype(np.float32)
    if str(processing_roi.get("obstacle_analysis_mask", "legacy_analysis_mask")) == "processing_roi":
        result.obstacle_analysis_mask = roi_mask
    result.processing_roi = processing_roi
    return result


def _mask_to_roi_candidates(
    mask: np.ndarray,
    source: str,
    config: dict[str, Any],
    source_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[RoiCandidate]]:
    min_area_px = int(config.get("roi_min_area_px", 80))
    boundary_margin_px = int(config.get("boundary_margin_px", 2))
    reject_boundary_touching = bool(config.get("reject_boundary_touching", True))
    allow_bottom_boundary_touching = bool(config.get("allow_bottom_boundary_touching", False))

    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    roi_mask = np.zeros_like(mask, dtype=bool)
    candidates: list[RoiCandidate] = []
    next_id = 0

    for label in range(1, num_labels):
        component = label_img == label
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue
        border_touch = _border_touch_flags(component, boundary_margin_px)
        touches_border = any(border_touch.values())
        touches_forbidden_border = (
            border_touch["top"]
            or border_touch["left"]
            or border_touch["right"]
            or (border_touch["bottom"] and not allow_bottom_boundary_touching)
        )
        if reject_boundary_touching and touches_forbidden_border:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        roi_mask |= component
        candidates.append(
            RoiCandidate(
                roi_id=f"roi_{next_id:03d}",
                bbox=[x, y, w, h],
                area_px=area_px,
                touch_border=touches_border,
                source=source,
                metadata={
                    "mask_fill_ratio": float(area_px / max(1, w * h)),
                    "sources": _component_sources(component, source_masks, source),
                    "border_touch": border_touch,
                    "allow_bottom_boundary_touching": allow_bottom_boundary_touching,
                },
            )
        )
        next_id += 1
    return roi_mask, candidates


def _build_roi_candidates(road_mask: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, list[RoiCandidate]]:
    roi_mode = str(config.get("roi_mode", config.get("mode", "negative_space"))).lower()
    negative_space_mask = _build_negative_space_mask(road_mask, config)
    near_road_nonroad_mask = _build_nonroad_near_road_mask(road_mask, config)

    if roi_mode == "negative_space":
        return _mask_to_roi_candidates(
            mask=negative_space_mask,
            source="negative_space",
            config=config,
            source_masks={"negative_space": negative_space_mask},
        )
    if roi_mode == "non_road_near_road":
        return _mask_to_roi_candidates(
            mask=near_road_nonroad_mask,
            source="non_road_near_road",
            config=config,
            source_masks={"non_road_near_road": near_road_nonroad_mask},
        )
    if roi_mode == "hybrid":
        hybrid_mask = negative_space_mask | near_road_nonroad_mask
        return _mask_to_roi_candidates(
            mask=hybrid_mask,
            source="hybrid",
            config=config,
            source_masks={
                "negative_space": negative_space_mask,
                "non_road_near_road": near_road_nonroad_mask,
            },
        )
    raise ValueError(f"Unsupported roi mode: {roi_mode}")


def _clean_road_mask(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    post_kernel_px = max(1, int(config.get("post_kernel_size", 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (post_kernel_px, post_kernel_px))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    cleaned = _remove_small_regions(closed.astype(bool), int(config.get("min_road_area_px", 300)))
    return cleaned


def _official_pidnet_backend(frame: np.ndarray, config: dict[str, Any], strict: bool) -> FrontendResult | None:
    try:
        inference = infer_official_pidnet(frame, config)
    except Exception:
        if strict:
            raise
        return None
    road_prob = inference["road_probability"]
    road_mask = _clean_road_mask(road_prob >= float(config.get("tau_road", 0.45)), config)
    roi_mask, roi_candidates = _build_roi_candidates(road_mask, config)
    return FrontendResult(
        road_mask=road_mask,
        road_probability=road_prob.astype(np.float32),
        analysis_mask=roi_mask,
        roi_candidates=roi_candidates,
        backend=str(inference["backend"]),
    )


def _tensorrt_backend(frame: np.ndarray, config: dict[str, Any], strict: bool) -> FrontendResult | None:
    try:
        inference = infer_tensorrt_pidnet(frame, config)
    except Exception:
        if strict:
            raise
        return None
    road_prob = inference["road_probability"]
    road_mask = _clean_road_mask(road_prob >= float(config.get("tau_road", 0.45)), config)
    roi_mask, roi_candidates = _build_roi_candidates(road_mask, config)
    return FrontendResult(
        road_mask=road_mask,
        road_probability=road_prob.astype(np.float32),
        analysis_mask=roi_mask,
        roi_candidates=roi_candidates,
        backend=str(inference["backend"]),
    )


def _onnx_backend(frame: np.ndarray, config: dict[str, Any], strict: bool) -> FrontendResult | None:
    try:
        inference = infer_onnx_pidnet(frame, config)
    except Exception:
        if strict:
            raise
        return None
    road_prob = inference["road_probability"]
    road_mask = _clean_road_mask(road_prob >= float(config.get("tau_road", 0.45)), config)
    roi_mask, roi_candidates = _build_roi_candidates(road_mask, config)
    return FrontendResult(
        road_mask=road_mask,
        road_probability=road_prob.astype(np.float32),
        analysis_mask=roi_mask,
        roi_candidates=roi_candidates,
        backend=str(inference["backend"]),
    )


def _fallback_backend(frame: np.ndarray, config: dict[str, Any]) -> FrontendResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(blur, cv2.CV_32F)
    texture = np.abs(lap)

    h, w = gray.shape
    lower_prior = np.zeros((h, w), dtype=bool)
    start_row = int(h * float(config.get("fallback_lower_half_ratio", 0.45)))
    lower_prior[start_row:, :] = True

    road_min_intensity = int(config.get("fallback_road_min_intensity", 70))
    texture_threshold = float(config.get("fallback_texture_threshold", 9.0))
    intensity_score = np.clip((blur.astype(np.float32) - road_min_intensity) / max(1.0, 255.0 - road_min_intensity), 0.0, 1.0)
    texture_score = np.clip(1.0 - (texture / max(1.0, texture_threshold * 3.0)), 0.0, 1.0)
    road_prob = (0.6 * intensity_score + 0.4 * texture_score) * lower_prior.astype(np.float32)

    road_mask = lower_prior & (blur >= road_min_intensity) & (texture <= texture_threshold)
    road_mask = _clean_road_mask(road_mask, config)
    roi_mask, roi_candidates = _build_roi_candidates(road_mask, config)
    return FrontendResult(
        road_mask=road_mask,
        road_probability=road_prob.astype(np.float32),
        analysis_mask=roi_mask,
        roi_candidates=roi_candidates,
        backend="fallback",
    )


def _normalize_backend_name(name: str) -> str:
    normalized = str(name).lower()
    if normalized in {"pidnet", "pidnet_official"}:
        return "official_pidnet"
    return normalized


def _call_backend(frame: np.ndarray, config: dict[str, Any], backend: str, strict: bool) -> FrontendResult | None:
    if backend == "tensorrt":
        return _tensorrt_backend(frame, config, strict=strict)
    if backend == "onnx":
        return _onnx_backend(frame, config, strict=strict)
    if backend == "official_pidnet":
        return _official_pidnet_backend(frame, config, strict=strict)
    if backend == "fallback":
        return _fallback_backend(frame, config)
    raise ValueError(f"Unsupported ROI backend: {backend}")


def _backend_order(backend: str, config: dict[str, Any]) -> list[str]:
    if backend == "auto":
        configured = config.get("auto_backend_order")
        if configured:
            return [_normalize_backend_name(str(item)) for item in configured]
        return ["tensorrt", "onnx", "official_pidnet", "fallback"]
    return [_normalize_backend_name(backend)]


def _augment_with_fallback_rois(result: FrontendResult, frame: np.ndarray, config: dict[str, Any]) -> FrontendResult:
    if not bool(config.get("augment_with_fallback_rois", False)) or result.backend == "fallback":
        return result
    fallback = _fallback_backend(frame, config)
    result.analysis_mask = result.analysis_mask.astype(bool) | fallback.analysis_mask.astype(bool)
    start_index = len(result.roi_candidates)
    for offset, candidate in enumerate(fallback.roi_candidates):
        result.roi_candidates.append(
            RoiCandidate(
                roi_id=f"fallback_{start_index + offset:03d}",
                bbox=list(candidate.bbox),
                area_px=int(candidate.area_px),
                touch_border=bool(candidate.touch_border),
                source=f"fallback_{candidate.source}",
                metadata={**candidate.metadata, "auxiliary_backend": "fallback"},
            )
        )
    result.backend_attempts.append(
        {
            "backend": "fallback_roi_augmentation",
            "status": "ok",
            "roi_count": len(fallback.roi_candidates),
        }
    )
    return result


def segment_frame(frame: np.ndarray, config: dict[str, Any]) -> FrontendResult:
    backend = _normalize_backend_name(str(config.get("backend", "auto")))
    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    requested_order = _backend_order(backend, config)

    for candidate_backend in requested_order:
        strict = backend != "auto" and candidate_backend != "fallback"
        attempt: dict[str, Any] = {"backend": candidate_backend, "status": "failed"}
        try:
            result = _call_backend(frame, config, candidate_backend, strict=strict)
        except Exception as exc:
            last_error = exc
            attempt["error"] = str(exc)
            attempts.append(attempt)
            if strict:
                raise
            continue
        if result is None:
            attempt["error"] = "backend_unavailable"
            attempts.append(attempt)
            continue
        attempt["status"] = "ok"
        attempt["selected_backend"] = result.backend
        attempts.append(attempt)
        result.backend_attempts = attempts
        if result.backend == "fallback" and bool(config.get("fail_on_fallback", False)):
            raise RuntimeError("ROI backend fell back to fallback while fail_on_fallback is enabled.")
        result = _augment_with_fallback_rois(result, frame, config)
        return _apply_processing_rect(result, frame.shape, config)

    if backend == "auto":
        raise RuntimeError(f"No ROI backend could be initialized: {attempts}")
    if last_error is not None:
        raise RuntimeError(f"Configured ROI backend {backend} could not be initialized: {last_error}") from last_error
    raise ValueError(f"Unsupported ROI backend: {backend}")
