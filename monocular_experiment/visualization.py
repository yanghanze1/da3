from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Configure CJK font for matplotlib (thesis requires Chinese labels)
_fonts = [f.name for f in fm.fontManager.ttflist]
_CJK_FONT = next((f for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"] if f in _fonts), "DejaVu Sans")
plt.rcParams["font.family"] = _CJK_FONT
plt.rcParams["axes.unicode_minus"] = False

from .geometry import RelativeDepthScaleResult
from .models import CandidateObservation, PlaneEstimate, RiskAssessment, RoiCandidate
from .segmentation import FrontendResult


def _overlay_mask(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    if mask.size == 0:
        return canvas
    overlay = canvas.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0)


def save_overlay(
    frame_bgr: np.ndarray,
    frame_id: str,
    frontend: FrontendResult,
    road_points: np.ndarray,
    plane: dict,
    tracked_objects: list[dict],
    output_path: str | Path,
    candidate_rois: list[RoiCandidate] | None = None,
) -> None:
    """Render the road mask, plane state, and final obstacle boxes."""

    canvas = frame_bgr.copy()
    canvas = _overlay_mask(canvas, frontend.road_mask, (60, 140, 60), 0.30)

    for point in road_points.astype(int):
        cv2.circle(canvas, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)

    cv2.putText(
        canvas,
        f"frame={frame_id} plane={plane.get('status', 'unknown')}/{plane.get('source', 'none')}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    processing_roi = frontend.processing_roi or {}
    if processing_roi.get("draw_overlay", True) and processing_roi.get("polygon"):
        points = np.asarray(processing_roi["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        label_x, label_y = points.reshape((-1, 2))[0]
        cv2.putText(canvas, "processing_roi", (int(label_x), max(12, int(label_y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    elif processing_roi.get("draw_overlay", True) and processing_roi.get("bbox"):
        x, y, w, h = [int(value) for value in processing_roi["bbox"]]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(canvas, "processing_roi", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    for roi in candidate_rois or []:
        if roi.source != "depth_residual_contour":
            continue
        x, y, w, h = [int(value) for value in roi.bbox]
        color = (0, 165, 255) if roi.metadata.get("sign") == "positive" else (255, 255, 0)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
        cv2.putText(canvas, "depth_roi", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    for idx, obj in enumerate(tracked_objects):
        risk = obj.get("risk") or {}
        x, y, w, h = obj.get("metadata", {}).get("bbox", obj.get("metadata", {}).get("roi_bbox", [0, 0, 0, 0]))
        if w > 0 and h > 0:
            # BGR color mapping aligned with depth semantics:
            # positive obstacle (closer / "blue area") -> blue box,
            # negative obstacle (farther / "yellow area") -> yellow box.
            if obj.get("obstacle_type") == "positive":
                color = (255, 0, 0)
            elif obj.get("obstacle_type") == "negative":
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

        label = (
            f"{obj.get('object_id', f'obj_{idx}')}"
            f" {obj.get('state', 'na')}"
            f" {risk.get('risk_weight', 'omega0')}/{risk.get('decision', 'safe')}"
        )
        metadata = obj.get("metadata", {})
        matched_link = metadata.get("matched_cluster_link_count", 0)
        match_score = metadata.get("match_score")
        match_score_text = f"{float(match_score):.2f}" if isinstance(match_score, (int, float)) else "na"
        temporal = metadata.get("temporal_measurement") or {}
        temporal_quality = temporal.get("quality")
        temporal_text = f" TQ={float(temporal_quality):.2f}" if isinstance(temporal_quality, (int, float)) else ""
        anchor = obj.get("anchor", [0.0, 0.0, 0.0])
        y_text = 42 + idx * 32

        cv2.putText(canvas, label, (8, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"link={int(matched_link)} score={match_score_text}",
            (8, y_text + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Z={anchor[2]:.2f} H={obj.get('height_m', 0.0):.2f} W={obj.get('width_m', 0.0):.2f}{temporal_text}",
            (8, y_text + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _draw_xywh(canvas: np.ndarray, bbox: Any, color: tuple[int, int, int], thickness: int = 1) -> None:
    if not bbox:
        return
    x, y, w, h = [int(value) for value in list(bbox)[:4]]
    if w <= 0 or h <= 0:
        return
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)


def _put_label(canvas: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    cv2.putText(canvas, text[:80], (int(x), max(12, int(y))), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def save_roi_to_cluster_trace_overlay(
    frame_bgr: np.ndarray,
    frame_id: str,
    gt_bbox: list[int],
    selected_roi: dict[str, Any] | None,
    trace: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    canvas = frame_bgr.copy()
    _draw_xywh(canvas, gt_bbox, (0, 0, 255), 2)
    _put_label(canvas, f"GT label07 {frame_id}", (gt_bbox[0], gt_bbox[1] - 6), (0, 0, 255))

    if selected_roi:
        roi_bbox = selected_roi.get("bbox") or []
        _draw_xywh(canvas, roi_bbox, (0, 165, 255), 2)
        _put_label(
            canvas,
            f"ROI {selected_roi.get('roi_id', '')} {selected_roi.get('source', '')}",
            (int(roi_bbox[0]) if roi_bbox else 8, int(roi_bbox[1]) - 18 if roi_bbox else 36),
            (0, 165, 255),
        )

    if trace:
        for point in trace.get("positive_abnormal_image_points_sample") or []:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
        for point in trace.get("negative_abnormal_image_points_sample") or []:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)
        for sign_trace in trace.get("sign_traces") or []:
            sign = sign_trace.get("sign", "")
            for cluster in sign_trace.get("clusters") or []:
                bbox = cluster.get("image_bbox")
                accepted = bool(cluster.get("accepted"))
                color = (0, 255, 0) if accepted else (180, 180, 180)
                _draw_xywh(canvas, bbox, color, 1)
                if bbox:
                    reason = cluster.get("reject_reason") or "unknown"
                    _put_label(canvas, f"{sign} {reason}", (int(bbox[0]), int(bbox[1]) - 4), color)
                padded = cluster.get("padded_bbox")
                if padded and not accepted:
                    _draw_xywh(canvas, padded, (120, 120, 120), 1)
                decision = cluster.get("bbox_decision") or {}
                final_bbox = decision.get("final_bbox")
                if accepted and final_bbox:
                    _draw_xywh(canvas, final_bbox, (0, 255, 0), 2)

    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        bbox = candidate.get("bbox") or metadata.get("bbox")
        _draw_xywh(canvas, bbox, (0, 255, 0), 2)
        if bbox:
            _put_label(canvas, str(candidate.get("candidate_id", "cand")), (int(bbox[0]), int(bbox[1]) + int(bbox[3]) + 12), (0, 255, 0))

    if trace:
        text = (
            f"reject={trace.get('primary_reject_reason', '')} pts={trace.get('total_point_count', 0)} "
            f"valid={trace.get('valid_residual_point_count', 0)} clusters={trace.get('cluster_count', 0)}"
        )
        _put_label(canvas, text, (8, 22), (255, 255, 255))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_geometry_error_scatter(geometry_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if not geometry_df.empty:
        ax.scatter(geometry_df["gt_distance_m"], geometry_df["distance_ae_m"], label="distance AE", color="tab:red")
        ax.scatter(geometry_df["gt_distance_m"], geometry_df["height_ae_m"], label="height AE", color="tab:blue")
    ax.set_xlabel("GT distance (m)")
    ax.set_ylabel("Absolute error (m)")
    ax.set_title("Geometry Error vs Distance")
    ax.legend(loc="upper left")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_latency_histogram(latency_ms: Iterable[float], output_path: str | Path) -> None:
    latency = np.asarray(list(latency_ms), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    if latency.size > 0:
        bins = min(25, max(5, latency.size))
        ax.hist(latency, bins=bins, color="tab:green", edgecolor="black")
    ax.set_xlabel("Per-frame latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Latency Distribution")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 第四章實驗視覺化函式
# ─────────────────────────────────────────────────────────────────────────────


def colorize_depth_map(depth_map: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """將深度圖著色為彩虹色（JET colormap），返回 BGR 影像供 OpenCV 保存。"""
    d = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((*d.shape, 3), dtype=np.uint8)

    vmin_val = float(vmin) if vmin is not None else float(np.percentile(d[valid], 2))
    vmax_val = float(vmax) if vmax is not None else float(np.percentile(d[valid], 98))
    vrange = vmax_val - vmin_val
    if vrange < 1e-6:
        vrange = 1e-6
    d_norm = (d - vmin_val) / vrange
    d_norm = np.clip(d_norm, 0.0, 1.0)

    cmap = plt.cm.jet(d_norm)
    rgb = (cmap[:, :, :3] * 255).astype(np.uint8)
    bgr = rgb[..., ::-1].copy()
    return bgr


def save_depth_colorized(
    depth_map: np.ndarray,
    output_path: str | Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """儲存深度圖的彩虹色著色圖。"""
    colored = colorize_depth_map(depth_map, vmin=vmin, vmax=vmax)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), colored)


def save_intermediate_visualization(
    frame_bgr: np.ndarray,
    frame_id: str,
    frontend: FrontendResult,
    depth_map: np.ndarray,
    absolute_depth_map: np.ndarray,
    scale_result: ScaleAlignmentResult,
    plane: PlaneEstimate,
    candidate_observations: list[CandidateObservation],
    risk_assessments: list[RiskAssessment],
    output_dir: str | Path,
) -> None:
    """為單幀生成所有中間產物視覺化圖，結構對應論文圖 3-5 / 第四章實驗。

    輸出至 output_dir/intermediates/，包含：
      1. 原始 RGB 影像
      2. DA3 深度圖著色
      3. 尺度對齊後深度圖著色
      4. PIDNet 道路遮罩疊加
      5. 分析遮罩（ROI 候選區域）
      6. 異常點圖（按帶符號高差著色）
      7. 候選群集圖（按類型著色）
      8. 風險決策圖
    """
    output_dir = Path(output_dir)
    intermediates_dir = output_dir / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    h, w = frame_bgr.shape[:2]
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    frame_disp = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # 1. 原始 RGB 影像
    cv2.imwrite(str(intermediates_dir / f"{frame_id}_01_original.png"), frame_bgr)

    # 2. DA3 相對深度圖著色
    rel_colored = colorize_depth_map(depth_map)
    cv2.imwrite(str(intermediates_dir / f"{frame_id}_02_da3_depth.png"), rel_colored)

    # 3. 尺度對齊後深度圖著色
    abs_colored = colorize_depth_map(absolute_depth_map)
    status_text = f"sf={scale_result.scale_factor:.3f} h_hat={scale_result.h_hat_cam:.3f} [{scale_result.status}]"
    cv2.putText(abs_colored, status_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(intermediates_dir / f"{frame_id}_03_absolute_depth.png"), abs_colored)

    # 4. 道路遮罩疊加
    road_overlay = frame_bgr.copy()
    road_vis = _overlay_mask(road_overlay, frontend.road_mask, (60, 140, 60), 0.35)
    cv2.putText(road_vis, "PIDNet-S Road Mask", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(intermediates_dir / f"{frame_id}_04_road_mask.png"), road_vis)

    # 5. 分析遮罩（ROI 候選區域）
    analysis_overlay = frame_bgr.copy()
    analysis_vis = _overlay_mask(analysis_overlay, frontend.analysis_mask, (0, 80, 220), 0.30)
    for roi in frontend.roi_candidates:
        x, y, bw, bh = roi.bbox
        cv2.rectangle(analysis_vis, (x, y), (x + bw, y + bh), (0, 200, 255), 1)
    cv2.putText(analysis_vis, "Analysis Mask + ROI Candidates", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(intermediates_dir / f"{frame_id}_05_analysis_mask.png"), analysis_vis)

    # 6. 候選群集圖（按障礙物類型著色）
    cluster_overlay = frame_bgr.copy()
    # 道路遮罩為背景（半透明綠）
    cluster_overlay = _overlay_mask(cluster_overlay, frontend.road_mask, (60, 140, 60), 0.20)
    curb_color = (220, 140, 0)   # 青色 = 路緣/階差
    pothole_color = (50, 80, 230)  # 橙色 = 坑洞/凹陷
    unknown_color = (0, 0, 255)
    for cand in candidate_observations:
        x, y, bw, bh = cand.bbox
        if cand.candidate_type == "curb/step":
            color = curb_color
        elif cand.candidate_type == "pothole/pothole":
            color = pothole_color
        else:
            color = unknown_color
        cv2.rectangle(cluster_overlay, (x, y), (x + bw, y + bh), color, 2)
        label = f"{cand.candidate_id} {cand.obstacle_type} {cand.candidate_type}"
        cv2.putText(cluster_overlay, label, (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(intermediates_dir / f"{frame_id}_06_candidate_clusters.png"), cluster_overlay)

    # 7. 風險決策圖（邊框顏色表示風險等級）
    risk_overlay = frame_bgr.copy()
    risk_overlay = _overlay_mask(risk_overlay, frontend.road_mask, (60, 140, 60), 0.15)
    risk_by_id: dict[str, RiskAssessment] = {r.object_id: r for r in risk_assessments}
    for cand in candidate_observations:
        x, y, bw, bh = cand.bbox
        risk = risk_by_id.get(cand.candidate_id)
        if risk is None:
            continue
        if risk.decision == "danger":
            color = (0, 0, 255)
        elif risk.decision == "warning":
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(risk_overlay, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(
            risk_overlay,
            f"{risk.decision.upper()} w={risk.weight}",
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(intermediates_dir / f"{frame_id}_07_risk_decision.png"), risk_overlay)


def plot_scale_alignment_summary(
    frame_ids: list[str],
    scale_factors: list[float],
    statuses: list[str],
    h_hat_cam_values: list[float],
    output_path: str | Path,
) -> None:
    """生成尺度因子收斂圖（論文 Fig 3-5 類型）。"""
    valid_idx = [i for i, s in enumerate(statuses) if s == "ok"]
    if not valid_idx:
        valid_idx = list(range(len(frame_ids)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左：尺度因子隨幀變化
    ax = axes[0]
    sf_vals = [scale_factors[i] for i in valid_idx]
    fid_vals = [frame_ids[i] for i in valid_idx]
    ax.plot(range(len(sf_vals)), sf_vals, "b-o", markersize=4)
    ax.axhline(np.mean(sf_vals), color="red", linestyle="--", label=f"mean={np.mean(sf_vals):.3f}")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Scale factor")
    ax.set_title("Scale Alignment — s_t Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右：虛擬相機高度收斂
    ax = axes[1]
    hh_vals = [h_hat_cam_values[i] for i in valid_idx]
    ax.plot(range(len(hh_vals)), hh_vals, "g-o", markersize=4)
    ax.axhline(np.mean(hh_vals), color="red", linestyle="--", label=f"mean={np.mean(hh_vals):.3f}")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Virtual camera height (relative depth space)")
    ax.set_title("Relative-depth Plane — h_hat_cam Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("§3.3.3 Scale Alignment — Fig 3-5 Style", fontsize=11)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_candidate_type_distribution(
    frame_ids: list[str],
    candidate_counts: list[dict[str, int]],
    output_path: str | Path,
) -> None:
    """生成候選類型分佈柱狀圖（論文 Table 4-2 類型）。"""
    n_frames = len(frame_ids)
    curb_counts = [c.get("curb/step", 0) for c in candidate_counts]
    pothole_counts = [c.get("pothole/pothole", 0) for c in candidate_counts]
    unknown_counts = [c.get("", 0) for c in candidate_counts]

    x = np.arange(n_frames)
    width = 0.6
    fig, ax = plt.subplots(figsize=(max(6, n_frames * 0.8), 4))
    ax.bar(x, curb_counts, width, label="Curb/Step", color="#2196F3")
    ax.bar(x, pothole_counts, width, bottom=curb_counts, label="Pothole/Pothole", color="#FF9800")
    ax.bar(x, unknown_counts, width, bottom=[c + p for c, p in zip(curb_counts, pothole_counts)], label="Unknown", color="#9E9E9E")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Candidate count")
    ax.set_title("§4.3.3 Candidate Type Distribution per Frame")
    ax.set_xticks(x)
    ax.set_xticklabels(frame_ids, rotation=45, fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_risk_decision_matrix(
    tracked_objects: list[dict[str, Any]],
    risk_assessments: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """生成風險決策視覺化（論文 Table 5-2 類型）。

    輸出：
    - 混淆矩陣熱力圖（風除等級 vs 預測等級）
    - 風險決策散點圖（高度 vs 距離，顏色=決策）
    """
    risk_by_id = {r["object_id"]: r for r in risk_assessments}
    decisions = {"safe": 0, "warning": 1, "danger": 2}
    labels = ["Safe", "Warning", "Danger"]

    # 矩陣計數
    matrix = np.zeros((3, 3), dtype=int)
    for obj in tracked_objects:
        gt = obj.get("metadata", {}).get("risk_gt", None)
        ra = risk_by_id.get(obj.get("object_id"))
        if ra is None:
            continue
        pred = decisions.get(ra.get("decision", "safe"), 0)
        if gt is not None:
            gt_idx = labels.index(gt) if gt in labels else 0
            matrix[gt_idx, pred] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左：混淆矩陣
    ax = axes[0]
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("§5.4 Risk Decision Confusion Matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black" if matrix[i, j] < matrix.max() / 2 else "white")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    # 右：風險決策散點圖（高度 vs 前向距離）
    ax = axes[1]
    colors = {"safe": "green", "warning": "orange", "danger": "red"}
    for ra in risk_assessments:
        col = colors.get(ra.get("decision", "safe"), "gray")
        ax.scatter(ra.get("metadata", {}).get("distance_m", 0), ra.get("metadata", {}).get("height_m", 0), c=col, s=60, alpha=0.7)
    ax.set_xlabel("Forward distance (m)")
    ax.set_ylabel("Obstacle height (m)")
    ax.set_title("§5.4 Risk Decision — Height vs Distance")
    ax.legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=l, markersize=8) for l, c in colors.items()])
    ax.grid(True, alpha=0.3)

    fig.suptitle("§5.4 風險判斷與決策輸出", fontsize=11)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_plane_inlier_ratio(
    frame_ids: list[str],
    inlier_ratios: list[float],
    support_counts: list[int],
    output_path: str | Path,
) -> None:
    """生成地平面擬合品質圖（論文 Section 3.3.3 / 地平面章節）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(range(len(frame_ids)), inlier_ratios, "g-o", markersize=4)
    ax.axhline(np.mean(inlier_ratios), color="red", linestyle="--", label=f"mean={np.mean(inlier_ratios):.2f}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Inlier ratio")
    ax.set_title("§3.3.3 Ground Plane — RANSAC Inlier Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(frame_ids)), support_counts, color="steelblue", edgecolor="black")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Supporting point count")
    ax.set_title("§3.3.3 Ground Plane — Support Point Count")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("§3.3.3 地平面擬合品質", fontsize=11)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pipeline_overview(
    frame_ids: list[str],
    stage_latencies_ms: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    """生成管線各階段延遲堆疊柱狀圖（論文部署效能章節）。"""
    fig, ax = plt.subplots(figsize=(max(8, len(frame_ids) * 1.2), 5))
    bottom = np.zeros(len(frame_ids))
    colors = plt.cm.tab10.colors
    stage_names = list(stage_latencies_ms.keys())
    for i, stage in enumerate(stage_names):
        vals = stage_latencies_ms.get(stage, [])
        if len(vals) < len(frame_ids):
            vals = list(vals) + [0.0] * (len(frame_ids) - len(vals))
        ax.bar(range(len(frame_ids)), vals[:len(frame_ids)], bottom=bottom, label=stage, color=colors[i % len(colors)], edgecolor="black", linewidth=0.3)
        bottom += np.array(vals[:len(frame_ids)])

    ax.set_xlabel("Frame")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("§6.2 Deployment — Per-Stage Latency Breakdown")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
