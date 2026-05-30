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
from .models import CandidateObservation, PlaneEstimate, RiskAssessment, RoiCandidate, ScaleAlignmentResult
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

    _draw_processing_roi(canvas, frontend.processing_roi, (0, 0, 255))

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
    h_hat_cam = (scale_result.metadata or {}).get("h_hat_cam", 0.0)
    status_text = f"sf={scale_result.scale_factor:.3f} h_hat={float(h_hat_cam):.3f} [{scale_result.status}]"
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
            f"{risk.decision.upper()} w={risk.risk_weight}",
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


_DECISION_COLORS_BGR = {"safe": (0, 180, 0), "warning": (0, 165, 255), "danger": (0, 0, 255)}
_DECISION_COLORS_MPL = {"safe": "tab:green", "warning": "tab:orange", "danger": "tab:red"}
_RISK_MARKERS = {"omega0": "o", "omega1": "^", "omega2": "s"}


def _ensure_parent(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _draw_processing_roi(canvas: np.ndarray, processing_roi: dict[str, Any] | None, color: tuple[int, int, int]) -> None:
    roi = processing_roi or {}
    if roi.get("polygon"):
        points = np.asarray(roi["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=2)
        label_x, label_y = points.reshape((-1, 2))[0]
        label = str(roi.get("effective_mode") or roi.get("mode") or "processing_roi")
        _put_label(canvas, label, (int(label_x), int(label_y) - 6), color)
    elif roi.get("bbox"):
        _draw_xywh(canvas, roi["bbox"], color, 2)
        x, y = [int(value) for value in roi["bbox"][:2]]
        _put_label(canvas, "processing_roi", (x, y - 6), color)

    foe_roi = roi.get("foe_road_triangle") or {}
    foe = foe_roi.get("foe") or {}
    foe_xy = foe.get("smoothed") or foe.get("raw") or foe.get("foe")
    if foe_xy is not None:
        x, y = int(round(float(foe_xy[0]))), int(round(float(foe_xy[1])))
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            cv2.drawMarker(canvas, (x, y), (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
            _put_label(canvas, "FoE", (x + 6, y - 6), (255, 0, 255))
    road_edges = foe_roi.get("road_edges") or {}
    for name, edge in road_edges.items():
        if not isinstance(edge, dict):
            continue
        endpoints = edge.get("endpoints") or []
        if len(endpoints) == 2:
            p0 = tuple(int(value) for value in endpoints[0])
            p1 = tuple(int(value) for value in endpoints[1])
            cv2.line(canvas, p0, p1, (255, 0, 255), 2)
            _put_label(canvas, str(name), p0, (255, 0, 255))
    if roi.get("fallback") and roi.get("fallback_reason"):
        _put_label(canvas, f"fallback={roi.get('fallback_reason')}", (8, canvas.shape[0] - 12), color)


def _draw_points(
    canvas: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    *,
    max_points: int = 1200,
    radius: int = 1,
) -> None:
    pts = np.asarray(points)
    if pts.size == 0:
        return
    pts = pts.reshape((-1, pts.shape[-1]))[:, :2]
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    h, w = canvas.shape[:2]
    for point in pts:
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), radius, color, -1)


def _source_color(source: str) -> tuple[int, int, int]:
    if source == "depth_gradient_contour":
        return (255, 0, 255)
    if source == "depth_residual_contour":
        return (0, 165, 255)
    if source == "hybrid":
        return (0, 255, 255)
    if source.startswith("fallback"):
        return (255, 180, 0)
    return (0, 255, 0)


def _risk_by_object_id(risk_assessments: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in risk_assessments:
        object_id = getattr(item, "object_id", None)
        if object_id is None and isinstance(item, dict):
            object_id = item.get("object_id")
        if object_id is not None:
            out[str(object_id)] = item
    return out


def _get_risk_value(risk: Any, key: str, default: Any = None) -> Any:
    if isinstance(risk, dict):
        return risk.get(key, default)
    return getattr(risk, key, default)


def plot_motion_profile(
    motion_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    frame_ids = [str(row.get("frame_id", idx)) for idx, row in enumerate(motion_rows)]
    x = np.arange(len(motion_rows))
    speeds = [float(row.get("speed_mps", 0.0) or 0.0) for row in motion_rows]
    displacements = [float(row.get("forward_displacement_m", 0.0) or 0.0) for row in motion_rows]
    cumulative = [float(row.get("cumulative_forward_m", 0.0) or 0.0) for row in motion_rows]

    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(motion_rows) * 0.65), 8), sharex=True)
    axes[0].plot(x, speeds, "o-", color="tab:blue")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].set_title("Step2 Motion Profile")
    axes[1].bar(x, displacements, color="tab:orange")
    axes[1].set_ylabel("forward Δ (m)")
    axes[2].plot(x, cumulative, "o-", color="tab:green")
    axes[2].set_ylabel("cumulative (m)")
    axes[2].set_xlabel("frame")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(frame_ids, rotation=45, ha="right", fontsize=7)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_parent(output_path), dpi=160)
    plt.close(fig)


def plot_calibration_basis(
    intrinsics: dict[str, Any],
    extrinsics: dict[str, Any],
    output_path: str | Path,
) -> None:
    camera_matrix = np.asarray(intrinsics.get("camera_matrix", np.eye(3)), dtype=float)
    rotation = np.asarray(extrinsics.get("rotation_matrix", np.eye(3)), dtype=float)
    translation = np.asarray(extrinsics.get("translation_vector", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)

    fig = plt.figure(figsize=(11, 5))
    ax_text = fig.add_subplot(1, 2, 1)
    ax_text.axis("off")
    text = (
        "Step3 Calibration Basis\n\n"
        f"image size: {intrinsics.get('image_width', 'na')} x {intrinsics.get('image_height', 'na')}\n\n"
        f"camera_matrix:\n{np.array2string(camera_matrix, precision=3)}\n\n"
        f"rotation_matrix:\n{np.array2string(rotation, precision=3)}\n\n"
        f"translation_vector:\n{np.array2string(translation, precision=3)}"
    )
    ax_text.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    origin = np.zeros(3)
    axis_len = 1.0
    colors = ["tab:red", "tab:green", "tab:blue"]
    labels = ["X", "Y", "Z"]
    for vec, color, label in zip(np.eye(3), colors, labels):
        ax.quiver(*origin, *(vec * axis_len), color=color, linewidth=2)
        ax.text(*(vec * axis_len * 1.1), label, color=color)
    frustum = np.array([
        [0.0, 0.0, 0.0],
        [-0.45, -0.30, 0.9],
        [0.45, -0.30, 0.9],
        [0.45, 0.30, 0.9],
        [-0.45, 0.30, 0.9],
    ])
    for start, end in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
        pts = frustum[[start, end]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="black", alpha=0.6)
    ax.set_title("Camera coordinates / view frustum")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(_ensure_parent(output_path), dpi=160)
    plt.close(fig)


def save_workflow_frame_figures(
    *,
    frame_bgr: np.ndarray,
    frame_id: str,
    frame_path: str | Path,
    frontend: FrontendResult,
    relative_depth: np.ndarray,
    absolute_depth: np.ndarray,
    scale_result: ScaleAlignmentResult,
    road_sample: dict[str, Any],
    plane: PlaneEstimate,
    obstacle_analysis_mask: np.ndarray,
    candidate_sample: dict[str, Any],
    depth_roi_candidates: list[RoiCandidate],
    selected_rois: list[RoiCandidate],
    candidate_clusters: list[CandidateObservation],
    tracked_objects: list[dict[str, Any]],
    risk_assessments: list[Any],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig01 = frame_bgr.copy()
    _put_label(fig01, f"Step1 frame_id={frame_id}", (8, 22), (255, 255, 255))
    _put_label(fig01, f"source={Path(frame_path).as_posix()}", (8, 42), (255, 255, 255))
    cv2.imwrite(str(output_dir / "fig01_frame_input.png"), fig01)

    fig04 = frame_bgr.copy()
    fig04 = _overlay_mask(fig04, frontend.road_mask, (60, 140, 60), 0.30)
    if getattr(frontend, "analysis_mask", None) is not None:
        fig04 = _overlay_mask(fig04, frontend.analysis_mask, (0, 80, 220), 0.20)
    if obstacle_analysis_mask is not None:
        fig04 = _overlay_mask(fig04, obstacle_analysis_mask, (180, 80, 0), 0.18)
    _draw_processing_roi(fig04, frontend.processing_roi, (0, 0, 255))
    for roi in frontend.roi_candidates:
        _draw_xywh(fig04, roi.bbox, (0, 200, 255), 1)
    _put_label(fig04, f"Step4 backend={frontend.backend} frontend_rois={len(frontend.roi_candidates)}", (8, 22), (255, 255, 255))
    cv2.imwrite(str(output_dir / "fig04_fallback_segmentation_roi.png"), fig04)

    fig05 = colorize_depth_map(relative_depth)
    _put_label(fig05, f"Step5 relative depth shape={list(relative_depth.shape)}", (8, 22), (255, 255, 255))
    cv2.imwrite(str(output_dir / "fig05_relative_depth_da3.png"), fig05)

    fig06 = colorize_depth_map(absolute_depth)
    metadata = scale_result.metadata or {}
    _put_label(
        fig06,
        f"Step6 scale={scale_result.scale_factor:.4f} status={scale_result.status} h_hat={float(metadata.get('h_hat_cam', 0.0)):.4f}",
        (8, 22),
        (255, 255, 255),
    )
    _put_label(fig06, f"candidates={scale_result.candidate_count} inliers={scale_result.selected_count}", (8, 42), (255, 255, 255))
    cv2.imwrite(str(output_dir / "fig06_scale_alignment_absolute_depth.png"), fig06)

    fig07 = frame_bgr.copy()
    fig07 = _overlay_mask(fig07, frontend.road_mask, (60, 140, 60), 0.20)
    _draw_points(fig07, np.asarray(road_sample.get("image_points", [])), (0, 255, 255), max_points=1800, radius=1)
    _put_label(fig07, f"Step7 plane={plane.status}/{plane.source}", (8, 22), (255, 255, 255))
    _put_label(fig07, f"support={plane.support_count} inlier={plane.inlier_ratio:.3f} n={np.asarray(plane.normal).round(3).tolist()}", (8, 42), (255, 255, 255))
    cv2.imwrite(str(output_dir / "fig07_ground_plane_ransac.png"), fig07)

    fig08 = frame_bgr.copy()
    fig08 = _overlay_mask(fig08, obstacle_analysis_mask, (0, 80, 220), 0.25)
    _draw_processing_roi(fig08, frontend.processing_roi, (0, 0, 255))
    _draw_points(fig08, np.asarray(candidate_sample.get("image_points", [])), (0, 255, 255), max_points=2500, radius=1)
    _put_label(fig08, f"Step8 candidate sampled points={len(candidate_sample.get('image_points', []))}", (8, 22), (255, 255, 255))
    cv2.imwrite(str(output_dir / "fig08_candidate_point_sampling.png"), fig08)

    fig09 = frame_bgr.copy()
    source_counts: dict[str, int] = {}
    for roi in depth_roi_candidates:
        source_counts[roi.source] = source_counts.get(roi.source, 0) + 1
        color = _source_color(roi.source)
        _draw_xywh(fig09, roi.bbox, color, 2)
        x, y = [int(value) for value in roi.bbox[:2]]
        _put_label(fig09, roi.source.replace("_contour", ""), (x, y - 4), color)
    _put_label(fig09, f"Step9 depth rois={len(depth_roi_candidates)} {source_counts}", (8, 22), (255, 255, 255))
    cv2.imwrite(str(output_dir / "fig09_depth_roi_generation.png"), fig09)

    fig10 = frame_bgr.copy()
    selected_counts: dict[str, int] = {}
    for roi in selected_rois:
        selected_counts[roi.source] = selected_counts.get(roi.source, 0) + 1
        color = _source_color(roi.source)
        _draw_xywh(fig10, roi.bbox, color, 2)
        x, y = [int(value) for value in roi.bbox[:2]]
        _put_label(fig10, roi.roi_id, (x, y - 4), color)
    _put_label(fig10, f"Step10 selected rois={len(selected_rois)} {selected_counts}", (8, 22), (255, 255, 255))
    cv2.imwrite(str(output_dir / "fig10_selected_roi_generation.png"), fig10)

    fig11 = frame_bgr.copy()
    fig11 = _overlay_mask(fig11, frontend.road_mask, (60, 140, 60), 0.15)
    for cand in candidate_clusters:
        color = (255, 0, 0) if cand.obstacle_type == "positive" else (0, 255, 255) if cand.obstacle_type == "negative" else (0, 0, 255)
        _draw_xywh(fig11, cand.bbox, color, 2)
        x, y = [int(value) for value in cand.bbox[:2]]
        _put_label(fig11, f"{cand.candidate_id} pts={cand.point_count} abn={cand.abnormal_count}", (x, y - 4), color)
    _put_label(fig11, f"Step11 DBSCAN clusters={len(candidate_clusters)}", (8, 22), (255, 255, 255))
    cv2.imwrite(str(output_dir / "fig11_dbscan_candidate_clusters.png"), fig11)

    fig12 = frame_bgr.copy()
    for cand in candidate_clusters:
        color = (255, 0, 0) if cand.obstacle_type == "positive" else (0, 255, 255) if cand.obstacle_type == "negative" else (0, 0, 255)
        _draw_xywh(fig12, cand.bbox, color, 2)
        x, y = [int(value) for value in cand.bbox[:2]]
        label = f"{cand.candidate_type or 'unknown'} {cand.obstacle_type} Z={cand.distance_m:.2f} H={cand.height_m:.2f} W={cand.width_m:.2f}"
        _put_label(fig12, label, (x, y - 4), color)
    _put_label(fig12, "Step12 candidate type classification", (8, 22), (255, 255, 255))
    cv2.imwrite(str(output_dir / "fig12_candidate_type_classification.png"), fig12)

    save_overlay(
        frame_bgr=frame_bgr,
        frame_id=frame_id,
        frontend=frontend,
        road_points=np.asarray(road_sample.get("image_points", [])),
        plane=plane.to_dict(),
        tracked_objects=tracked_objects,
        output_path=output_dir / "fig18_frame_state_overlay.png",
        candidate_rois=selected_rois,
    )


def plot_sequence_workflow_figures(
    *,
    sequence_rows: list[dict[str, Any]],
    raw_risk_rows: list[dict[str, Any]],
    risk_event_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = [str(row.get("frame_id", idx)) for idx, row in enumerate(sequence_rows)]
    x = np.arange(len(sequence_rows))

    cross_counts = [int(row.get("cross_frame_match_count", 0) or 0) for row in sequence_rows]
    fig, ax = plt.subplots(figsize=(max(7, len(sequence_rows) * 0.55), 4))
    ax.bar(x, cross_counts, color="tab:blue", edgecolor="black")
    ax.set_title("Step13 Cross-frame Matches")
    ax.set_xlabel("frame")
    ax.set_ylabel("match count")
    ax.set_xticks(x)
    ax.set_xticklabels(frame_ids, rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig13_cross_frame_matches.png", dpi=160)
    plt.close(fig)

    states = sorted({state for row in sequence_rows for state in (row.get("tracked_state_counts") or {}).keys()})
    if not states:
        states = ["none"]
    bottom = np.zeros(len(sequence_rows))
    fig, ax = plt.subplots(figsize=(max(7, len(sequence_rows) * 0.7), 4.5))
    for idx, state in enumerate(states):
        values = [int((row.get("tracked_state_counts") or {}).get(state, 0) or 0) for row in sequence_rows]
        ax.bar(x, values, bottom=bottom, label=state, color=plt.cm.tab10(idx % 10), edgecolor="black", linewidth=0.25)
        bottom += np.asarray(values, dtype=float)
    ax.set_title("Step14 Tracker Lifecycle States")
    ax.set_xlabel("frame")
    ax.set_ylabel("tracked object count")
    ax.set_xticks(x)
    ax.set_xticklabels(frame_ids, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig14_tracker_lifecycle.png", dpi=160)
    plt.close(fig)

    temporal_keyframes = [int(row.get("temporal_keyframe_count", 0) or 0) for row in sequence_rows]
    temporal_attempted = [int(row.get("temporal_attempted_count", 0) or 0) for row in sequence_rows]
    temporal_ok = [int(row.get("temporal_ok_count", 0) or 0) for row in sequence_rows]
    temporal_applied = [int(row.get("temporal_applied_count", 0) or 0) for row in sequence_rows]
    fig, ax = plt.subplots(figsize=(max(7, len(sequence_rows) * 0.55), 4))
    ax.plot(x, temporal_keyframes, "o-", label="history/keyframes")
    ax.plot(x, temporal_attempted, "o-", label="attempted")
    ax.plot(x, temporal_ok, "o-", label="ok")
    ax.plot(x, temporal_applied, "o-", label="applied")
    ax.set_title("Step15 Temporal Snapshot / Measurement")
    ax.set_xlabel("frame")
    ax.set_ylabel("count")
    ax.set_xticks(x)
    ax.set_xticklabels(frame_ids, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig15_temporal_snapshot_history.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if raw_risk_rows:
        for decision, color in _DECISION_COLORS_MPL.items():
            for weight, marker in _RISK_MARKERS.items():
                rows = [row for row in raw_risk_rows if row.get("decision") == decision and row.get("risk_weight") == weight]
                if not rows:
                    continue
                ax.scatter(
                    [float(row.get("distance_m", 0.0) or 0.0) for row in rows],
                    [float(row.get("height_m", 0.0) or 0.0) for row in rows],
                    c=color,
                    marker=marker,
                    alpha=0.75,
                    label=f"{decision}/{weight}",
                )
    ax.set_title("Step16 Raw Risk Assessment")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("height (m)")
    ax.grid(True, alpha=0.3)
    if raw_risk_rows:
        ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig16_raw_risk_assessment.png", dpi=160)
    plt.close(fig)

    decisions = ["warning", "danger", "safe"]
    bottom = np.zeros(len(sequence_rows))
    fig, ax = plt.subplots(figsize=(max(7, len(sequence_rows) * 0.55), 4))
    for decision in decisions:
        values = [int((row.get("risk_event_decision_counts") or {}).get(decision, 0) or 0) for row in sequence_rows]
        ax.bar(x, values, bottom=bottom, label=decision, color=_DECISION_COLORS_MPL.get(decision, "tab:gray"), edgecolor="black", linewidth=0.25)
        bottom += np.asarray(values, dtype=float)
    ax.set_title("Step17 Filtered Risk Events")
    ax.set_xlabel("frame")
    ax.set_ylabel("event count")
    ax.set_xticks(x)
    ax.set_xticklabels(frame_ids, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig17_filtered_risk_events.png", dpi=160)
    plt.close(fig)

    stage_names = [
        "frontend",
        "depth_inference",
        "scale_alignment",
        "plane_fit",
        "candidate_generation",
        "temporal_measurement",
        "tracking_and_risk",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].axis("off")
    summary_text = "Step19 Pipeline Summary\n\n" + "\n".join(
        [
            f"sequence_id: {summary.get('sequence_id', '')}",
            f"frames_processed: {summary.get('frames_processed', 0)}",
            f"mean_latency_ms: {float(summary.get('mean_latency_ms', 0.0)):.2f}",
            f"max_latency_ms: {float(summary.get('max_latency_ms', 0.0)):.2f}",
            f"risk_event_count: {summary.get('risk_event_count', 0)}",
        ]
    )
    axes[0].text(0.02, 0.98, summary_text, va="top", ha="left", family="monospace", fontsize=11)
    bottom = np.zeros(len(sequence_rows))
    for idx, stage in enumerate(stage_names):
        values = [float((row.get("timing_ms") or {}).get(stage, 0.0) or 0.0) for row in sequence_rows]
        axes[1].bar(x, values, bottom=bottom, label=stage, color=plt.cm.tab20(idx % 20), linewidth=0.2)
        bottom += np.asarray(values, dtype=float)
    axes[1].set_title("Per-stage latency")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("ms")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(frame_ids, rotation=45, ha="right", fontsize=7)
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig19_pipeline_summary.png", dpi=160)
    plt.close(fig)


def plot_evaluation_summary_figure(summary: dict[str, Any], output_path: str | Path) -> None:
    detection = summary.get("detection_metrics") or {}
    risk_conf = np.asarray(summary.get("risk_confusion_matrix") or np.zeros((3, 3)), dtype=float)
    plane = summary.get("plane_metrics") or {}
    scale = summary.get("scale_metrics") or {}
    tracking = summary.get("tracking_metrics") or {}
    latency = summary.get("latency_metrics") or {}
    temporal = summary.get("temporal_measurement_metrics") or {}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    metric_names = ["precision", "recall", "f1", "accuracy"]
    metric_values = [float(detection.get(name, 0.0) or 0.0) for name in metric_names]
    axes[0, 0].bar(metric_names, metric_values, color=["tab:blue", "tab:green", "tab:orange", "tab:purple"])
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_title("Detection metrics")
    axes[0, 0].grid(axis="y", alpha=0.3)
    for idx, value in enumerate(metric_values):
        axes[0, 0].text(idx, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)

    im = axes[0, 1].imshow(risk_conf, cmap="Blues")
    axes[0, 1].set_xticks(range(3))
    axes[0, 1].set_yticks(range(3))
    axes[0, 1].set_xticklabels(["omega0", "omega1", "omega2"])
    axes[0, 1].set_yticklabels(["omega0", "omega1", "omega2"])
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("GT")
    axes[0, 1].set_title("Risk confusion matrix")
    for row in range(risk_conf.shape[0]):
        for col in range(risk_conf.shape[1]):
            axes[0, 1].text(col, row, int(risk_conf[row, col]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.045, pad=0.04)

    axes[1, 0].axis("off")
    scale_stability = scale.get("scale_factor_stability") or {}
    plane_normal = plane.get("plane_normal_error_deg") or {}
    plane_inlier = plane.get("plane_inlier_ratio") or {}
    summary_text = "Geometry / scale / tracking\n\n" + "\n".join(
        [
            f"plane normal mean deg: {float(plane_normal.get('mean', 0.0) or 0.0):.4f}",
            f"plane inlier mean: {float(plane_inlier.get('mean', 0.0) or 0.0):.4f}",
            f"scale factor mean: {float(scale_stability.get('mean', 0.0) or 0.0):.4f}",
            f"scale factor std: {float(scale_stability.get('std', 0.0) or 0.0):.4f}",
            f"tracking continuity: {float(tracking.get('tracking_continuity', 0.0) or 0.0):.4f}",
            f"active objects: {int(tracking.get('active_object_count', 0) or 0)}",
        ]
    )
    axes[1, 0].text(0.02, 0.98, summary_text, va="top", ha="left", family="monospace", fontsize=10)

    axes[1, 1].axis("off")
    counts_text = "Latency / temporal\n\n" + "\n".join(
        [
            f"mean latency ms: {float(latency.get('mean_latency_ms', 0.0) or 0.0):.2f}",
            f"max latency ms: {float(latency.get('max_latency_ms', 0.0) or 0.0):.2f}",
            f"temporal attempted: {int(temporal.get('attempted_count', 0) or 0)}",
            f"temporal ok: {int(temporal.get('ok_count', 0) or 0)}",
            f"temporal applied: {int(temporal.get('applied_count', 0) or 0)}",
            f"TP/FP/FN/TN: {int(detection.get('tp', 0) or 0)}/{int(detection.get('fp', 0) or 0)}/{int(detection.get('fn', 0) or 0)}/{int(detection.get('tn', 0) or 0)}",
        ]
    )
    axes[1, 1].text(0.02, 0.98, counts_text, va="top", ha="left", family="monospace", fontsize=10)

    fig.suptitle("Step20 Evaluation Summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(_ensure_parent(output_path), dpi=160)
    plt.close(fig)
