from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import json

from .io_utils import read_gt_obstacles, read_gt_plane, read_jsonl
from .visualization import plot_confusion_matrix, plot_geometry_error_scatter, plot_latency_histogram

RISK_ORDER = {"omega0": 0, "omega1": 1, "omega2": 2}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _risk_max(risks: list[str]) -> str:
    if not risks:
        return "omega0"
    return max(risks, key=lambda item: RISK_ORDER.get(item, 0))


def _active_objects(frame_state: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in frame_state.get("tracked_objects", []):
        if item.get("state") == "retired":
            continue
        out.append(item)
    return out


def _per_frame_gt(gt_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for frame_id, rows in gt_df.groupby("frame_id"):
        risks = rows["risk_gt"].tolist()
        out[str(frame_id)] = {
            "has_hazard": any(risk != "omega0" for risk in risks),
            "max_risk": _risk_max(risks),
            "rows": rows.copy(),
        }
    return out


def _per_frame_pred(frame_states: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in frame_states:
        objects = _active_objects(row)
        risks = [(obj.get("risk") or {}).get("risk_weight", "omega0") for obj in objects]
        out[str(row["frame_id"])] = {
            "has_hazard": any(risk != "omega0" for risk in risks),
            "max_risk": _risk_max(risks),
            "row": row,
            "objects": objects,
        }
    return out


def _pair_geometry(gt_rows: pd.DataFrame, pred_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if gt_rows.empty or not pred_objects:
        return []
    matches: list[dict[str, Any]] = []
    used: set[int] = set()
    for _, gt in gt_rows.iterrows():
        best_idx: int | None = None
        best_err: float | None = None
        for idx, pred in enumerate(pred_objects):
            if idx in used:
                continue
            if pred.get("obstacle_type") != gt["label"]:
                continue
            err = abs(float(pred.get("distance_m", 0.0)) - float(gt["d_gt_m"]))
            if best_err is None or err < best_err:
                best_err = err
                best_idx = idx
        if best_idx is None:
            continue
        pred = pred_objects[best_idx]
        used.add(best_idx)
        matches.append(
            {
                "frame_id": str(gt["frame_id"]),
                "object_id_gt": str(gt["object_id"]),
                "object_id_pred": str(pred.get("object_id", "")),
                "label": str(gt["label"]),
                "gt_distance_m": float(gt["d_gt_m"]),
                "height_ae_m": abs(float(pred.get("height_m", 0.0)) - float(gt["h_gt_m"])),
                "width_ae_m": abs(float(pred.get("width_m", 0.0)) - float(gt["w_gt_m"])),
                "distance_ae_m": abs(float(pred.get("distance_m", 0.0)) - float(gt["d_gt_m"])),
            }
        )
    return matches


def _err_stats(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"mae": 0.0, "rmse": 0.0, "max_error": 0.0}
    arr = series.to_numpy(dtype=float)
    return {
        "mae": float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "max_error": float(np.max(arr)),
    }


def _bbox_iou_xywh(a: list[Any], b: list[Any]) -> float:
    ax, ay, aw, ah = [float(value) for value in a]
    bx, by, bw, bh = [float(value) for value in b]
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return float(inter / union) if union > 0.0 else 0.0


def _best_bbox_iou(gt_bbox: list[int], items: list[dict[str, Any]], *, skip_retired: bool = False) -> dict[str, Any]:
    best_iou = 0.0
    best_id = ""
    best_state = ""
    for item in items:
        if skip_retired and item.get("state") == "retired":
            continue
        bbox = item.get("bbox") or (item.get("metadata") or {}).get("bbox") or (item.get("metadata") or {}).get("roi_bbox")
        if not bbox:
            continue
        score = _bbox_iou_xywh(gt_bbox, bbox)
        if score > best_iou:
            best_iou = score
            best_id = str(item.get("roi_id") or item.get("candidate_id") or item.get("object_id") or "")
            best_state = str(item.get("state") or "")
    return {"iou": float(best_iou), "id": best_id, "state": best_state, "hit": bool(best_iou > 0.0)}


def _label_layer_diagnostics(
    gt_df: pd.DataFrame,
    frame_states: list[dict[str, Any]],
    label: str,
    tables_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_frame = {str(frame_state["frame_id"]): frame_state for frame_state in frame_states}
    label_text = str(label)
    gt_label_text = gt_df["label"].map(lambda value: str(value).zfill(len(label_text)))
    gt_rows = gt_df[gt_label_text == label_text]
    layer_names = ["frontend_roi", "depth_roi", "selected_roi", "candidate", "tracked"]
    hit_counts = {name: 0 for name in layer_names}
    miss_layer_counts: dict[str, int] = {}

    for _, gt in gt_rows.iterrows():
        frame_id = str(gt["frame_id"])
        frame_state = by_frame.get(frame_id, {})
        stats = frame_state.get("road_mask_stats") or {}
        gt_bbox = [int(gt["bbox_x"]), int(gt["bbox_y"]), int(gt["bbox_w"]), int(gt["bbox_h"])]
        layer_items = {
            "frontend_roi": stats.get("frontend_roi_diagnostics") or [],
            "depth_roi": stats.get("depth_roi_diagnostics") or [],
            "selected_roi": stats.get("selected_roi_diagnostics") or [],
            "candidate": frame_state.get("candidate_clusters") or [],
            "tracked": frame_state.get("tracked_objects") or [],
        }
        layer_hits = {
            name: _best_bbox_iou(gt_bbox, items, skip_retired=(name == "tracked"))
            for name, items in layer_items.items()
        }
        for name, hit in layer_hits.items():
            if hit["hit"]:
                hit_counts[name] += 1
        miss_layer = "hit"
        for name in layer_names:
            if not layer_hits[name]["hit"]:
                miss_layer = name
                break
        miss_layer_counts[miss_layer] = miss_layer_counts.get(miss_layer, 0) + 1
        rows.append(
            {
                "frame_id": frame_id,
                "object_id": str(gt["object_id"]),
                "label": str(label),
                "miss_layer": miss_layer,
                **{f"{name}_hit": layer_hits[name]["hit"] for name in layer_names},
                **{f"{name}_best_iou": layer_hits[name]["iou"] for name in layer_names},
                **{f"{name}_best_id": layer_hits[name]["id"] for name in layer_names},
                "tracked_best_state": layer_hits["tracked"]["state"],
            }
        )

    diag_df = pd.DataFrame(rows)
    if not diag_df.empty:
        diag_df.to_csv(tables_dir / f"label_{label}_layer_diagnostics.csv", index=False)
    total = int(len(rows))
    return {
        "label": str(label),
        "total_gt_frames": total,
        "hit_counts": {key: int(value) for key, value in hit_counts.items()},
        "miss_layer_counts": {key: int(value) for key, value in miss_layer_counts.items()},
        "mean_iou": {
            name: float(diag_df[f"{name}_best_iou"].mean()) if not diag_df.empty else 0.0
            for name in layer_names
        },
        "max_iou": {
            name: float(diag_df[f"{name}_best_iou"].max()) if not diag_df.empty else 0.0
            for name in layer_names
        },
    }


def _temporal_measurement_metrics(frame_states: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for frame_state in frame_states:
        rows.extend(frame_state.get("temporal_measurements") or [])
    status_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    qualities: list[float] = []
    applied_count = 0
    ok_count = 0
    for row in rows:
        status = str(row.get("status", "unknown"))
        method = str(row.get("method", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        method_counts[method] = method_counts.get(method, 0) + 1
        if status == "ok":
            ok_count += 1
        if row.get("applied"):
            applied_count += 1
        if row.get("quality") is not None:
            qualities.append(float(row.get("quality", 0.0)))
    return {
        "attempted_count": int(len(rows)),
        "ok_count": int(ok_count),
        "applied_count": int(applied_count),
        "mean_quality": float(np.mean(qualities)) if qualities else 0.0,
        "status_counts": status_counts,
        "method_counts": method_counts,
    }


def evaluate_pipeline(dataset_dir: str | Path, output_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """以第五章定義的事件與幾何指標評估管線輸出。"""

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    frame_states = read_jsonl(output_dir / "frame_states.jsonl")
    gt_df = read_gt_obstacles(dataset_dir / "gt_obstacles.csv")
    gt_plane = read_gt_plane(dataset_dir / "gt_plane.json")

    gt_by_frame = _per_frame_gt(gt_df)
    pred_by_frame = _per_frame_pred(frame_states)
    all_ids = sorted(set(gt_by_frame) | set(pred_by_frame))

    tp = fp = fn = tn = 0
    risk_conf = np.zeros((3, 3), dtype=int)
    geometry_rows: list[dict[str, Any]] = []
    plane_err_deg: list[float] = []
    plane_inlier: list[float] = []
    timing_ms: list[float] = []
    scale_factors: list[float] = []
    active_object_count = 0
    continuing_object_count = 0

    gt_n = np.asarray(gt_plane.get("normal", [1.0, 0.0, 0.0]), dtype=float)
    gt_norm = np.linalg.norm(gt_n)
    gt_n = gt_n / gt_norm if gt_norm > 0 else np.array([1.0, 0.0, 0.0], dtype=float)

    for frame_id in all_ids:
        gt = gt_by_frame.get(frame_id, {"has_hazard": False, "max_risk": "omega0", "rows": pd.DataFrame(columns=gt_df.columns)})
        pred = pred_by_frame.get(frame_id, {"has_hazard": False, "max_risk": "omega0", "row": {"timing_ms": {}}, "objects": []})

        if gt["has_hazard"] and pred["has_hazard"]:
            tp += 1
        elif not gt["has_hazard"] and pred["has_hazard"]:
            fp += 1
        elif gt["has_hazard"] and not pred["has_hazard"]:
            fn += 1
        else:
            tn += 1

        gidx = RISK_ORDER.get(gt["max_risk"], 0)
        pidx = RISK_ORDER.get(pred["max_risk"], 0)
        risk_conf[gidx, pidx] += 1
        geometry_rows.extend(_pair_geometry(gt["rows"], pred["objects"]))

        plane = pred["row"].get("plane_model", {})
        if plane.get("status") == "ok":
            n = np.asarray(plane.get("normal", [1.0, 0.0, 0.0]), dtype=float)
            n_norm = np.linalg.norm(n)
            if n_norm > 0:
                n = n / n_norm
                cos = np.clip(float(np.dot(n, gt_n)), -1.0, 1.0)
                plane_err_deg.append(float(np.degrees(np.arccos(cos))))
            plane_inlier.append(float(plane.get("inlier_ratio", 0.0)))

        timing = pred["row"].get("timing_ms") or {}
        if timing:
            timing_ms.append(float(timing.get("total", 0.0)))

        scale_alignment = pred["row"].get("scale_alignment") or {}
        if scale_alignment.get("status") == "ok":
            scale_factors.append(float(scale_alignment.get("scale_factor", 1.0)))

        objects = pred["objects"]
        active_object_count += len(objects)
        continuing_object_count += sum(1 for item in objects if item.get("state") in {"updated", "reactivated", "predicted"})

    detection = _binary_metrics(tp, fp, fn, tn)
    geom_df = pd.DataFrame(geometry_rows)
    if not geom_df.empty:
        geom_df.to_csv(tables_dir / "object_geometry_matches.csv", index=False)

    geometry_metrics = {
        "height": _err_stats(geom_df["height_ae_m"]) if not geom_df.empty else _err_stats(pd.Series(dtype=float)),
        "width": _err_stats(geom_df["width_ae_m"]) if not geom_df.empty else _err_stats(pd.Series(dtype=float)),
        "distance": _err_stats(geom_df["distance_ae_m"]) if not geom_df.empty else _err_stats(pd.Series(dtype=float)),
        "matched_pairs": int(len(geom_df)),
    }
    plane_metrics = {
        "plane_normal_error_deg": {
            "mean": float(np.mean(plane_err_deg)) if plane_err_deg else 0.0,
            "max": float(np.max(plane_err_deg)) if plane_err_deg else 0.0,
        },
        "plane_inlier_ratio": {
            "mean": float(np.mean(plane_inlier)) if plane_inlier else 0.0,
            "min": float(np.min(plane_inlier)) if plane_inlier else 0.0,
        },
    }
    scale_metrics = {
        "scale_factor_stability": {
            "mean": float(np.mean(scale_factors)) if scale_factors else 0.0,
            "std": float(np.std(scale_factors)) if scale_factors else 0.0,
            "count": int(len(scale_factors)),
        }
    }
    tracking_metrics = {
        "tracking_continuity": _safe_div(continuing_object_count, active_object_count),
        "active_object_count": int(active_object_count),
        "continuing_object_count": int(continuing_object_count),
    }
    latency_metrics = {
        "mean_latency_ms": float(np.mean(timing_ms)) if timing_ms else 0.0,
        "max_latency_ms": float(np.max(timing_ms)) if timing_ms else 0.0,
    }
    temporal_metrics = _temporal_measurement_metrics(frame_states)
    label_diagnostics = {
        "07": _label_layer_diagnostics(gt_df, frame_states, "07", tables_dir),
    }

    summary = {
        "detection_metrics": detection,
        "risk_confusion_matrix": risk_conf.tolist(),
        "geometry_metrics": geometry_metrics,
        "plane_metrics": plane_metrics,
        "scale_metrics": scale_metrics,
        "tracking_metrics": tracking_metrics,
        "latency_metrics": latency_metrics,
        "temporal_measurement_metrics": temporal_metrics,
        "label_layer_diagnostics": label_diagnostics,
    }

    plot_confusion_matrix(
        matrix=risk_conf,
        labels=["omega0", "omega1", "omega2"],
        title="Risk Confusion Matrix",
        output_path=plots_dir / "risk_confusion_matrix.png",
    )
    plot_geometry_error_scatter(geom_df, plots_dir / "geometry_error_scatter.png")
    plot_latency_histogram(timing_ms, plots_dir / "latency_histogram.png")

    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary
