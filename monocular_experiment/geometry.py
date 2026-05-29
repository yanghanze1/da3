from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RelativeDepthScaleResult:
    """论文 Eq.15-20 的尺度对齐结果：在相对深度空间中拟合地平面，求得虚拟相机高度后推导 scale_factor。"""

    scale_factor: float
    plane_normal_rel: list[float]
    plane_offset_rel: float
    h_hat_cam: float
    status: str
    metadata: dict[str, Any]


def _backproject_rel_depth_points(
    depth_map: np.ndarray,
    intrinsics: dict[str, Any],
    mask: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """在相对深度空间中反向投影像素点到相机坐标系三维点（未尺度化）。

    Returns:
        image_points: (N, 2) 像素坐标
        camera_points: (N, 3) 相机坐标，Z=相对深度，X,Y按射线缩放
    """
    h, w = depth_map.shape
    ys = np.arange(0, h, max(1, int(stride)))
    xs = np.arange(0, w, max(1, int(stride)))
    grid_x, grid_y = np.meshgrid(xs, ys)
    image_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
    depths = depth_map[grid_y.ravel(), grid_x.ravel()].astype(np.float64)
    valid = np.isfinite(depths) & (depths > 0.0)
    if mask is not None:
        sampled_mask = mask[grid_y.ravel(), grid_x.ravel()].astype(bool)
        valid &= sampled_mask
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    image_points = image_points[valid]
    depths = depths[valid]
    matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    u, v = image_points[:, 0], image_points[:, 1]
    x = (u - cx) * depths / fx
    y = (v - cy) * depths / fy
    camera_points = np.stack([x, y, depths], axis=1)
    return image_points, camera_points


def _fit_plane_ransac_rel(
    points: np.ndarray,
    iterations: int,
    threshold: float,
    min_inliers: int,
    min_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """在相对深度空间中 RANSAC 拟合地面平面。返回 (normal, offset, inlier_mask) 或 None。"""
    if len(points) < min_points:
        return None
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        ids = rng.choice(len(points), size=3, replace=False)
        sample = points[ids]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        if normal[0] < 0:
            normal = -normal
        offset = -float(np.dot(normal, sample[0]))
        dist = np.abs(np.dot(points, normal) + offset)
        mask = dist <= threshold
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count < min_inliers:
        return None
    # SVD refine on inliers
    inliers = points[best_mask]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[0] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, centroid))
    dist = np.abs(np.dot(points, normal) + offset)
    final_mask = dist <= threshold
    return normal, offset, final_mask


def estimate_scale_factor_from_relative_depth_plane(
    relative_depth_map: np.ndarray,
    road_mask: np.ndarray,
    intrinsics: dict[str, Any],
    camera_height_m: float,
    stride: int = 6,
    min_rel_depth: float = 0.01,
    trim_percentile: float = 15.0,
    min_candidates: int = 20,
    ransac_iterations: int = 80,
    ransac_threshold_rel: float = 0.05,
    ransac_min_inliers: int = 20,
) -> RelativeDepthScaleResult:
    """论文 Eq.15-20：相对深度空间中的尺度对齐。

    1. 在相对深度空间中用 RANSAC 拟合地面平面（Eq.17）
    2. 由平面偏移量得到虚拟相机高度 h_hat_cam（相机原点到平面的正交距离，Eq.18-19）
    3. scale_factor = camera_height_m / h_hat_cam（Eq.20）

    Args:
        relative_depth_map: DA3 输出的相对深度图
        road_mask: PIDNet 输出的二值道路遮罩
        intrinsics: 相机内参
        camera_height_m: 相机安装高度（米），作为尺度基准
    """
    image_pts, cam_pts = _backproject_rel_depth_points(
        relative_depth_map.astype(np.float64), intrinsics, mask=road_mask, stride=stride
    )
    if len(image_pts) == 0:
        return RelativeDepthScaleResult(
            scale_factor=1.0,
            plane_normal_rel=[0.0, 0.0, 1.0],
            plane_offset_rel=0.0,
            h_hat_cam=1.0,
            status="no_road_points",
            metadata={"candidate_count": 0},
        )

    # Filter by minimum relative depth
    valid = cam_pts[:, 2] > float(min_rel_depth)
    cam_pts = cam_pts[valid]
    if len(cam_pts) < ransac_min_inliers:
        return RelativeDepthScaleResult(
            scale_factor=1.0,
            plane_normal_rel=[0.0, 0.0, 1.0],
            plane_offset_rel=0.0,
            h_hat_cam=1.0,
            status="insufficient_points",
            metadata={"candidate_count": len(cam_pts)},
        )

    rng = np.random.default_rng(42)
    result = _fit_plane_ransac_rel(
        cam_pts,
        iterations=ransac_iterations,
        threshold=ransac_threshold_rel,
        min_inliers=ransac_min_inliers,
        min_points=ransac_min_inliers,
        rng=rng,
    )
    if result is None:
        return RelativeDepthScaleResult(
            scale_factor=1.0,
            plane_normal_rel=[0.0, 0.0, 1.0],
            plane_offset_rel=0.0,
            h_hat_cam=1.0,
            status="ransac_failed",
            metadata={"candidate_count": len(cam_pts)},
        )

    normal, offset, inlier_mask = result
    # Camera is at origin in camera coords, so perpendicular distance = |offset| / ||normal||
    h_hat_cam = abs(offset) / math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if h_hat_cam < 1e-6:
        return RelativeDepthScaleResult(
            scale_factor=1.0,
            plane_normal_rel=normal.tolist(),
            plane_offset_rel=offset,
            h_hat_cam=h_hat_cam,
            status="degenerate_plane",
            metadata={"candidate_count": len(cam_pts), "inlier_count": int(np.count_nonzero(inlier_mask))},
        )

    scale_factor = float(camera_height_m) / float(h_hat_cam)
    inlier_count = int(np.count_nonzero(inlier_mask))
    return RelativeDepthScaleResult(
        scale_factor=scale_factor,
        plane_normal_rel=normal.tolist(),
        plane_offset_rel=offset,
        h_hat_cam=float(h_hat_cam),
        status="ok",
        metadata={
            "candidate_count": len(cam_pts),
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_count / len(cam_pts)),
        },
    )


def focal_length_px(intrinsics: dict[str, Any]) -> float:
    """由相機內參矩陣估計等效焦距像素值。"""

    matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    return float((matrix[0, 0] + matrix[1, 1]) / 2.0)


def backproject_points(image_points: np.ndarray, depths_m: np.ndarray, intrinsics: dict[str, Any]) -> np.ndarray:
    """把影像像點與深度反投影為相機座標系中的三維點。"""

    matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    u = image_points[:, 0]
    v = image_points[:, 1]
    z = depths_m
    x = ((u - cx) * z) / fx
    y = ((v - cy) * z) / fy
    return np.stack([x, y, z], axis=1)


def camera_to_world(camera_points: np.ndarray, extrinsics: dict[str, Any]) -> np.ndarray:
    """把相機座標系中的點轉回車體局部世界座標系。"""

    rotation = np.asarray(extrinsics["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(extrinsics["translation_vector"], dtype=np.float64).reshape(3)
    inv_rotation = np.linalg.inv(rotation)
    return (inv_rotation @ (camera_points - translation).T).T


def world_to_camera(world_points: np.ndarray, extrinsics: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(extrinsics["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(extrinsics["translation_vector"], dtype=np.float64).reshape(3)
    return (rotation @ np.asarray(world_points, dtype=np.float64).reshape(-1, 3).T).T + translation


def project_camera_points(camera_points: np.ndarray, intrinsics: dict[str, Any]) -> np.ndarray:
    points = np.asarray(camera_points, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    z = points[:, 2]
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(z) & (z > 0.0)
    projected[valid, 0] = (points[valid, 0] * fx / z[valid]) + cx
    projected[valid, 1] = (points[valid, 1] * fy / z[valid]) + cy
    return projected


def project_world_points(world_points: np.ndarray, intrinsics: dict[str, Any], extrinsics: dict[str, Any]) -> np.ndarray:
    return project_camera_points(world_to_camera(world_points, extrinsics), intrinsics)


def backproject_depth_map(
    depth_map_m: np.ndarray,
    intrinsics: dict[str, Any],
    mask: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """把深度圖反投影為像素點與相機三維點。"""

    h, w = depth_map_m.shape
    ys = np.arange(0, h, max(1, int(stride)))
    xs = np.arange(0, w, max(1, int(stride)))
    grid_x, grid_y = np.meshgrid(xs, ys)
    image_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    depths = depth_map_m[grid_y, grid_x].astype(np.float64).ravel()
    valid = np.isfinite(depths) & (depths > 0.0)
    if mask is not None:
        sampled_mask = mask[grid_y, grid_x].ravel().astype(bool)
        valid &= sampled_mask
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float64)
    image_points = image_points[valid]
    depths = depths[valid]
    camera_points = backproject_points(image_points, depths, intrinsics)
    return image_points, camera_points


def sample_points_from_mask(
    depth_map_m: np.ndarray,
    intrinsics: dict[str, Any],
    extrinsics: dict[str, Any],
    mask: np.ndarray,
    stride: int,
    max_points: int = 0,
) -> dict[str, np.ndarray]:
    """從遮罩區域取樣深度點並轉成世界座標。"""

    image_points, camera_points = backproject_depth_map(depth_map_m, intrinsics, mask=mask, stride=stride)
    if len(image_points) == 0:
        return {
            "image_points": np.empty((0, 2), dtype=np.float32),
            "depths_m": np.empty((0,), dtype=np.float64),
            "camera_points": np.empty((0, 3), dtype=np.float64),
            "world_points": np.empty((0, 3), dtype=np.float64),
        }
    if max_points > 0 and len(image_points) > max_points:
        ids = np.linspace(0, len(image_points) - 1, num=max_points, dtype=int)
        image_points = image_points[ids]
        camera_points = camera_points[ids]
    world_points = camera_to_world(camera_points, extrinsics)
    return {
        "image_points": image_points,
        "depths_m": camera_points[:, 2].copy(),
        "camera_points": camera_points,
        "world_points": world_points,
    }


def estimate_scale_factor_from_road_mask(
    relative_depth_map: np.ndarray,
    road_mask: np.ndarray,
    intrinsics: dict[str, Any],
    extrinsics: dict[str, Any],
    stride: int,
    min_rel_depth: float,
    trim_percentile: float,
    min_candidates: int,
) -> tuple[float, dict[str, Any]]:
    """由路面像素解出單幀尺度因子 s_t。"""

    image_points, camera_points = backproject_depth_map(
        relative_depth_map.astype(np.float64),
        intrinsics,
        mask=road_mask,
        stride=stride,
    )
    if len(image_points) == 0:
        return 1.0, {
            "status": "no_road_points",
            "candidate_count": 0,
            "selected_count": 0,
            "scale_candidates": [],
        }

    camera_points = camera_points[camera_points[:, 2] > float(min_rel_depth)]
    if len(camera_points) == 0:
        return 1.0, {
            "status": "no_valid_relative_depth",
            "candidate_count": 0,
            "selected_count": 0,
            "scale_candidates": [],
        }

    rotation = np.asarray(extrinsics["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(extrinsics["translation_vector"], dtype=np.float64).reshape(3)
    inv_rotation = np.linalg.inv(rotation)

    world_unit = (inv_rotation @ camera_points.T).T
    world_translation = inv_rotation @ translation
    denom = world_unit[:, 0]
    numer = world_translation[0]
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-8)
    scale_candidates = numer / denom[valid]
    scale_candidates = scale_candidates[np.isfinite(scale_candidates) & (scale_candidates > 0.0)]
    candidate_count = int(len(scale_candidates))
    if candidate_count < int(min_candidates):
        return 1.0, {
            "status": "insufficient_scale_candidates",
            "candidate_count": candidate_count,
            "selected_count": candidate_count,
            "scale_candidates": scale_candidates.round(6).tolist(),
        }

    lower = float(np.percentile(scale_candidates, trim_percentile))
    upper = float(np.percentile(scale_candidates, 100.0 - trim_percentile))
    selected = scale_candidates[(scale_candidates >= lower) & (scale_candidates <= upper)]
    if len(selected) < int(min_candidates):
        selected = scale_candidates
    scale_factor = float(np.median(selected))
    return scale_factor, {
        "status": "ok",
        "candidate_count": candidate_count,
        "selected_count": int(len(selected)),
        "trimmed_range": [lower, upper],
        "candidate_mean": float(np.mean(scale_candidates)),
        "candidate_std": float(np.std(scale_candidates)),
        "selected_median": scale_factor,
    }


def points_in_mask(points_2d: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """判斷每個二維點是否落在指定布林遮罩內。"""

    if mask is None or len(points_2d) == 0:
        return np.ones(len(points_2d), dtype=bool)
    h, w = mask.shape
    points = np.rint(points_2d).astype(int)
    inside = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    out = np.zeros(len(points_2d), dtype=bool)
    valid = points[inside]
    out[inside] = mask[valid[:, 1], valid[:, 0]].astype(bool)
    return out


def points_in_bbox(points_2d: np.ndarray, bbox: list[int]) -> np.ndarray:
    """判斷每個二維點是否落在指定外接框內。"""

    if len(points_2d) == 0:
        return np.zeros((0,), dtype=bool)
    x, y, w, h = bbox
    return (
        (points_2d[:, 0] >= x)
        & (points_2d[:, 0] <= x + w)
        & (points_2d[:, 1] >= y)
        & (points_2d[:, 1] <= y + h)
    )


def signed_plane_distance(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """計算三維點到平面的帶符號正交距離。"""

    normal = np.asarray(normal, dtype=np.float64)
    denom = np.linalg.norm(normal)
    if denom == 0.0:
        raise ValueError("Plane normal must be non-zero.")
    return (points @ normal + float(offset)) / denom


def infer_speed_mps(
    timestamp_prev_s: float | None,
    timestamp_curr_s: float,
    speed_value: float | None,
    forward_displacement_m: float | None,
) -> float:
    """由既有速度欄位或位移量推回車速。"""

    if speed_value is not None and not np.isnan(speed_value):
        return float(speed_value)
    if forward_displacement_m is None or np.isnan(forward_displacement_m):
        return 0.0
    if timestamp_prev_s is None:
        return 0.0
    dt = float(timestamp_curr_s) - float(timestamp_prev_s)
    if dt <= 0:
        return 0.0
    return float(forward_displacement_m) / dt


def infer_forward_displacement_m(
    timestamp_prev_s: float | None,
    timestamp_curr_s: float,
    speed_value: float | None,
    forward_displacement_m: float | None,
) -> float:
    """由既有欄位推回本幀前進量。"""

    if forward_displacement_m is not None and not np.isnan(forward_displacement_m):
        return float(forward_displacement_m)
    if speed_value is None or np.isnan(speed_value) or timestamp_prev_s is None:
        return 0.0
    dt = float(timestamp_curr_s) - float(timestamp_prev_s)
    if dt <= 0:
        return 0.0
    return float(speed_value) * dt


def bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    """計算兩個二維外接框的 IoU。"""

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0
