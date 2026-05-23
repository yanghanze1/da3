from __future__ import annotations

# 匯入 Any，讓設定字典可彈性接收多型別值。
from typing import Any

# 匯入 NumPy，供平面擬合與距離計算使用。
import numpy as np

# 匯入帶符號平面距離函式。
from .geometry import signed_plane_distance
# 匯入地平面資料型別。
from .models import PlaneEstimate


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    """對一組三維點做 SVD 平面擬合。"""

    # 先取得點雲質心。
    centroid = points.mean(axis=0)
    # 對去中心化點雲做奇異值分解。
    _, _, vh = np.linalg.svd(points - centroid)
    # 最小奇異值對應的右奇異向量視為法向量。
    normal = vh[-1]
    # 強制讓法向量的 X 分量朝上為正，維持方向一致性。
    if normal[0] < 0:
        normal = -normal
    # 將法向量正規化為單位向量。
    normal = normal / np.linalg.norm(normal)
    # 依點法式求得平面偏移量。
    offset = -float(np.dot(normal, centroid))
    # 回傳法向量與偏移量。
    return normal, offset


def fit_ransac_plane(points: np.ndarray, config: dict[str, Any]) -> PlaneEstimate | None:
    """使用 RANSAC 從路面點中擬合基準地平面。"""

    # 讀取最少點數門檻。
    min_points = int(config["min_points"])
    # 點數不足時直接失敗。
    if len(points) < min_points:
        return None

    # 使用固定種子，讓測試結果可重現。
    rng = np.random.default_rng(42)
    # 讀取 RANSAC 迭代次數。
    iterations = int(config["ransac_iterations"])
    # 讀取內點距離門檻。
    threshold = float(config["ransac_distance_threshold_m"])
    # 讀取最少內點數要求。
    min_inliers = int(config["ransac_min_inliers"])

    # 初始化最佳內點遮罩。
    best_mask: np.ndarray | None = None
    # 初始化最佳內點數。
    best_count = 0
    # 重複進行隨機抽樣。
    for _ in range(iterations):
        # 每次隨機取三點定義候選平面。
        ids = rng.choice(len(points), size=3, replace=False)
        # 取出這三個樣本點。
        sample = points[ids]
        # 計算第一條邊向量。
        vec1 = sample[1] - sample[0]
        # 計算第二條邊向量。
        vec2 = sample[2] - sample[0]
        # 以叉積取得候選平面法向量。
        normal = np.cross(vec1, vec2)
        # 計算法向量長度。
        norm = np.linalg.norm(normal)
        # 若三點幾乎共線，該平面無效。
        if norm < 1e-8:
            continue
        # 正規化法向量。
        normal = normal / norm
        # 強制法向量方向一致。
        if normal[0] < 0:
            normal = -normal
        # 用其中一點求偏移量。
        offset = -float(np.dot(normal, sample[0]))
        # 計算所有點到候選平面的距離。
        dist = np.abs(signed_plane_distance(points, normal, offset))
        # 門檻內視為內點。
        mask = dist <= threshold
        # 統計內點數量。
        count = int(np.count_nonzero(mask))
        # 若本輪更好，就更新最佳結果。
        if count > best_count:
            best_count = count
            best_mask = mask

    # 若找不到足夠好的平面，直接失敗。
    if best_mask is None or best_count < min_inliers:
        return None

    # 對最佳內點再做一次 SVD 精修。
    normal, offset = _fit_plane_svd(points[best_mask])
    # 重新計算所有點對精修平面的距離。
    distances = np.abs(signed_plane_distance(points, normal, offset))
    # 距離小於門檻者視為最終內點。
    inlier_mask = distances <= threshold
    # 封裝成標準地平面資料結構。
    return PlaneEstimate(
        normal=normal.tolist(),
        offset=float(offset),
        inlier_ratio=float(np.mean(inlier_mask)),
        support_count=int(np.count_nonzero(inlier_mask)),
        source="road_mask_ransac",
        status="ok",
    )


def invalid_plane(status: str = "invalid") -> PlaneEstimate:
    """建立一個明確標記失敗狀態的預設平面。"""

    # 回傳方向固定但狀態非 ok 的平面結果，方便下游判斷。
    return PlaneEstimate(
        normal=[1.0, 0.0, 0.0],
        offset=0.0,
        inlier_ratio=0.0,
        support_count=0,
        source="none",
        status=status,
    )


def estimate_ground_plane(road_world_points: np.ndarray, config: dict[str, Any]) -> PlaneEstimate:
    """由路面稀疏三維點估計當前影格的地平面。"""

    # 若路面支持點不足，直接回報 invalid。
    if len(road_world_points) < int(config["min_points"]):
        return invalid_plane("invalid")
    # 嘗試用 RANSAC 擬合平面。
    plane = fit_ransac_plane(road_world_points, config)
    # 若 RANSAC 失敗，也回傳 invalid。
    if plane is None:
        return invalid_plane("invalid")
    # 成功時回傳平面模型。
    return plane
