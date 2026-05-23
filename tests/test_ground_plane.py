from __future__ import annotations

# 匯入 importlib，進行可選模組載入。
import importlib
# 匯入 sys，補上模組搜尋路徑。
import sys
# 匯入 unittest，建立測試。
import unittest
# 匯入 Path，定位專案根目錄。
from pathlib import Path

# 匯入 NumPy，建立測試點集。
import numpy as np

# 取得 code 根目錄。
CODE_ROOT = Path(__file__).resolve().parents[1]
# 若根目錄未在 sys.path 中，就加入。
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _optional_import(module_name: str):
    """嘗試匯入模組，失敗時回傳 None。"""

    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


# 嘗試匯入地平面模組。
ground_plane = _optional_import("monocular_experiment.ground_plane")


@unittest.skipIf(ground_plane is None, "ground_plane module not ready yet")
class GroundPlaneTests(unittest.TestCase):
    """驗證地平面擬合相關邏輯。"""

    def test_plane_fit_uses_only_road_points(self) -> None:
        """確認足夠路面點可成功擬合平面。"""

        rng = np.random.default_rng(13)
        z = np.linspace(1.0, 3.0, 40)
        y = np.linspace(-0.4, 0.4, 40)
        x = rng.normal(0.0, 0.01, size=40)
        road_points = np.stack([x, y, z], axis=1)

        plane = ground_plane.estimate_ground_plane(
            road_points,
            {
                "min_points": 12,
                "ransac_iterations": 120,
                "ransac_distance_threshold_m": 0.03,
                "ransac_min_inliers": 12,
            },
        )
        self.assertEqual(plane.status, "ok")
        self.assertGreater(plane.inlier_ratio, 0.70)

    def test_insufficient_road_points_returns_invalid(self) -> None:
        """確認路面點不足時會回報 invalid。"""

        plane = ground_plane.estimate_ground_plane(
            np.array([[0.0, 0.0, 1.0], [0.0, 0.1, 1.1]], dtype=float),
            {
                "min_points": 12,
                "ransac_iterations": 120,
                "ransac_distance_threshold_m": 0.03,
                "ransac_min_inliers": 12,
            },
        )
        self.assertEqual(plane.status, "invalid")


# 直接執行本檔時，啟動單元測試。
if __name__ == "__main__":
    unittest.main()
