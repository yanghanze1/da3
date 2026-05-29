from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


foe_roi = _optional_import("monocular_experiment.foe_roi")


@unittest.skipIf(foe_roi is None, "foe_roi module not ready yet")
class FoERoiTests(unittest.TestCase):
    def test_flow_lines_estimate_expected_foe(self) -> None:
        foe = np.array([100.0, 40.0], dtype=float)
        current_points = np.array(
            [[40.0, 100.0], [60.0, 120.0], [140.0, 100.0], [160.0, 130.0], [90.0, 150.0], [115.0, 160.0]],
            dtype=float,
        )
        vectors = current_points - foe
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None] * 8.0
        previous_points = current_points - vectors

        result = foe_roi.estimate_foe_from_flow_lines(
            previous_points,
            current_points,
            (180, 220, 3),
            {"min_flow_points": 4, "flow_min_length_px": 2.0, "flow_max_length_px": 20.0},
        )

        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["foe"][0], foe[0], delta=1.0)
        self.assertAlmostEqual(result["foe"][1], foe[1], delta=1.0)

    def test_road_mask_edges_build_corridor(self) -> None:
        road_mask = np.zeros((120, 160), dtype=bool)
        for y in range(50, 120):
            left = int(70 - (y - 50) * 0.6)
            right = int(90 + (y - 50) * 0.6)
            road_mask[y, left:right] = True

        edges = foe_roi.estimate_road_edge_lines(
            road_mask,
            {"road_edge_search_bottom_ratio": 0.45, "min_edge_points": 8, "min_road_row_width_px": 8, "min_bottom_width_ratio": 0.1},
        )
        corridor = foe_roi.build_triangle_from_foe_and_edges(
            [80.0, 35.0],
            edges,
            (120, 160, 3),
            {"roi_shape": "road_corridor", "min_corridor_area_ratio": 0.05},
        )

        self.assertEqual(edges["status"], "ok")
        self.assertEqual(corridor["status"], "ok")
        self.assertEqual(len(corridor["polygon"]), 4)
        self.assertLess(corridor["polygon"][0][0], corridor["polygon"][1][0])
        self.assertLess(corridor["polygon"][3][0], corridor["polygon"][2][0])

    def test_corridor_width_scales_can_make_bottom_wider_than_top(self) -> None:
        edges = {
            "left": {"slope": 0.0, "intercept": 20.0},
            "right": {"slope": 0.0, "intercept": 140.0},
        }

        corridor = foe_roi.build_triangle_from_foe_and_edges(
            [80.0, 40.0],
            edges,
            (120, 160, 3),
            {
                "roi_shape": "road_corridor",
                "min_corridor_area_ratio": 0.01,
                "corridor_top_width_scale": 0.62,
                "corridor_bottom_width_scale": 0.74,
            },
        )

        self.assertEqual(corridor["status"], "ok")
        polygon = corridor["polygon"]
        top_width = polygon[1][0] - polygon[0][0]
        bottom_width = polygon[2][0] - polygon[3][0]
        self.assertGreater(bottom_width, top_width)

    def test_triangle_rejects_tiny_area(self) -> None:
        edges = {
            "left": {"slope": 0.0, "intercept": 75.0},
            "right": {"slope": 0.0, "intercept": 85.0},
        }

        result = foe_roi.build_triangle_from_foe_and_edges(
            [80.0, 100.0],
            edges,
            (120, 160, 3),
            {"min_triangle_area_ratio": 0.2},
        )

        self.assertEqual(result["status"], "fallback")

    def test_no_previous_frame_falls_back(self) -> None:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        road_mask = np.ones((80, 100), dtype=bool)

        result = foe_roi.build_foe_road_triangle_roi(None, frame, road_mask, {"min_flow_points": 4})

        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["reason"], "no_previous_frame")


if __name__ == "__main__":
    unittest.main()
