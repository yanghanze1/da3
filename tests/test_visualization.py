from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


segmentation = _optional_import("monocular_experiment.segmentation")
visualization = _optional_import("monocular_experiment.visualization")


@unittest.skipIf(segmentation is None or visualization is None, "visualization modules unavailable")
class VisualizationTests(unittest.TestCase):
    def test_overlay_draws_processing_roi_polygon(self) -> None:
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        frontend = segmentation.FrontendResult(
            road_mask=np.zeros((40, 60), dtype=bool),
            road_probability=np.zeros((40, 60), dtype=np.float32),
            analysis_mask=np.zeros((40, 60), dtype=bool),
            roi_candidates=[],
            backend="test",
            processing_roi={
                "draw_overlay": True,
                "polygon": [[15, 10], [45, 10], [59, 39], [0, 39]],
            },
        )
        with tempfile.TemporaryDirectory(prefix="overlay_polygon_") as tmpdir:
            output_path = Path(tmpdir) / "overlay.png"
            visualization.save_overlay(
                frame_bgr=frame,
                frame_id="frame_000",
                frontend=frontend,
                road_points=np.empty((0, 2), dtype=float),
                plane={"status": "ok", "source": "test"},
                tracked_objects=[],
                output_path=output_path,
            )
            image = cv2.imread(str(output_path))
        self.assertIsNotNone(image)
        red_pixels = (image[:, :, 2] > 180) & (image[:, :, 1] < 80) & (image[:, :, 0] < 80)
        self.assertTrue(np.any(red_pixels[38:, :]))


if __name__ == "__main__":
    unittest.main()
