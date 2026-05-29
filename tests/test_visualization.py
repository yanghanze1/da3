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
pipeline = _optional_import("monocular_experiment.pipeline")


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

    @unittest.skipIf(pipeline is None, "pipeline module unavailable")
    def test_workflow_figure_candidate_prefers_temporal_applied_gray_box(self) -> None:
        frame = np.full((80, 120, 3), 160, dtype=np.uint8)
        frame[:, 70:] = (0, 0, 255)
        base_temporal = {
            "status": "ok",
            "applied": True,
            "quality": 0.7,
            "method": "support_point_fusion",
            "selected_keyframes": [{"frame_id": "prev"}],
        }
        selected = pipeline._score_workflow_figure_candidate(
            frame_bgr=frame,
            frame_index=4,
            frame_count=5,
            tracked_objects=[
                {
                    "object_id": "obj_red",
                    "candidate_id": "cand_red",
                    "state": "updated",
                    "metadata": {"bbox": [72, 20, 32, 32], "temporal_measurement": base_temporal},
                },
                {
                    "object_id": "obj_gray",
                    "candidate_id": "cand_gray",
                    "state": "updated",
                    "metadata": {"bbox": [16, 20, 32, 32], "temporal_measurement": base_temporal},
                },
                {
                    "object_id": "obj_predicted",
                    "candidate_id": "cand_predicted",
                    "state": "predicted",
                    "metadata": {"bbox": [16, 20, 48, 48], "temporal_measurement": {**base_temporal, "quality": 0.95}},
                },
                {
                    "object_id": "obj_unmatched",
                    "candidate_id": "cand_unmatched",
                    "state": "updated",
                    "metadata": {"bbox": [16, 20, 48, 48], "temporal_measurement": {**base_temporal, "status": "no_candidate_match"}},
                },
            ],
            config={
                "workflow_figure_selection_mode": "auto_temporal_boxed",
                "workflow_figure_final_window": 5,
                "workflow_figure_min_temporal_quality": 0.6,
                "workflow_figure_require_temporal_applied": True,
                "workflow_figure_min_box_area_px": 600,
                "workflow_figure_prefer_gray_object": True,
            },
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["object_id"], "obj_gray")
        self.assertTrue(selected["temporal_applied"])
        self.assertEqual(selected["temporal_status"], "ok")


if __name__ == "__main__":
    unittest.main()
