from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


demo_data = _optional_import("monocular_experiment.demo_data")
evaluation = _optional_import("monocular_experiment.evaluation")
pipeline = _optional_import("monocular_experiment.pipeline")
config_mod = _optional_import("monocular_experiment.config")


@unittest.skipIf(demo_data is None or evaluation is None or pipeline is None or config_mod is None, "smoke modules unavailable")
class SmokeTests(unittest.TestCase):
    def test_demo_generation_pipeline_and_readme_contract(self) -> None:
        workspace = Path(tempfile.mkdtemp(prefix="mono_demo_v3_"))
        try:
            datasets_root = workspace / "datasets"
            calibration_dir = workspace / "calibration"
            out_dir = workspace / "outputs" / "demo_run_v3"

            demo_data.make_demo_sequence(datasets_root, "demo_sequence", calibration_dir)
            self.assertTrue((datasets_root / "demo_sequence" / "frames").exists())
            self.assertTrue((datasets_root / "demo_sequence" / "motion.csv").exists())
            self.assertTrue((datasets_root / "demo_sequence" / "gt_obstacles.csv").exists())
            self.assertTrue((datasets_root / "demo_sequence" / "gt_plane.json").exists())

            config_path = CODE_ROOT / "configs" / "default.yaml"
            config = config_mod.load_config(config_path)
            config["calibration"]["intrinsics_path"] = str(calibration_dir / "intrinsics_demo.yaml")
            config["calibration"]["extrinsics_path"] = str(calibration_dir / "extrinsics_demo.yaml")
            config["depth_model"]["backend"] = "mock"
            config["roi"]["backend"] = "fallback"
            config["roi"]["processing_rect"] = {
                "enabled": True,
                "mode": "trapezoid",
                "normalized_bbox_xywh": [0.05, 0.25, 0.9, 0.7],
                "obstacle_analysis_mask": "processing_roi",
                "restrict_analysis_mask": False,
                "restrict_road_mask": False,
                "trapezoid_bottom_full_width": True,
                "draw_overlay": True,
            }
            config["candidate_generation"]["height_reference"] = "depth_residual_to_plane"
            config["candidate_generation"]["depth_contour_roi"] = {
                "enabled": True,
                "fallback_to_frontend_rois": False,
                "min_area_px": 2,
                "dilate_kernel_px": 1,
                "close_kernel_px": 1,
                "reject_boundary_touching": False,
            }
            config["temporal_measurement"] = {
                "enabled": True,
                "history_size": 12,
                "min_baseline_m": 0.16,
                "max_baseline_m": 1.2,
                "target_baseline_m": 0.24,
                "min_time_gap_s": 0.0,
                "max_time_gap_s": 4.0,
                "max_keyframes_per_candidate": 2,
                "min_support_points": 2,
                "min_history_support_points": 2,
                "min_quality_to_apply": 0.95,
                "apply_to_candidate_metrics": False,
                "max_anchor_distance_m": 1.0,
                "max_centroid_shift_px": 200.0,
                "min_match_score": 0.0,
            }

            pipeline.run_pipeline(config, datasets_root, "demo_sequence", out_dir, control_backend="noop")
            summary = evaluation.evaluate_pipeline(datasets_root / "demo_sequence", out_dir, config)

            frame_states_path = out_dir / "frame_states.jsonl"
            self.assertTrue(frame_states_path.exists())
            frame_states = [json.loads(line) for line in frame_states_path.read_text(encoding="utf-8").splitlines()]
            first_state = frame_states[0]
            self.assertIn("processing_roi", first_state["road_mask_stats"])
            processing_roi = first_state["road_mask_stats"]["processing_roi"]
            self.assertEqual(processing_roi["mode"], "trapezoid")
            self.assertIn("polygon", processing_roi)
            frame_shape = first_state["depth_stats"]["shape"]
            self.assertEqual(processing_roi["polygon"][2], [frame_shape[1] - 1, frame_shape[0] - 1])
            self.assertEqual(processing_roi["polygon"][3], [0, frame_shape[0] - 1])
            self.assertIn("obstacle_analysis_mask_source", first_state["road_mask_stats"])
            self.assertEqual(first_state["road_mask_stats"]["obstacle_analysis_mask_source"], "processing_roi")
            self.assertIn("depth_roi_count", first_state["road_mask_stats"])
            self.assertIn("candidate_roi_source", first_state["road_mask_stats"])
            self.assertIn(first_state["road_mask_stats"]["candidate_roi_source"], {"depth_residual_contour", "none"})
            self.assertIn("temporal_measurements", first_state)
            self.assertIn("temporal_measurement", first_state["timing_ms"])
            self.assertTrue(any("temporal_measurements" in state for state in frame_states))
            candidate_clusters = [item for state in frame_states for item in state["candidate_clusters"]]
            exported_rois = [item for state in frame_states for item in state["roi_candidates"]]
            tracked_ids = {item["object_id"] for state in frame_states for item in state["tracked_objects"]}
            self.assertTrue(all("object_id" in item for item in candidate_clusters))
            self.assertTrue(all("object_id" in item for item in exported_rois))
            self.assertTrue({item["object_id"] for item in candidate_clusters if item.get("object_id")}.issubset(tracked_ids))
            temporal_measurements = [item for state in frame_states for item in state["temporal_measurements"]]
            self.assertTrue(all("object_id" in item for item in temporal_measurements))
            self.assertTrue((out_dir / "pipeline_summary.json").exists())
            self.assertTrue((out_dir / "evaluation_summary.json").exists())
            self.assertIn("detection_metrics", summary)
            self.assertIn("scale_metrics", summary)
            self.assertIn("tracking_metrics", summary)
            self.assertIn("temporal_measurement_metrics", summary)

            readme_text = (CODE_ROOT / "README.md").read_text(encoding="utf-8")
            self.assertIn("run-pipeline", readme_text)
            self.assertIn("evaluate-pipeline", readme_text)
            self.assertIn("depth_model", readme_text)
            self.assertIn("frame_states.jsonl", readme_text)
            self.assertIn("scale_factor_stability", readme_text)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
