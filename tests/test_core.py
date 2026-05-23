from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


geometry = _optional_import("monocular_experiment.geometry")
risk = _optional_import("monocular_experiment.risk")
models = _optional_import("monocular_experiment.models")
pipeline = _optional_import("monocular_experiment.pipeline")
evaluation = _optional_import("monocular_experiment.evaluation")


@unittest.skipIf(geometry is None or risk is None or models is None or pipeline is None or evaluation is None, "core modules not ready yet")
class CoreMathTests(unittest.TestCase):
    def test_backproject_points(self) -> None:
        intrinsics = {"camera_matrix": [[200.0, 0.0, 100.0], [0.0, 200.0, 50.0], [0.0, 0.0, 1.0]]}
        image_points = np.array([[100.0, 50.0], [120.0, 70.0]], dtype=float)
        depths = np.array([2.0, 4.0], dtype=float)
        camera_points = geometry.backproject_points(image_points, depths, intrinsics)
        np.testing.assert_allclose(camera_points[0], np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(camera_points[1], np.array([0.4, 0.4, 4.0]))

    def test_scale_alignment_recovers_expected_scale(self) -> None:
        intrinsics = {"camera_matrix": [[200.0, 0.0, 1.0], [0.0, 200.0, 1.0], [0.0, 0.0, 1.0]]}
        extrinsics = {
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translation_vector": [1.0, 0.0, 0.0],
        }
        relative_depth = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        road_mask = np.ones((3, 3), dtype=bool)
        scale, meta = geometry.estimate_scale_factor_from_road_mask(
            relative_depth_map=relative_depth,
            road_mask=road_mask,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            stride=1,
            min_rel_depth=0.01,
            trim_percentile=0.0,
            min_candidates=1,
        )
        self.assertEqual(meta["status"], "ok")
        self.assertAlmostEqual(scale, 200.0, places=4)

    def test_safe_braking_distance(self) -> None:
        d = risk.safe_braking_distance_m(speed_mps=1.0, reaction_time_s=0.5, max_decel_mps2=1.0)
        self.assertAlmostEqual(d, 1.0)

    def _tracked_object(
        self,
        *,
        object_id: str = "obj_000",
        state: str = "updated",
        hit_count: int = 3,
        support_count: int = 6,
    ) -> "models.TrackedObject":
        support_points = [[0.0, 0.0, 1.0] for _ in range(support_count)]
        return models.TrackedObject(
            object_id=object_id,
            roi_id="roi_000",
            candidate_id="cand_000",
            obstacle_type="positive",
            anchor=[0.08, 0.0, 1.0],
            height_m=0.08,
            width_m=0.15,
            distance_m=1.0,
            state=state,
            miss_count=0,
            hit_count=hit_count,
            last_seen_frame="frame_001",
            support_points=support_points,
            metadata={"roi_source": "depth_residual_contour"},
        )

    def test_risk_emitter_filters_predicted_objects_and_low_support(self) -> None:
        emitter = pipeline._RiskEventEmitter(
            {
                "emit_min_consecutive_hits": 3,
                "emit_predicted_objects": False,
                "emit_min_support_points": 6,
            }
        )
        risk_event = {"object_id": "obj_000", "decision": "danger", "metadata": {}}

        predicted = self._tracked_object(state="predicted", hit_count=5, support_count=8)
        self.assertEqual(emitter.filter_events([predicted], {"obj_000": risk_event}), [])

        weak = self._tracked_object(state="updated", hit_count=5, support_count=3)
        self.assertEqual(emitter.filter_events([weak], {"obj_000": risk_event}), [])

    def test_risk_emitter_cooldown_allows_decision_upgrade(self) -> None:
        emitter = pipeline._RiskEventEmitter(
            {
                "emit_min_consecutive_hits": 3,
                "emit_predicted_objects": False,
                "emit_min_support_points": 0,
                "emit_cooldown_frames": 5,
            }
        )
        obj = self._tracked_object(state="updated", hit_count=3)
        warning = {"object_id": "obj_000", "decision": "warning", "metadata": {}}
        danger = {"object_id": "obj_000", "decision": "danger", "metadata": {}}

        self.assertEqual(len(emitter.filter_events([obj], {"obj_000": warning})), 1)
        self.assertEqual(emitter.filter_events([obj], {"obj_000": warning}), [])
        upgraded = emitter.filter_events([obj], {"obj_000": danger})
        self.assertEqual(len(upgraded), 1)
        self.assertEqual(upgraded[0]["decision"], "danger")

    def test_risk_assessment_uses_clear_path_and_braking(self) -> None:
        road_points = np.array(
            [
                [0.0, -0.6, 1.0],
                [0.0, 0.6, 1.0],
                [0.0, -0.55, 1.1],
                [0.0, 0.55, 1.1],
            ],
            dtype=float,
        )
        tracked = models.TrackedObject(
            object_id="obj_000",
            roi_id="roi_000",
            candidate_id="cand_000",
            obstacle_type="positive",
            anchor=[0.08, 0.0, 1.0],
            height_m=0.08,
            width_m=0.15,
            distance_m=1.0,
            state="updated",
            miss_count=0,
            hit_count=1,
            last_seen_frame="frame_001",
            support_points=[],
        )
        assessments = risk.assess_tracked_objects(
            tracked_objects=[tracked],
            road_world_points=road_points,
            speed_mps=1.0,
            config={
                "positive_safe_height_m": 0.03,
                "positive_limit_height_m": 0.10,
                "negative_safe_height_m": 0.03,
                "negative_limit_height_m": 0.06,
                "vehicle_width_m": 0.65,
                "side_clearance_m": 0.10,
                "reaction_time_s": 0.5,
                "max_decel_mps2": 1.0,
                "warning_distance_scale": 1.5,
                "z_window_m": 0.2,
                "min_free_space_points": 2,
                "default_clear_path_width_m": 1.2,
            },
        )
        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].risk_weight, "omega1")
        self.assertEqual(assessments[0].decision, "danger")

    def test_label_layer_diagnostics_handles_zero_padded_labels(self) -> None:
        gt_df = pd.DataFrame(
            [
                {
                    "frame_id": "frame_001",
                    "object_id": "gt_07_0",
                    "label": 7,
                    "bbox_x": 10,
                    "bbox_y": 10,
                    "bbox_w": 10,
                    "bbox_h": 10,
                }
            ]
        )
        frame_states = [
            {
                "frame_id": "frame_001",
                "road_mask_stats": {"selected_roi_diagnostics": [{"roi_id": "roi_001", "bbox": [10, 10, 10, 10]}]},
                "candidate_clusters": [],
                "tracked_objects": [],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = evaluation._label_layer_diagnostics(gt_df, frame_states, "07", Path(tmp_dir))
        self.assertEqual(summary["total_gt_frames"], 1)
        self.assertEqual(summary["hit_counts"]["selected_roi"], 1)

    def test_risk_assessment_skips_invalid_geometry(self) -> None:
        tracked = models.TrackedObject(
            object_id="obj_000",
            roi_id="roi_000",
            candidate_id="cand_000",
            obstacle_type="positive",
            anchor=[0.08, 0.0, 1.0],
            height_m=2.0,
            width_m=0.15,
            distance_m=1.0,
            state="updated",
            miss_count=0,
            hit_count=1,
            last_seen_frame="frame_001",
            support_points=[],
        )
        assessments = risk.assess_tracked_objects(
            tracked_objects=[tracked],
            road_world_points=np.empty((0, 3), dtype=float),
            speed_mps=0.0,
            config={
                "positive_safe_height_m": 0.03,
                "positive_limit_height_m": 0.10,
                "negative_safe_height_m": 0.03,
                "negative_limit_height_m": 0.06,
                "vehicle_width_m": 0.65,
                "side_clearance_m": 0.10,
                "reaction_time_s": 0.5,
                "max_decel_mps2": 1.0,
                "warning_distance_scale": 1.5,
                "z_window_m": 0.2,
                "min_free_space_points": 2,
                "default_clear_path_width_m": 1.2,
                "max_valid_height_m": 1.5,
            },
        )
        self.assertEqual(assessments, [])


if __name__ == "__main__":
    unittest.main()
