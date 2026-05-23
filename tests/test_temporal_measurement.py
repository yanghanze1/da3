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


models = _optional_import("monocular_experiment.models")
temporal = _optional_import("monocular_experiment.temporal_measurement")


@unittest.skipIf(models is None or temporal is None, "temporal measurement modules not ready yet")
class TemporalMeasurementTests(unittest.TestCase):
    def _candidate(
        self,
        *,
        candidate_id: str = "cand_000",
        anchor_z: float = 1.0,
        object_id: str | None = None,
        bbox: list[int] | None = None,
        support_points: list[list[float]] | None = None,
    ):
        if support_points is None:
            support_points = [
                [0.00, -0.10, anchor_z],
                [0.04, -0.05, anchor_z + 0.02],
                [0.08, 0.00, anchor_z + 0.04],
                [0.04, 0.05, anchor_z + 0.06],
                [0.00, 0.10, anchor_z + 0.08],
                [0.08, 0.12, anchor_z + 0.10],
            ]
        if bbox is None:
            bbox = [100, 100, 30, 30]
        metadata = {"bbox": bbox, "roi_bbox": bbox}
        if object_id:
            metadata["object_id"] = object_id
        return models.CandidateObservation(
            candidate_id=candidate_id,
            roi_id="roi_000",
            obstacle_type="positive",
            bbox=bbox,
            centroid_2d=[bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0],
            anchor=[0.04, 0.0, anchor_z],
            height_m=0.08,
            width_m=0.20,
            distance_m=anchor_z,
            z_range_m=[anchor_z, anchor_z + 0.1],
            point_count=len(support_points),
            abnormal_count=len(support_points),
            support_points=support_points,
            metadata=metadata,
            object_id=object_id,
        )

    def _snapshot(self, frame_id: str, index: int, cumulative: float, timestamp: float, candidate=None):
        return temporal.TemporalFrameSnapshot(
            frame_id=frame_id,
            frame_index=index,
            timestamp_s=timestamp,
            cumulative_forward_m=cumulative,
            candidates=[candidate or self._candidate(candidate_id=f"cand_{index:03d}")],
        )

    def _cfg(self, **overrides):
        cfg = {
            "enabled": True,
            "history_size": 30,
            "min_forward_distance_m": 0.05,
            "min_baseline_m": 0.20,
            "max_baseline_m": 1.20,
            "target_baseline_m": 0.30,
            "min_time_gap_s": 0.0,
            "max_time_gap_s": 4.0,
            "max_keyframes_per_candidate": 3,
            "min_support_points": 4,
            "min_history_support_points": 4,
            "min_quality_to_apply": 0.30,
            "apply_to_candidate_metrics": True,
            "max_anchor_distance_m": 0.50,
            "max_centroid_shift_px": 120.0,
            "min_match_score": 0.05,
            "robust_percentile_low": 0.0,
            "robust_percentile_high": 100.0,
        }
        cfg.update(overrides)
        return cfg

    def test_selector_skips_adjacent_frame_when_baseline_too_small(self) -> None:
        history = [
            self._snapshot("frame_000", 0, 0.00, 0.0),
            self._snapshot("frame_001", 1, 0.08, 0.2),
            self._snapshot("frame_002", 2, 0.16, 0.4),
        ]
        current = self._snapshot("frame_003", 3, 0.24, 0.6)

        selections = temporal.select_keyframes(current, history, self._cfg())
        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].snapshot.frame_id, "frame_000")
        self.assertGreaterEqual(selections[0].baseline_m, 0.20)

    def test_selector_rejects_baseline_too_large(self) -> None:
        history = [self._snapshot("frame_000", 0, 0.0, 0.0)]
        current = self._snapshot("frame_010", 10, 2.0, 2.0)

        selections = temporal.select_keyframes(current, history, self._cfg(max_baseline_m=1.0))
        self.assertEqual(selections, [])

    def test_time_gap_filtering(self) -> None:
        history = [self._snapshot("frame_000", 0, 0.0, 0.0)]
        current = self._snapshot("frame_003", 3, 0.3, 10.0)

        selections = temporal.select_keyframes(current, history, self._cfg(max_time_gap_s=1.0))
        self.assertEqual(selections, [])

    def test_history_points_transform_subtracts_baseline_from_z(self) -> None:
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.5]], dtype=float)
        transformed = temporal.transform_history_points_to_current(points, 0.25)
        self.assertTrue(np.allclose(transformed[:, 2], [0.75, 1.25]))

    def test_temporal_measurement_updates_candidate_when_quality_is_high(self) -> None:
        manager = temporal.TemporalMeasurementManager(self._cfg())
        history_candidate = self._candidate(candidate_id="hist", anchor_z=1.24)
        manager.add_snapshot(frame_bgr=None, frame_id="frame_000", frame_index=0, timestamp_s=0.0, cumulative_forward_m=0.0, candidates=[history_candidate])
        current = self._candidate(candidate_id="curr", anchor_z=1.0)

        measurements = manager.measure_candidates(
            frame_bgr=None,
            frame_id="frame_003",
            frame_index=3,
            timestamp_s=0.6,
            cumulative_forward_m=0.24,
            candidates=[current],
        )
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0]["status"], "ok")
        self.assertTrue(measurements[0]["applied"])
        self.assertIn("temporal_measurement", current.metadata)
        self.assertGreater(current.width_m, 0.0)

    def test_temporal_measurement_keeps_single_frame_metrics_when_quality_is_low(self) -> None:
        manager = temporal.TemporalMeasurementManager(self._cfg(min_quality_to_apply=0.99))
        manager.add_snapshot(frame_bgr=None, frame_id="frame_000", frame_index=0, timestamp_s=0.0, cumulative_forward_m=0.0, candidates=[self._candidate(candidate_id="hist", anchor_z=1.24)])
        current = self._candidate(candidate_id="curr", anchor_z=1.0)
        original_height = current.height_m

        measurements = manager.measure_candidates(
            frame_bgr=None,
            frame_id="frame_003",
            frame_index=3,
            timestamp_s=0.6,
            cumulative_forward_m=0.24,
            candidates=[current],
        )
        self.assertEqual(measurements[0]["status"], "ok")
        self.assertFalse(measurements[0]["applied"])
        self.assertEqual(current.height_m, original_height)

    def test_temporal_fusion_rejects_invalid_geometry(self) -> None:
        current = self._candidate(anchor_z=1.0)
        history = self._candidate(anchor_z=1.24)
        matched_history = [
            {
                "candidate": history,
                "baseline_m": 0.24,
                "match_score": 1.0,
                "frame_id": "frame_000",
                "frame_index": 0,
                "time_gap_s": 0.6,
                "selector_score": 1.0,
                "match_method": "object_id",
                "object_id_match": True,
                "projection_used": False,
            }
        ]

        measurement = temporal.fuse_candidate_measurement(current, matched_history, self._cfg(max_valid_height_m=0.05))
        self.assertEqual(measurement["status"], "invalid_geometry")
        self.assertEqual(measurement["reject_reasons"], ["invalid_geometry"])

    def test_temporal_matching_prefers_same_object_id(self) -> None:
        history_same_id = self._candidate(candidate_id="hist_same", anchor_z=1.24, object_id="obj_001", bbox=[180, 180, 30, 30])
        history_geometry = self._candidate(candidate_id="hist_geom", anchor_z=1.24, object_id="obj_999")
        current = self._candidate(candidate_id="curr", anchor_z=1.0, object_id="obj_001")

        match = temporal.match_history_candidate(current, [history_geometry, history_same_id], 0.24, self._cfg())
        self.assertIsNotNone(match)
        self.assertEqual(match["candidate"].candidate_id, "hist_same")
        self.assertEqual(match["match_method"], "object_id")
        self.assertTrue(match["object_id_match"])

    def test_temporal_matching_falls_back_to_geometry_without_object_id(self) -> None:
        current = self._candidate(candidate_id="curr", anchor_z=1.0)
        history = self._candidate(candidate_id="hist", anchor_z=1.24)

        match = temporal.match_history_candidate(current, [history], 0.24, self._cfg())
        self.assertIsNotNone(match)
        self.assertEqual(match["match_method"], "geometry")

    def test_temporal_matching_rejects_known_different_object_id(self) -> None:
        current = self._candidate(candidate_id="curr", anchor_z=1.0, object_id="obj_001")
        history = self._candidate(candidate_id="hist", anchor_z=1.24, object_id="obj_002")

        match = temporal.match_history_candidate(current, [history], 0.24, self._cfg())
        self.assertIsNone(match)

    def test_temporal_matching_allows_known_id_geometry_fallback_when_enabled(self) -> None:
        current = self._candidate(candidate_id="curr", anchor_z=1.0, object_id="obj_001")
        history = self._candidate(candidate_id="hist", anchor_z=1.24, object_id="obj_002")

        match = temporal.match_history_candidate(
            current,
            [history],
            0.24,
            self._cfg(allow_known_object_id_geometry_fallback=True),
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["match_method"], "geometry")
        self.assertFalse(match["object_id_match"])

    def test_same_object_id_match_still_obeys_geometry_gates(self) -> None:
        current = self._candidate(candidate_id="curr", anchor_z=1.0, object_id="obj_001")
        history = self._candidate(candidate_id="hist", anchor_z=5.0, object_id="obj_001")

        match = temporal.match_history_candidate(current, [history], 0.24, self._cfg(max_anchor_distance_m=0.5))
        self.assertIsNone(match)

    def test_temporal_fusion_rejects_invalid_forward_points(self) -> None:
        current = self._candidate(
            candidate_id="curr",
            anchor_z=0.01,
            support_points=[[0.0, 0.0, 0.01], [0.0, 0.1, 0.02], [0.1, 0.0, 0.03], [0.1, 0.1, 0.04]],
        )
        measurement = temporal.fuse_candidate_measurement(current, [], self._cfg(min_support_points=2, min_history_support_points=1))
        self.assertEqual(measurement["status"], "insufficient_current_support")

    def test_projection_compensation_can_satisfy_large_centroid_shift_gate(self) -> None:
        intrinsics = {"camera_matrix": [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]]}
        extrinsics = {
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translation_vector": [0.0, 0.0, 0.0],
        }
        history = self._candidate(
            candidate_id="hist",
            anchor_z=1.24,
            bbox=[10, 10, 10, 10],
            support_points=[[0.0, 0.0, 1.24], [0.1, 0.0, 1.24], [0.0, 0.1, 1.24], [0.1, 0.1, 1.24]],
        )
        current = self._candidate(
            candidate_id="curr",
            anchor_z=1.0,
            bbox=[100, 100, 11, 11],
            support_points=[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]],
        )
        match = temporal.match_history_candidate(
            current,
            [history],
            0.24,
            self._cfg(max_centroid_shift_px=20.0, max_anchor_distance_m=0.5),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
        self.assertIsNotNone(match)
        self.assertTrue(match["projection_used"])

    def test_candidate_metadata_contains_selected_non_adjacent_keyframe(self) -> None:
        manager = temporal.TemporalMeasurementManager(self._cfg())
        manager.add_snapshot(frame_bgr=None, frame_id="frame_000", frame_index=0, timestamp_s=0.0, cumulative_forward_m=0.0, candidates=[self._candidate(candidate_id="hist", anchor_z=1.24, object_id="obj_001")])
        manager.add_snapshot(frame_bgr=None, frame_id="frame_002", frame_index=2, timestamp_s=0.4, cumulative_forward_m=0.16, candidates=[self._candidate(candidate_id="near", anchor_z=1.08, object_id="obj_001")])
        current = self._candidate(candidate_id="curr", anchor_z=1.0, object_id="obj_001")

        manager.measure_candidates(
            frame_bgr=None,
            frame_id="frame_003",
            frame_index=3,
            timestamp_s=0.6,
            cumulative_forward_m=0.24,
            candidates=[current],
        )
        measurement = current.metadata["temporal_measurement"]
        self.assertEqual(measurement["object_id"], "obj_001")
        self.assertEqual(measurement["selected_keyframes"][0]["frame_id"], "frame_000")
        self.assertEqual(measurement["selected_keyframes"][0]["object_id"], "obj_001")
        self.assertEqual(measurement["selected_keyframes"][0]["match_method"], "object_id")
        self.assertGreaterEqual(measurement["selected_keyframes"][0]["baseline_m"], 0.20)


if __name__ == "__main__":
    unittest.main()
