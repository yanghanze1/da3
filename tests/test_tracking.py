from __future__ import annotations

import importlib
import sys
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


models = _optional_import("monocular_experiment.models")
tracking = _optional_import("monocular_experiment.tracking")


@unittest.skipIf(models is None or tracking is None, "tracking modules not ready yet")
class TrackingTests(unittest.TestCase):
    def _tracker(self) -> "tracking.ObjectTracker":
        return tracking.ObjectTracker(
            {
                "max_anchor_distance_m": 0.5,
                "min_forward_distance_m": 0.05,
                "max_missed_frames": 1,
                "bbox_iou_weight": 0.25,
                "cross_frame_match_weight": 0.35,
                "obstacle_type_mismatch_penalty": 0.08,
                "size_consistency_weight": 0.35,
                "reactivation_window_frames": 3,
                "reactivation_max_anchor_distance_m": 0.35,
            }
        )

    def _candidate(
        self,
        *,
        candidate_id: str,
        roi_id: str,
        obstacle_type: str = "positive",
        bbox: list[int] | None = None,
        anchor: list[float] | None = None,
        height_m: float = 0.08,
        width_m: float = 0.20,
    ) -> "models.CandidateObservation":
        if bbox is None:
            bbox = [100, 100, 20, 20]
        if anchor is None:
            anchor = [0.08, 0.0, 1.2]
        return models.CandidateObservation(
            candidate_id=candidate_id,
            roi_id=roi_id,
            obstacle_type=obstacle_type,
            bbox=bbox,
            centroid_2d=[bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0],
            anchor=anchor,
            height_m=height_m,
            width_m=width_m,
            distance_m=anchor[2],
            z_range_m=[anchor[2], anchor[2] + 0.2],
            point_count=4,
            abnormal_count=4,
            support_points=[anchor],
            metadata={"bbox": bbox, "roi_bbox": bbox},
        )

    def _match(self, prev_candidate_id: str, curr_candidate_id: str, score: float = 0.8) -> "models.CrossFrameMatch":
        return models.CrossFrameMatch(
            prev_candidate_id=prev_candidate_id,
            curr_candidate_id=curr_candidate_id,
            score=score,
            metadata={},
        )

    def test_lifecycle_generation_update_predict_retire(self) -> None:
        tracker = self._tracker()
        first_candidate = self._candidate(candidate_id="cand_000", roi_id="roi_000")
        first = tracker.update(
            "frame_001",
            [first_candidate],
            cross_frame_matches=[],
            forward_displacement_m=0.08,
        )
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].state, "generated")
        self.assertEqual(first_candidate.object_id, first[0].object_id)
        self.assertEqual(first_candidate.metadata["tracking_state"], "generated")

        second_candidate = self._candidate(candidate_id="cand_001", roi_id="roi_000", bbox=[102, 102, 20, 20], anchor=[0.08, 0.0, 1.12])
        second = tracker.update(
            "frame_002",
            [second_candidate],
            cross_frame_matches=[self._match("cand_000", "cand_001")],
            forward_displacement_m=0.08,
        )
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].state, "updated")
        self.assertEqual(second[0].object_id, first[0].object_id)
        self.assertEqual(second_candidate.object_id, first[0].object_id)
        self.assertEqual(second_candidate.metadata["object_id"], first[0].object_id)

        third = tracker.update("frame_003", [], cross_frame_matches=[], forward_displacement_m=0.08)
        self.assertEqual(len(third), 1)
        self.assertEqual(third[0].state, "predicted")

        fourth = tracker.update("frame_004", [], cross_frame_matches=[], forward_displacement_m=0.08)
        retired = [item for item in fourth if item.state == "retired"]
        self.assertEqual(len(retired), 1)

    def test_reactivated_candidate_keeps_original_object_id(self) -> None:
        tracker = self._tracker()
        first = tracker.update(
            "frame_001",
            [self._candidate(candidate_id="cand_000", roi_id="roi_000")],
            cross_frame_matches=[],
            forward_displacement_m=0.08,
        )
        tracker.update("frame_002", [], cross_frame_matches=[], forward_displacement_m=0.08)
        tracker.update("frame_003", [], cross_frame_matches=[], forward_displacement_m=0.08)
        fourth_candidate = self._candidate(candidate_id="cand_200", roi_id="roi_200", bbox=[103, 100, 20, 20], anchor=[0.08, 0.0, 0.95])
        fourth = tracker.update(
            "frame_004",
            [fourth_candidate],
            cross_frame_matches=[],
            forward_displacement_m=0.08,
        )
        reactivated = [item for item in fourth if item.state == "reactivated"]
        self.assertEqual(len(reactivated), 1)
        self.assertEqual(reactivated[0].object_id, first[0].object_id)
        self.assertEqual(fourth_candidate.object_id, first[0].object_id)
        self.assertEqual(fourth_candidate.metadata["tracking_state"], "reactivated")

    def test_tracker_callback_runs_after_object_id_assignment(self) -> None:
        tracker = self._tracker()
        candidate = self._candidate(candidate_id="cand_000", roi_id="roi_000")
        seen_object_ids: list[str | None] = []

        def callback(candidates):
            seen_object_ids.append(candidates[0].object_id)
            candidates[0].metadata["temporal_measurement"] = {"status": "ok", "object_id": candidates[0].object_id}

        tracked = tracker.update(
            "frame_001",
            [candidate],
            cross_frame_matches=[],
            forward_displacement_m=0.08,
            post_assignment_callback=callback,
        )
        self.assertEqual(seen_object_ids, [tracked[0].object_id])
        self.assertEqual(tracked[0].metadata["temporal_measurement"]["object_id"], tracked[0].object_id)

    def test_far_candidate_does_not_hijack_existing_object(self) -> None:
        tracker = self._tracker()
        first = tracker.update(
            "frame_001",
            [self._candidate(candidate_id="cand_000", roi_id="roi_000")],
            cross_frame_matches=[],
            forward_displacement_m=0.08,
        )
        second = tracker.update(
            "frame_002",
            [self._candidate(candidate_id="cand_010", roi_id="roi_010", bbox=[260, 260, 22, 22], anchor=[0.50, 0.4, 2.0])],
            cross_frame_matches=[],
            forward_displacement_m=0.08,
        )
        generated = [item for item in second if item.state == "generated"]
        predicted = [item for item in second if item.state == "predicted"]
        self.assertEqual(len(generated), 1)
        self.assertEqual(len(predicted), 1)
        self.assertNotEqual(generated[0].object_id, first[0].object_id)

    def test_predicted_object_past_camera_is_retired_without_zero_distance_output(self) -> None:
        tracker = self._tracker()
        tracker.update(
            "frame_001",
            [self._candidate(candidate_id="cand_000", roi_id="roi_000", anchor=[0.08, 0.0, 0.12])],
            cross_frame_matches=[],
            forward_displacement_m=0.0,
        )
        predicted = tracker.update("frame_002", [], cross_frame_matches=[], forward_displacement_m=0.2)
        self.assertEqual(len(predicted), 1)
        self.assertEqual(predicted[0].state, "retired")
        self.assertGreater(predicted[0].distance_m, 0.05)
        self.assertEqual(tracker.active_tracks, [])
        self.assertEqual(len(tracker.retired_tracks), 1)

    def test_temporal_metadata_is_preserved_on_tracked_object(self) -> None:
        tracker = self._tracker()
        candidate = self._candidate(candidate_id="cand_000", roi_id="roi_000")
        candidate.metadata["temporal_measurement"] = {"status": "ok", "quality": 0.75, "applied": True}

        tracked = tracker.update(
            "frame_001",
            [candidate],
            cross_frame_matches=[],
            forward_displacement_m=0.08,
        )
        self.assertEqual(tracked[0].metadata["temporal_measurement"]["status"], "ok")
        self.assertAlmostEqual(tracked[0].metadata["temporal_measurement"]["quality"], 0.75)


if __name__ == "__main__":
    unittest.main()
