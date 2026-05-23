from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import bbox_iou
from .models import CandidateObservation, CrossFrameMatch, TrackedObject


@dataclass
class _TrackMemory:
    object_id: str
    roi_id: str
    candidate_id: str
    obstacle_type: str
    anchor: np.ndarray
    bbox: list[int]
    height_m: float
    width_m: float
    distance_m: float
    support_points: list[list[float]]
    miss_count: int
    last_seen_frame: str
    state: str
    metadata: dict[str, Any]
    hit_count: int
    age: int
    last_match_score: float | None
    recently_retired_frame: str | None
    retired_tick: int | None


class ObjectTracker:
    """將 cluster 級候選轉成具生命週期的時序障礙物物件。"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.active_tracks: list[_TrackMemory] = []
        self.retired_tracks: list[_TrackMemory] = []
        self._next_object_id = 0
        self._frame_tick = 0

    def _new_object_id(self) -> str:
        object_id = f"obj_{self._next_object_id:03d}"
        self._next_object_id += 1
        return object_id

    def _predict_anchor(self, track: _TrackMemory, forward_displacement_m: float) -> np.ndarray | None:
        predicted = track.anchor.copy()
        predicted[2] = predicted[2] - float(forward_displacement_m)
        min_forward_distance = float(self.config.get("min_forward_distance_m", 0.0))
        if not np.isfinite(predicted[2]) or predicted[2] <= min_forward_distance:
            return None
        return predicted

    def _cross_frame_bonus(
        self,
        track: _TrackMemory,
        candidate: CandidateObservation,
        matches: dict[tuple[str, str], CrossFrameMatch],
    ) -> float:
        link = matches.get((track.candidate_id, candidate.candidate_id))
        if link is None:
            return 0.0
        return float(link.score)

    def _matching_metrics(
        self,
        track: _TrackMemory,
        candidate: CandidateObservation,
        predicted_anchor: np.ndarray,
        matches: dict[tuple[str, str], CrossFrameMatch],
        max_anchor_distance: float,
    ) -> dict[str, float] | None:
        if predicted_anchor is None:
            return None
        candidate_anchor = np.asarray(candidate.anchor, dtype=float)
        anchor_dist = float(np.linalg.norm(predicted_anchor - candidate_anchor))
        if anchor_dist > max_anchor_distance:
            return None
        iou = bbox_iou(track.bbox, candidate.bbox)
        type_penalty = (
            float(self.config.get("obstacle_type_mismatch_penalty", 0.08))
            if track.obstacle_type != candidate.obstacle_type
            else 0.0
        )
        size_penalty = float(self.config.get("size_consistency_weight", 0.35)) * (
            abs(float(track.height_m) - float(candidate.height_m)) + abs(float(track.width_m) - float(candidate.width_m))
        )
        continuity_bonus = float(self.config.get("cross_frame_match_weight", 0.35)) * self._cross_frame_bonus(
            track,
            candidate,
            matches,
        )
        score = (
            anchor_dist
            - float(self.config.get("bbox_iou_weight", 0.25)) * iou
            - continuity_bonus
            + type_penalty
            + size_penalty
        )
        return {
            "score": float(score),
            "iou": float(iou),
            "anchor_dist": float(anchor_dist),
            "continuity_bonus": float(continuity_bonus),
        }

    def _match_track_pool(
        self,
        tracks: list[_TrackMemory],
        candidates: list[CandidateObservation],
        predicted_anchors: dict[str, np.ndarray | None],
        candidate_indices: list[int],
        matches: dict[tuple[str, str], CrossFrameMatch],
        max_anchor_distance: float,
    ) -> list[dict[str, Any]]:
        candidate_pairs: list[dict[str, Any]] = []
        for t_idx, track in enumerate(tracks):
            for c_idx in candidate_indices:
                metrics = self._matching_metrics(
                    track,
                    candidates[c_idx],
                    predicted_anchors[track.object_id],
                    matches,
                    max_anchor_distance,
                )
                if metrics is None:
                    continue
                candidate_pairs.append(
                    {
                        "t_idx": t_idx,
                        "c_idx": c_idx,
                        "score": float(metrics["score"]),
                        "miss_count": int(track.miss_count),
                        "hit_count": int(track.hit_count),
                        "metrics": metrics,
                    }
                )
        candidate_pairs.sort(
            key=lambda item: (
                float(item["score"]),
                int(item["miss_count"]),
                -int(item["hit_count"]),
            )
        )

        matched_tracks: set[int] = set()
        matched_candidates: set[int] = set()
        results: list[dict[str, Any]] = []
        for item in candidate_pairs:
            t_idx = int(item["t_idx"])
            c_idx = int(item["c_idx"])
            if t_idx in matched_tracks or c_idx in matched_candidates:
                continue
            matched_tracks.add(t_idx)
            matched_candidates.add(c_idx)
            results.append(item)
        return results

    def _apply_debug_metadata(
        self,
        track: _TrackMemory,
        *,
        match_mode: str,
        match_score: float | None,
        matched_iou: float | None,
        continuity_bonus: float | None,
        source_track_miss_count: int,
    ) -> None:
        updated = dict(track.metadata)
        updated["match_mode"] = match_mode
        updated["match_score"] = None if match_score is None else float(match_score)
        updated["matched_iou"] = None if matched_iou is None else float(matched_iou)
        updated["continuity_bonus"] = None if continuity_bonus is None else float(continuity_bonus)
        updated["source_track_miss_count"] = int(source_track_miss_count)
        track.metadata = updated

    def _annotate_candidate(self, candidate: CandidateObservation, object_id: str, tracking_state: str) -> None:
        candidate.object_id = object_id
        candidate.metadata["object_id"] = object_id
        candidate.metadata["tracking_state"] = tracking_state

    def _copy_candidate_to_track(self, track: _TrackMemory, candidate: CandidateObservation, frame_id: str, state: str) -> None:
        track.roi_id = candidate.roi_id
        track.candidate_id = candidate.candidate_id
        track.obstacle_type = candidate.obstacle_type
        track.anchor = np.asarray(candidate.anchor, dtype=float)
        track.bbox = list(candidate.bbox)
        track.height_m = float(candidate.height_m)
        track.width_m = float(candidate.width_m)
        track.distance_m = float(candidate.distance_m)
        track.support_points = list(candidate.support_points)
        track.miss_count = 0
        track.last_seen_frame = frame_id
        track.state = state
        track.metadata = dict(candidate.metadata)

    def _to_output(self, track: _TrackMemory, state_override: str | None = None) -> TrackedObject:
        return TrackedObject(
            object_id=track.object_id,
            roi_id=track.roi_id,
            candidate_id=track.candidate_id,
            obstacle_type=track.obstacle_type,
            anchor=track.anchor.round(6).tolist(),
            height_m=track.height_m,
            width_m=track.width_m,
            distance_m=track.distance_m,
            state=state_override or track.state,
            miss_count=track.miss_count,
            hit_count=track.hit_count,
            last_seen_frame=track.last_seen_frame,
            support_points=track.support_points,
            metadata=dict(track.metadata),
        )

    def update(
        self,
        frame_id: str,
        candidates: list[CandidateObservation],
        cross_frame_matches: list[CrossFrameMatch],
        forward_displacement_m: float,
        post_assignment_callback: Callable[[list[CandidateObservation]], None] | None = None,
    ) -> list[TrackedObject]:
        current_tick = self._frame_tick
        self._frame_tick += 1

        match_lookup = {
            (item.prev_candidate_id, item.curr_candidate_id): item
            for item in cross_frame_matches
        }
        active_pool = list(self.active_tracks)
        retired_pool = list(self.retired_tracks)
        predicted_active = {
            track.object_id: self._predict_anchor(track, forward_displacement_m)
            for track in active_pool
        }
        predicted_retired = {
            track.object_id: self._predict_anchor(track, forward_displacement_m)
            for track in retired_pool
        }

        assigned_tracks: list[tuple[_TrackMemory, CandidateObservation, str, dict[str, Any] | None, int]] = []
        outputs: list[TrackedObject] = []
        next_active_tracks: list[_TrackMemory] = []
        next_retired_tracks: list[_TrackMemory] = []

        active_matches = self._match_track_pool(
            active_pool,
            candidates,
            predicted_active,
            list(range(len(candidates))),
            match_lookup,
            max_anchor_distance=float(self.config["max_anchor_distance_m"]),
        )
        matched_active_indices = {int(item["t_idx"]) for item in active_matches}
        matched_candidate_indices = {int(item["c_idx"]) for item in active_matches}

        for item in active_matches:
            track = active_pool[int(item["t_idx"])]
            candidate = candidates[int(item["c_idx"])]
            metrics = dict(item["metrics"])
            previous_miss_count = track.miss_count
            self._annotate_candidate(candidate, track.object_id, "updated")
            assigned_tracks.append((track, candidate, "updated", metrics, previous_miss_count))

        unmatched_candidate_indices = [idx for idx in range(len(candidates)) if idx not in matched_candidate_indices]
        retired_matches = self._match_track_pool(
            retired_pool,
            candidates,
            predicted_retired,
            unmatched_candidate_indices,
            match_lookup,
            max_anchor_distance=float(self.config.get("reactivation_max_anchor_distance_m", 0.35)),
        )
        matched_retired_indices = {int(item["t_idx"]) for item in retired_matches}
        matched_reactivated_candidates = {int(item["c_idx"]) for item in retired_matches}

        for item in retired_matches:
            track = retired_pool[int(item["t_idx"])]
            candidate = candidates[int(item["c_idx"])]
            metrics = dict(item["metrics"])
            previous_miss_count = track.miss_count
            self._annotate_candidate(candidate, track.object_id, "reactivated")
            assigned_tracks.append((track, candidate, "reactivated", metrics, previous_miss_count))

        for idx, track in enumerate(active_pool):
            if idx in matched_active_indices:
                continue
            predicted_anchor = predicted_active[track.object_id]
            track.miss_count += 1
            track.age += 1
            if predicted_anchor is None:
                track.state = "retired"
                track.recently_retired_frame = frame_id
                track.retired_tick = current_tick
                self._apply_debug_metadata(
                    track,
                    match_mode="retired_invalid_forward_distance",
                    match_score=None,
                    matched_iou=None,
                    continuity_bonus=None,
                    source_track_miss_count=track.miss_count,
                )
                next_retired_tracks.append(track)
                continue
            track.anchor = predicted_anchor
            track.distance_m = float(track.anchor[2])
            if track.miss_count <= int(self.config["max_missed_frames"]):
                track.state = "predicted"
                self._apply_debug_metadata(
                    track,
                    match_mode="predicted",
                    match_score=None,
                    matched_iou=None,
                    continuity_bonus=None,
                    source_track_miss_count=track.miss_count,
                )
                next_active_tracks.append(track)
                outputs.append(self._to_output(track))
            else:
                track.state = "retired"
                track.recently_retired_frame = frame_id
                track.retired_tick = current_tick
                self._apply_debug_metadata(
                    track,
                    match_mode="retired",
                    match_score=None,
                    matched_iou=None,
                    continuity_bonus=None,
                    source_track_miss_count=track.miss_count,
                )
                next_retired_tracks.append(track)

        for idx, track in enumerate(retired_pool):
            if idx in matched_retired_indices:
                continue
            if track.retired_tick is None:
                continue
            if current_tick - track.retired_tick < int(self.config.get("reactivation_window_frames", 3)):
                next_retired_tracks.append(track)

        fresh_candidate_indices = [
            idx for idx in unmatched_candidate_indices
            if idx not in matched_reactivated_candidates
        ]
        for idx in fresh_candidate_indices:
            candidate = candidates[idx]
            object_id = self._new_object_id()
            self._annotate_candidate(candidate, object_id, "generated")
            track = _TrackMemory(
                object_id=object_id,
                roi_id=candidate.roi_id,
                candidate_id=candidate.candidate_id,
                obstacle_type=candidate.obstacle_type,
                anchor=np.asarray(candidate.anchor, dtype=float),
                bbox=list(candidate.bbox),
                height_m=float(candidate.height_m),
                width_m=float(candidate.width_m),
                distance_m=float(candidate.distance_m),
                support_points=list(candidate.support_points),
                miss_count=0,
                last_seen_frame=frame_id,
                state="generated",
                metadata=dict(candidate.metadata),
                hit_count=1,
                age=1,
                last_match_score=None,
                recently_retired_frame=None,
                retired_tick=None,
            )
            assigned_tracks.append((track, candidate, "generated", None, 0))

        if post_assignment_callback is not None:
            post_assignment_callback(candidates)

        for track, candidate, state, metrics, source_miss_count in assigned_tracks:
            self._copy_candidate_to_track(track, candidate, frame_id, state)
            track.hit_count += 1
            track.age += 1
            track.last_match_score = None if metrics is None else float(metrics["score"])
            track.recently_retired_frame = None
            track.retired_tick = None
            self._apply_debug_metadata(
                track,
                match_mode=state,
                match_score=None if metrics is None else metrics["score"],
                matched_iou=None if metrics is None else metrics["iou"],
                continuity_bonus=None if metrics is None else metrics["continuity_bonus"],
                source_track_miss_count=source_miss_count,
            )
            next_active_tracks.append(track)
            outputs.append(self._to_output(track))

        # 只有在退休當幀輸出 retired 狀態，方便生命週期驗證。
        for track in next_retired_tracks:
            if track.recently_retired_frame == frame_id:
                outputs.append(self._to_output(track, state_override="retired"))

        self.active_tracks = next_active_tracks
        self.retired_tracks = next_retired_tracks
        outputs.sort(key=lambda item: item.object_id)
        return outputs
