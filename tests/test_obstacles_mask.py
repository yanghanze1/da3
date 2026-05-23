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
obstacles = _optional_import("monocular_experiment.obstacles")
pipeline = _optional_import("monocular_experiment.pipeline")


@unittest.skipIf(models is None or obstacles is None or pipeline is None, "obstacle modules not ready yet")
class ObstacleValidationTests(unittest.TestCase):
    def _depth_roi_geometry(self):
        plane = models.PlaneEstimate(
            normal=[0.0, 0.0, 1.0],
            offset=-2.0,
            inlier_ratio=0.95,
            support_count=40,
            source="road_mask_ransac",
            status="ok",
        )
        intrinsics = {"camera_matrix": [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]}
        extrinsics = {
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translation_vector": [0.0, 0.0, 0.0],
        }
        return plane, intrinsics, extrinsics

    def test_depth_residual_roi_detects_positive_contour(self) -> None:
        plane, intrinsics, extrinsics = self._depth_roi_geometry()
        image_points = np.array([[20.0, 20.0], [21.0, 20.0], [20.0, 21.0], [21.0, 21.0], [80.0, 80.0]], dtype=float)
        depths = np.array([1.9, 1.9, 1.9, 1.9, 2.0], dtype=float)
        analysis_mask = np.zeros((100, 100), dtype=bool)
        analysis_mask[10:40, 10:40] = True

        rois = obstacles.build_depth_residual_roi_candidates(
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            frame_shape=(100, 100, 3),
            analysis_mask=analysis_mask,
            config={
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "depth_contour_roi": {"enabled": True, "min_area_px": 2, "dilate_kernel_px": 1, "close_kernel_px": 1},
            },
        )
        self.assertEqual(len(rois), 1)
        self.assertEqual(rois[0].source, "depth_residual_contour")
        self.assertEqual(rois[0].metadata["sign"], "positive")
        self.assertEqual(rois[0].metadata["raw_bbox"], rois[0].bbox)
        x, y, w, h = rois[0].bbox
        self.assertLessEqual(x, 20)
        self.assertLessEqual(y, 20)
        self.assertGreaterEqual(x + w, 21)
        self.assertGreaterEqual(y + h, 21)

    def test_depth_residual_roi_bbox_can_expand_component(self) -> None:
        plane, intrinsics, extrinsics = self._depth_roi_geometry()
        image_points = np.array([[20.0, 20.0], [21.0, 20.0], [20.0, 21.0], [21.0, 21.0]], dtype=float)
        depths = np.array([1.9, 1.9, 1.9, 1.9], dtype=float)
        analysis_mask = np.ones((100, 100), dtype=bool)

        rois = obstacles.build_depth_residual_roi_candidates(
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            frame_shape=(100, 100, 3),
            analysis_mask=analysis_mask,
            config={
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "depth_contour_roi": {
                    "enabled": True,
                    "min_area_px": 2,
                    "dilate_kernel_px": 1,
                    "close_kernel_px": 1,
                    "bbox_expand_x_px": 4,
                    "bbox_expand_top_px": 2,
                    "bbox_expand_bottom_px": 6,
                },
            },
        )
        self.assertEqual(rois[0].metadata["raw_bbox"], [20, 20, 2, 2])
        self.assertEqual(rois[0].bbox, [16, 18, 10, 10])

    def test_depth_residual_roi_detects_negative_contour(self) -> None:
        plane, intrinsics, extrinsics = self._depth_roi_geometry()
        image_points = np.array([[30.0, 30.0], [31.0, 30.0], [30.0, 31.0], [31.0, 31.0]], dtype=float)
        depths = np.array([2.1, 2.1, 2.1, 2.1], dtype=float)
        analysis_mask = np.ones((100, 100), dtype=bool)

        rois = obstacles.build_depth_residual_roi_candidates(
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            frame_shape=(100, 100, 3),
            analysis_mask=analysis_mask,
            config={
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "depth_contour_roi": {"enabled": True, "min_area_px": 2, "dilate_kernel_px": 1, "close_kernel_px": 1},
            },
        )
        self.assertEqual(len(rois), 1)
        self.assertEqual(rois[0].metadata["sign"], "negative")

    def test_depth_roi_min_component_points_filters_singleton_noise(self) -> None:
        plane, intrinsics, extrinsics = self._depth_roi_geometry()
        image_points = np.array([[20.0, 20.0]], dtype=float)
        depths = np.array([1.9], dtype=float)
        analysis_mask = np.ones((100, 100), dtype=bool)

        rois = obstacles.build_depth_residual_roi_candidates(
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            frame_shape=(100, 100, 3),
            analysis_mask=analysis_mask,
            config={
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "depth_contour_roi": {
                    "enabled": True,
                    "min_area_px": 1,
                    "min_component_points": 2,
                    "dilate_kernel_px": 5,
                    "close_kernel_px": 1,
                },
            },
        )
        self.assertEqual(rois, [])

    def test_depth_roi_stricter_min_area_rejects_tiny_component(self) -> None:
        plane, intrinsics, extrinsics = self._depth_roi_geometry()
        image_points = np.array([[20.0, 20.0], [21.0, 20.0]], dtype=float)
        depths = np.array([1.9, 1.9], dtype=float)
        analysis_mask = np.ones((100, 100), dtype=bool)

        rois = obstacles.build_depth_residual_roi_candidates(
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            frame_shape=(100, 100, 3),
            analysis_mask=analysis_mask,
            config={
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "depth_contour_roi": {"enabled": True, "min_area_px": 10, "dilate_kernel_px": 1, "close_kernel_px": 1},
            },
        )
        self.assertEqual(rois, [])

    def test_depth_residual_roi_respects_area_and_analysis_mask(self) -> None:
        plane, intrinsics, extrinsics = self._depth_roi_geometry()
        image_points = np.array([[20.0, 20.0], [21.0, 20.0], [60.0, 60.0]], dtype=float)
        depths = np.array([1.9, 1.9, 1.9], dtype=float)
        analysis_mask = np.zeros((100, 100), dtype=bool)
        analysis_mask[10:40, 10:40] = True

        rois = obstacles.build_depth_residual_roi_candidates(
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            frame_shape=(100, 100, 3),
            analysis_mask=analysis_mask,
            config={
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "depth_contour_roi": {"enabled": True, "min_area_px": 3, "dilate_kernel_px": 1, "close_kernel_px": 1},
            },
        )
        self.assertEqual(rois, [])

    def test_refined_roi_limits_candidate_observations(self) -> None:
        plane, intrinsics, extrinsics = self._depth_roi_geometry()
        image_points = np.array([[20.0, 20.0], [21.0, 20.0], [20.0, 21.0], [70.0, 70.0], [71.0, 70.0], [70.0, 71.0]], dtype=float)
        depths = np.array([1.9, 1.9, 1.9, 1.9, 1.9, 1.9], dtype=float)
        world_points = np.array([[0.0, 0.00, 1.9], [0.0, 0.02, 1.9], [0.0, 0.04, 1.9], [0.0, 2.00, 1.9], [0.0, 2.02, 1.9], [0.0, 2.04, 1.9]], dtype=float)
        refined_roi = models.RoiCandidate(roi_id="depth_roi_000", bbox=[18, 18, 8, 8], area_px=64, touch_border=False, source="depth_residual_contour")

        clusters = obstacles.build_candidate_observations(
            roi_candidates=[refined_roi],
            candidate_world_points=world_points,
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            config={
                "height_reference": "depth_residual_to_plane",
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "dbscan_eps_m": 0.25,
                "dbscan_min_samples": 2,
                "min_cluster_points": 2,
                "min_abnormal_ratio": 0.4,
            },
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].point_count, 3)
        self.assertEqual(clusters[0].metadata["roi_source"], "depth_residual_contour")

    def test_depth_roi_uses_cluster_bbox_but_keeps_expanded_roi_metadata(self) -> None:
        plane, intrinsics, extrinsics = self._depth_roi_geometry()
        image_points = np.array([[20.0, 20.0], [21.0, 20.0], [20.0, 21.0], [50.0, 50.0]], dtype=float)
        depths = np.array([1.9, 1.9, 1.9, 2.0], dtype=float)
        world_points = np.array([[0.0, 0.00, 1.9], [0.0, 0.02, 1.9], [0.0, 0.04, 1.9], [0.0, 1.0, 2.0]], dtype=float)
        roi = models.RoiCandidate(
            roi_id="depth_roi_000",
            bbox=[8, 8, 40, 40],
            area_px=9,
            touch_border=False,
            source="depth_residual_contour",
            metadata={"raw_bbox": [20, 20, 2, 2]},
        )

        clusters = obstacles.build_candidate_observations(
            roi_candidates=[roi],
            candidate_world_points=world_points,
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            config={
                "height_reference": "depth_residual_to_plane",
                "use_roi_bbox_for_depth_roi": False,
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "dbscan_eps_m": 0.25,
                "dbscan_min_samples": 2,
                "min_cluster_points": 2,
                "min_abnormal_ratio": 0.4,
            },
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].bbox, [20, 20, 1, 1])
        self.assertEqual(clusters[0].metadata["cluster_bbox"], [20, 20, 1, 1])
        self.assertEqual(clusters[0].metadata["padded_cluster_bbox"], [20, 20, 1, 1])
        self.assertEqual(clusters[0].metadata["raw_bbox"], [20, 20, 2, 2])
        self.assertEqual(clusters[0].metadata["roi_bbox"], [8, 8, 40, 40])

    def test_cluster_bbox_padding_is_limited_to_support_roi(self) -> None:
        plane, intrinsics, extrinsics = self._depth_roi_geometry()
        image_points = np.array([[20.0, 20.0], [21.0, 20.0], [20.0, 21.0], [50.0, 50.0]], dtype=float)
        depths = np.array([1.9, 1.9, 1.9, 2.0], dtype=float)
        world_points = np.array([[0.0, 0.00, 1.9], [0.0, 0.02, 1.9], [0.0, 0.04, 1.9], [0.0, 1.0, 2.0]], dtype=float)
        roi = models.RoiCandidate(
            roi_id="depth_roi_000",
            bbox=[18, 18, 8, 8],
            area_px=9,
            touch_border=False,
            source="depth_residual_contour",
            metadata={"raw_bbox": [20, 20, 2, 2]},
        )

        clusters = obstacles.build_candidate_observations(
            roi_candidates=[roi],
            candidate_world_points=world_points,
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            config={
                "height_reference": "depth_residual_to_plane",
                "use_roi_bbox_for_depth_roi": False,
                "cluster_bbox_padding_x_px": 8,
                "cluster_bbox_padding_top_px": 4,
                "cluster_bbox_padding_bottom_px": 12,
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "dbscan_eps_m": 0.25,
                "dbscan_min_samples": 2,
                "min_cluster_points": 2,
                "min_abnormal_ratio": 0.4,
            },
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].metadata["cluster_bbox"], [20, 20, 1, 1])
        self.assertEqual(clusters[0].bbox, [18, 18, 8, 8])
        self.assertEqual(clusters[0].metadata["padded_cluster_bbox"], [18, 18, 8, 8])

    def test_positive_cluster_is_built_from_abnormal_points(self) -> None:
        plane = models.PlaneEstimate(
            normal=[1.0, 0.0, 0.0],
            offset=0.0,
            inlier_ratio=0.95,
            support_count=40,
            source="road_mask_ransac",
            status="ok",
        )
        roi = models.RoiCandidate(roi_id="roi_000", bbox=[110, 130, 40, 30], area_px=1200, touch_border=False, source="hybrid")
        world_points = np.array(
            [
                [0.08, -0.10, 1.20],
                [0.09, -0.08, 1.22],
                [0.07, -0.06, 1.18],
                [0.00, 0.20, 2.50],
            ],
            dtype=float,
        )
        image_points = np.array(
            [
                [120.0, 140.0],
                [125.0, 144.0],
                [122.0, 146.0],
                [10.0, 20.0],
            ],
            dtype=float,
        )
        depths = world_points[:, 2]
        clusters = obstacles.build_candidate_observations(
            roi_candidates=[roi],
            candidate_world_points=world_points,
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            config={
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "dbscan_eps_m": 0.25,
                "dbscan_min_samples": 2,
                "min_cluster_points": 2,
                "min_abnormal_ratio": 0.4,
            },
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].obstacle_type, "positive")
        self.assertEqual(clusters[0].point_count, 3)
        self.assertAlmostEqual(clusters[0].distance_m, 1.18, places=2)

    def test_candidate_generation_ignores_non_forward_points_for_distance(self) -> None:
        plane = models.PlaneEstimate(
            normal=[1.0, 0.0, 0.0],
            offset=0.0,
            inlier_ratio=0.95,
            support_count=40,
            source="road_mask_ransac",
            status="ok",
        )
        roi = models.RoiCandidate(roi_id="roi_000", bbox=[100, 100, 40, 60], area_px=2400, touch_border=False, source="hybrid")
        world_points = np.array(
            [
                [0.08, -0.10, -0.2],
                [0.09, -0.08, 0.01],
                [0.07, -0.06, 1.20],
                [0.08, -0.04, 1.30],
                [0.09, -0.02, 1.40],
            ],
            dtype=float,
        )
        image_points = np.array([[110.0, 110.0], [112.0, 112.0], [120.0, 140.0], [125.0, 144.0], [122.0, 146.0]], dtype=float)
        clusters = obstacles.build_candidate_observations(
            roi_candidates=[roi],
            candidate_world_points=world_points,
            candidate_image_points=image_points,
            candidate_depths_m=world_points[:, 2],
            plane=plane,
            config={
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "dbscan_eps_m": 0.25,
                "dbscan_min_samples": 2,
                "min_cluster_points": 2,
                "min_valid_forward_points": 2,
                "min_forward_distance_m": 0.05,
                "distance_percentile": 5.0,
                "min_abnormal_ratio": 0.2,
            },
        )
        self.assertEqual(len(clusters), 1)
        self.assertGreater(clusters[0].distance_m, 0.05)
        self.assertGreaterEqual(min(point[2] for point in clusters[0].support_points), 1.2)

    def test_candidate_generation_rejects_invalid_height(self) -> None:
        plane = models.PlaneEstimate(
            normal=[1.0, 0.0, 0.0],
            offset=0.0,
            inlier_ratio=0.95,
            support_count=40,
            source="road_mask_ransac",
            status="ok",
        )
        roi = models.RoiCandidate(roi_id="roi_000", bbox=[100, 100, 40, 60], area_px=2400, touch_border=False, source="hybrid")
        world_points = np.array(
            [
                [2.20, -0.10, 1.20],
                [2.30, -0.08, 1.25],
                [2.40, -0.06, 1.30],
            ],
            dtype=float,
        )
        image_points = np.array([[120.0, 140.0], [125.0, 144.0], [122.0, 146.0]], dtype=float)
        clusters = obstacles.build_candidate_observations(
            roi_candidates=[roi],
            candidate_world_points=world_points,
            candidate_image_points=image_points,
            candidate_depths_m=world_points[:, 2],
            plane=plane,
            config={
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "max_valid_height_m": 1.5,
                "dbscan_eps_m": 0.25,
                "dbscan_min_samples": 2,
                "min_cluster_points": 2,
                "min_abnormal_ratio": 0.4,
            },
        )
        self.assertEqual(clusters, [])

    def test_negative_cluster_is_built_from_negative_height_points(self) -> None:
        plane = models.PlaneEstimate(
            normal=[1.0, 0.0, 0.0],
            offset=0.0,
            inlier_ratio=0.95,
            support_count=40,
            source="road_mask_ransac",
            status="ok",
        )
        roi = models.RoiCandidate(roi_id="roi_001", bbox=[100, 100, 40, 40], area_px=1600, touch_border=False, source="hybrid")
        world_points = np.array(
            [
                [-0.08, -0.10, 1.0],
                [-0.09, -0.08, 1.02],
                [-0.07, -0.06, 1.04],
                [0.00, 0.20, 2.50],
            ],
            dtype=float,
        )
        image_points = np.array(
            [
                [110.0, 110.0],
                [112.0, 112.0],
                [115.0, 115.0],
                [20.0, 20.0],
            ],
            dtype=float,
        )
        depths = world_points[:, 2]
        clusters = obstacles.build_candidate_observations(
            roi_candidates=[roi],
            candidate_world_points=world_points,
            candidate_image_points=image_points,
            candidate_depths_m=depths,
            plane=plane,
            config={
                "h_pos_m": 0.03,
                "h_neg_m": 0.03,
                "dbscan_eps_m": 0.25,
                "dbscan_min_samples": 2,
                "min_cluster_points": 2,
                "min_abnormal_ratio": 0.4,
            },
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].obstacle_type, "negative")
        self.assertGreater(clusters[0].height_m, 0.05)

    def test_da3_only_mode_does_not_fallback_to_frontend_rois(self) -> None:
        frontend_roi = models.RoiCandidate(
            roi_id="frontend_000",
            bbox=[10, 10, 20, 20],
            area_px=400,
            touch_border=False,
            source="hybrid",
        )

        selected = pipeline._select_roi_candidates_for_obstacles(
            frontend_rois=[frontend_roi],
            depth_roi_candidates=[],
            candidate_cfg={"depth_contour_roi": {"enabled": True, "fallback_to_frontend_rois": False}},
        )
        self.assertEqual(selected, [])

    def test_union_mode_keeps_depth_and_frontend_rois_with_deduplication(self) -> None:
        frontend_roi = models.RoiCandidate(
            roi_id="frontend_000",
            bbox=[80, 80, 20, 20],
            area_px=400,
            touch_border=False,
            source="hybrid",
        )
        duplicate_frontend_roi = models.RoiCandidate(
            roi_id="frontend_001",
            bbox=[11, 11, 20, 20],
            area_px=400,
            touch_border=False,
            source="hybrid",
        )
        depth_roi = models.RoiCandidate(
            roi_id="depth_000",
            bbox=[10, 10, 20, 20],
            area_px=400,
            touch_border=False,
            source="depth_residual_contour",
        )

        selected = pipeline._select_roi_candidates_for_obstacles(
            frontend_rois=[duplicate_frontend_roi, frontend_roi],
            depth_roi_candidates=[depth_roi],
            candidate_cfg={
                "depth_contour_roi": {
                    "enabled": True,
                    "selection_mode": "union",
                    "deduplicate_iou_threshold": 0.7,
                }
            },
        )
        self.assertEqual([item.roi_id for item in selected], ["depth_000", "frontend_000"])

    def test_source_aware_selection_caps_and_prefers_depth_rois(self) -> None:
        depth_rois = [
            models.RoiCandidate(
                roi_id=f"depth_{idx:03d}",
                bbox=[10 * idx, 20, 6, 6],
                area_px=36,
                touch_border=False,
                source="depth_residual_contour",
                metadata={"point_count": 6 - idx, "mask_fill_ratio": 0.5},
            )
            for idx in range(3)
        ]
        frontend_roi = models.RoiCandidate(
            roi_id="frontend_000",
            bbox=[80, 80, 8, 8],
            area_px=64,
            touch_border=False,
            source="hybrid",
        )

        selected = pipeline._select_roi_candidates_for_obstacles(
            frontend_rois=[frontend_roi],
            depth_roi_candidates=depth_rois,
            candidate_cfg={
                "depth_contour_roi": {
                    "enabled": True,
                    "selection_mode": "depth_plus_verified_frontend",
                    "max_selected_rois_per_frame": 2,
                    "deduplicate_iou_threshold": 0.7,
                }
            },
            frame_shape=(100, 100, 3),
            frontend_backend="official_pidnet:pidnet-s",
        )

        self.assertEqual([item.roi_id for item in selected], ["depth_000", "frontend_000"])

    def test_source_aware_selection_filters_bottom_edge_depth_rois(self) -> None:
        bottom_roi = models.RoiCandidate(
            roi_id="depth_bottom",
            bbox=[20, 88, 12, 10],
            area_px=120,
            touch_border=False,
            source="depth_residual_contour",
            metadata={"point_count": 20, "mask_fill_ratio": 0.8},
        )
        valid_roi = models.RoiCandidate(
            roi_id="depth_valid",
            bbox=[20, 50, 12, 10],
            area_px=120,
            touch_border=False,
            source="depth_residual_contour",
            metadata={"point_count": 4, "mask_fill_ratio": 0.3},
        )

        selected = pipeline._select_roi_candidates_for_obstacles(
            frontend_rois=[],
            depth_roi_candidates=[bottom_roi, valid_roi],
            candidate_cfg={
                "depth_contour_roi": {
                    "enabled": True,
                    "selection_mode": "depth_plus_verified_frontend",
                    "max_roi_bottom_ratio": 0.82,
                    "deduplicate_iou_threshold": 0.7,
                }
            },
            frame_shape=(100, 100, 3),
            frontend_backend="official_pidnet:pidnet-s",
        )

        self.assertEqual([item.roi_id for item in selected], ["depth_valid"])

    def test_containment_dedup_drops_large_lower_priority_roi(self) -> None:
        depth_roi = models.RoiCandidate(
            roi_id="depth_000",
            bbox=[20, 20, 12, 12],
            area_px=144,
            touch_border=False,
            source="depth_residual_contour",
        )
        frontend_roi = models.RoiCandidate(
            roi_id="frontend_000",
            bbox=[10, 10, 40, 40],
            area_px=1600,
            touch_border=False,
            source="hybrid",
        )

        selected = pipeline._select_roi_candidates_for_obstacles(
            frontend_rois=[frontend_roi],
            depth_roi_candidates=[depth_roi],
            candidate_cfg={
                "depth_contour_roi": {
                    "enabled": True,
                    "selection_mode": "depth_plus_verified_frontend",
                    "deduplicate_iou_threshold": 0.9,
                    "containment_threshold": 0.8,
                }
            },
            frame_shape=(100, 100, 3),
            frontend_backend="official_pidnet:pidnet-s",
        )

        self.assertEqual([item.roi_id for item in selected], ["depth_000"])

    def test_obstacle_analysis_mask_uses_processing_roi_not_pidnet_analysis_mask(self) -> None:
        analysis_mask = np.zeros((20, 20), dtype=bool)
        obstacle_mask = np.zeros((20, 20), dtype=bool)
        obstacle_mask[10:, :] = True
        frontend = type(
            "FrontendLike",
            (),
            {"analysis_mask": analysis_mask, "obstacle_analysis_mask": obstacle_mask},
        )()

        resolved_mask, source = pipeline._resolve_obstacle_analysis_mask(frontend)
        self.assertEqual(source, "processing_roi")
        self.assertTrue(np.array_equal(resolved_mask, obstacle_mask))
        self.assertFalse(np.array_equal(resolved_mask, analysis_mask))

    def test_cross_frame_summary_prefers_geometry_consistent_match(self) -> None:
        prev_candidate = models.CandidateObservation(
            candidate_id="cand_000",
            roi_id="roi_000",
            obstacle_type="positive",
            bbox=[100, 100, 20, 20],
            centroid_2d=[110.0, 110.0],
            anchor=[0.08, 0.0, 1.2],
            height_m=0.08,
            width_m=0.20,
            distance_m=1.2,
            z_range_m=[1.2, 1.4],
            point_count=5,
            abnormal_count=5,
            support_points=[],
        )
        current_good = models.CandidateObservation(
            candidate_id="cand_001",
            roi_id="roi_001",
            obstacle_type="positive",
            bbox=[102, 102, 20, 20],
            centroid_2d=[112.0, 112.0],
            anchor=[0.08, 0.0, 1.1],
            height_m=0.08,
            width_m=0.19,
            distance_m=1.1,
            z_range_m=[1.1, 1.3],
            point_count=5,
            abnormal_count=5,
            support_points=[],
        )
        current_bad = models.CandidateObservation(
            candidate_id="cand_002",
            roi_id="roi_002",
            obstacle_type="negative",
            bbox=[260, 260, 20, 20],
            centroid_2d=[270.0, 270.0],
            anchor=[0.50, 0.4, 2.5],
            height_m=0.10,
            width_m=0.30,
            distance_m=2.5,
            z_range_m=[2.5, 2.8],
            point_count=5,
            abnormal_count=5,
            support_points=[],
        )
        matches = obstacles.summarize_cross_frame_matches(
            previous_candidates=[prev_candidate],
            current_candidates=[current_good, current_bad],
            config={
                "max_anchor_distance_m": 0.6,
                "max_centroid_shift_px": 60.0,
                "min_match_score": 0.05,
            },
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].curr_candidate_id, "cand_001")

    def test_cross_frame_summary_compensates_forward_displacement(self) -> None:
        prev_candidate = models.CandidateObservation(
            candidate_id="cand_000",
            roi_id="roi_000",
            obstacle_type="positive",
            bbox=[100, 100, 20, 20],
            centroid_2d=[110.0, 110.0],
            anchor=[0.08, 0.0, 1.4],
            height_m=0.08,
            width_m=0.20,
            distance_m=1.4,
            z_range_m=[1.4, 1.6],
            point_count=5,
            abnormal_count=5,
            support_points=[],
        )
        current_candidate = models.CandidateObservation(
            candidate_id="cand_001",
            roi_id="roi_001",
            obstacle_type="positive",
            bbox=[101, 101, 20, 20],
            centroid_2d=[111.0, 111.0],
            anchor=[0.08, 0.0, 1.0],
            height_m=0.08,
            width_m=0.19,
            distance_m=1.0,
            z_range_m=[1.0, 1.2],
            point_count=5,
            abnormal_count=5,
            support_points=[],
        )
        matches = obstacles.summarize_cross_frame_matches(
            previous_candidates=[prev_candidate],
            current_candidates=[current_candidate],
            config={"max_anchor_distance_m": 0.2, "max_centroid_shift_px": 60.0, "min_match_score": 0.05},
            forward_displacement_m=0.4,
        )
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0].metadata["anchor_distance_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
