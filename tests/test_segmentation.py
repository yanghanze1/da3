from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

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


@unittest.skipIf(segmentation is None, "segmentation module not ready yet")
class SegmentationTests(unittest.TestCase):
    def _base_cfg(self, *, mode: str) -> dict[str, object]:
        return {
            "backend": "fallback",
            "mode": mode,
            "min_road_area_px": 300,
            "post_kernel_size": 3,
            "hole_close_kernel_px": 11,
            "roi_min_area_px": 40,
            "nonroad_lower_half_ratio": 0.35,
            "nonroad_road_dilation_px": 31,
            "nonroad_min_area_px": 40,
            "reject_boundary_touching": True,
            "boundary_margin_px": 1,
            "fallback_lower_half_ratio": 0.40,
            "fallback_road_min_intensity": 70,
            "fallback_texture_threshold": 20.0,
        }

    def _auto_cfg(self, *, mode: str) -> dict[str, object]:
        cfg = self._base_cfg(mode=mode)
        cfg["backend"] = "auto"
        return cfg

    def test_negative_space_roi_contract(self) -> None:
        frame = np.zeros((160, 220, 3), dtype=np.uint8)
        frame[70:, :] = 110
        frame[105:128, 95:135] = 10
        frame[105:120, 0:20] = 10
        frame[130:134, 170:174] = 10

        result = segmentation.segment_frame(frame, self._base_cfg(mode="negative_space"))
        self.assertEqual(result.road_mask.shape, frame.shape[:2])
        self.assertEqual(result.roi_mask.shape, frame.shape[:2])
        self.assertGreaterEqual(len(result.roi_candidates), 1)
        boxes = [tuple(item.bbox) for item in result.roi_candidates]
        self.assertTrue(any(x <= 95 <= x + w and y <= 105 <= y + h for x, y, w, h in boxes))
        self.assertFalse(any(x == 0 for x, _, _, _ in boxes), "boundary-touching ROI should be rejected")
        self.assertTrue(all(item.source == "negative_space" for item in result.roi_candidates))
        self.assertTrue(all(item.metadata["sources"] == ["negative_space"] for item in result.roi_candidates))

    def test_nonroad_near_road_uses_lower_half_and_near_road_constraints(self) -> None:
        frame = np.zeros((200, 240, 3), dtype=np.uint8)
        frame[80:, :] = 110
        frame[110:145, 150:190] = 20
        frame[15:55, 160:210] = 20

        result = segmentation.segment_frame(frame, self._base_cfg(mode="non_road_near_road"))
        self.assertGreaterEqual(len(result.roi_candidates), 1)
        boxes = [tuple(item.bbox) for item in result.roi_candidates]
        self.assertTrue(any(x <= 150 <= x + w and y <= 110 <= y + h for x, y, w, h in boxes))
        self.assertFalse(any(y < 55 for _, y, _, _ in boxes), "upper-half background should not be promoted to ROI")
        self.assertTrue(all(item.source == "non_road_near_road" for item in result.roi_candidates))
        self.assertTrue(all(item.metadata["sources"] == ["non_road_near_road"] for item in result.roi_candidates))

    def test_hybrid_roi_merges_internal_hole_and_near_road_sources(self) -> None:
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        frame[75:, :] = 110
        frame[110:136, 90:124] = 10
        frame[110:136, 124:154] = 20

        result = segmentation.segment_frame(frame, self._base_cfg(mode="hybrid"))
        self.assertEqual(len(result.roi_candidates), 1)
        candidate = result.roi_candidates[0]
        self.assertEqual(candidate.source, "hybrid")
        self.assertCountEqual(candidate.metadata["sources"], ["negative_space", "non_road_near_road"])
        x, y, w, h = candidate.bbox
        self.assertLessEqual(x, 90)
        self.assertGreaterEqual(x + w, 154)
        self.assertLessEqual(y, 110)
        self.assertGreaterEqual(y + h, 136)

    def test_boundary_touching_filter_still_applies_in_hybrid_mode(self) -> None:
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        frame[75:, :] = 110
        frame[110:138, 90:124] = 10
        frame[110:138, 0:24] = 20

        result = segmentation.segment_frame(frame, self._base_cfg(mode="hybrid"))
        boxes = [tuple(item.bbox) for item in result.roi_candidates]
        self.assertTrue(any(x <= 90 <= x + w for x, _, w, _ in boxes))
        self.assertFalse(any(x == 0 for x, _, _, _ in boxes), "hybrid mode should still reject border-touching ROI")

    def test_processing_rect_limits_analysis_mask_without_limiting_road_mask(self) -> None:
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        frame[75:, :] = 110
        frame[110:136, 90:124] = 10
        frame[110:136, 170:205] = 10
        cfg = self._base_cfg(mode="negative_space")
        base = segmentation.segment_frame(frame, cfg)
        cfg["processing_rect"] = {
            "enabled": True,
            "bbox_xywh": [80, 95, 60, 55],
            "restrict_analysis_mask": True,
            "restrict_road_mask": False,
        }

        result = segmentation.segment_frame(frame, cfg)
        rect_mask = segmentation.bbox_to_mask(frame.shape, [80, 95, 60, 55])
        self.assertFalse(np.any(result.analysis_mask & ~rect_mask))
        self.assertTrue(np.array_equal(result.road_mask, base.road_mask))
        self.assertEqual(result.processing_roi["bbox"], [80, 95, 60, 55])
        self.assertTrue(all(item.bbox[0] < 140 for item in result.roi_candidates))

    def test_processing_rect_can_limit_road_mask(self) -> None:
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        frame[75:, :] = 110
        cfg = self._base_cfg(mode="negative_space")
        cfg["processing_rect"] = {
            "enabled": True,
            "bbox_xywh": [80, 95, 60, 55],
            "restrict_analysis_mask": True,
            "restrict_road_mask": True,
        }

        result = segmentation.segment_frame(frame, cfg)
        rect_mask = segmentation.bbox_to_mask(frame.shape, [80, 95, 60, 55])
        self.assertFalse(np.any(result.road_mask & ~rect_mask))

    def test_normalized_processing_rect_resolves_to_pixel_bbox(self) -> None:
        cfg = {
            "processing_rect": {
                "enabled": True,
                "normalized_bbox_xywh": [0.1, 0.25, 0.5, 0.5],
            }
        }

        rect = segmentation.resolve_processing_rect((100, 200, 3), cfg)
        self.assertEqual(rect["bbox"], [20, 25, 100, 50])
        self.assertEqual(rect["source"], "manual_normalized")

    def test_trapezoid_processing_roi_bottom_spans_full_image_width(self) -> None:
        cfg = {
            "processing_rect": {
                "enabled": True,
                "mode": "trapezoid",
                "normalized_bbox_xywh": [0.1, 0.25, 0.5, 0.5],
            }
        }

        roi = segmentation.resolve_processing_roi((100, 200, 3), cfg)
        self.assertEqual(roi["mode"], "trapezoid")
        self.assertEqual(roi["bbox"], [20, 25, 100, 50])
        self.assertEqual(roi["polygon"], [[20, 25], [120, 25], [199, 99], [0, 99]])
        self.assertEqual(roi["obstacle_analysis_mask"], "processing_roi")

    def test_trapezoid_processing_roi_builds_obstacle_mask_without_clipping_road_mask(self) -> None:
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        frame[45:, :] = 110
        frame[70:95, 86:114] = 10
        cfg = self._base_cfg(mode="negative_space")
        base = segmentation.segment_frame(frame, cfg)
        cfg["processing_rect"] = {
            "enabled": True,
            "mode": "trapezoid",
            "normalized_bbox_xywh": [0.25, 0.30, 0.50, 0.60],
            "obstacle_analysis_mask": "processing_roi",
            "restrict_analysis_mask": False,
            "restrict_road_mask": False,
        }

        result = segmentation.segment_frame(frame, cfg)
        self.assertIsNotNone(result.obstacle_analysis_mask)
        self.assertTrue(np.all(result.obstacle_analysis_mask[-1, :]))
        self.assertTrue(np.array_equal(result.road_mask, base.road_mask))
        self.assertTrue(np.array_equal(result.analysis_mask, base.analysis_mask))

    def test_auto_backend_prefers_tensorrt_before_onnx_or_fallback(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        expected = segmentation.FrontendResult(
            road_mask=np.zeros((32, 32), dtype=bool),
            road_probability=np.zeros((32, 32), dtype=np.float32),
            analysis_mask=np.zeros((32, 32), dtype=bool),
            roi_candidates=[],
            backend="tensorrt_pidnet:pidnet-s",
        )
        with (
            mock.patch.object(segmentation, "_tensorrt_backend", return_value=expected) as trt_mock,
            mock.patch.object(segmentation, "_onnx_backend") as onnx_mock,
            mock.patch.object(segmentation, "_fallback_backend") as fallback_mock,
        ):
            result = segmentation.segment_frame(frame, self._auto_cfg(mode="hybrid"))
        self.assertIs(result, expected)
        trt_mock.assert_called_once()
        onnx_mock.assert_not_called()
        fallback_mock.assert_not_called()

    def test_auto_backend_falls_back_to_onnx_then_fallback(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        onnx_result = segmentation.FrontendResult(
            road_mask=np.zeros((32, 32), dtype=bool),
            road_probability=np.zeros((32, 32), dtype=np.float32),
            analysis_mask=np.zeros((32, 32), dtype=bool),
            roi_candidates=[],
            backend="onnx_pidnet:pidnet-s",
        )
        fallback_result = segmentation.FrontendResult(
            road_mask=np.zeros((32, 32), dtype=bool),
            road_probability=np.zeros((32, 32), dtype=np.float32),
            analysis_mask=np.zeros((32, 32), dtype=bool),
            roi_candidates=[],
            backend="fallback",
        )

        with (
            mock.patch.object(segmentation, "_tensorrt_backend", return_value=None) as trt_mock,
            mock.patch.object(segmentation, "_onnx_backend", return_value=onnx_result) as onnx_mock,
            mock.patch.object(segmentation, "_fallback_backend", return_value=fallback_result) as fallback_mock,
        ):
            result = segmentation.segment_frame(frame, self._auto_cfg(mode="hybrid"))
            self.assertIs(result, onnx_result)
            fallback_mock.assert_not_called()

        with (
            mock.patch.object(segmentation, "_tensorrt_backend", return_value=None) as trt_mock,
            mock.patch.object(segmentation, "_onnx_backend", return_value=None) as onnx_mock,
            mock.patch.object(segmentation, "_official_pidnet_backend", return_value=None) as official_mock,
            mock.patch.object(segmentation, "_fallback_backend", return_value=fallback_result) as fallback_mock,
        ):
            result = segmentation.segment_frame(frame, self._auto_cfg(mode="hybrid"))
            self.assertIs(result, fallback_result)
            trt_mock.assert_called_once()
            onnx_mock.assert_called_once()
            official_mock.assert_called_once()
            fallback_mock.assert_called_once()
            self.assertEqual([item["backend"] for item in result.backend_attempts], ["tensorrt", "onnx", "official_pidnet", "fallback"])

    def test_auto_backend_order_can_prefer_official_pidnet(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        official_result = segmentation.FrontendResult(
            road_mask=np.zeros((32, 32), dtype=bool),
            road_probability=np.zeros((32, 32), dtype=np.float32),
            analysis_mask=np.zeros((32, 32), dtype=bool),
            roi_candidates=[],
            backend="official_pidnet:pidnet-s",
        )
        cfg = self._auto_cfg(mode="hybrid")
        cfg["auto_backend_order"] = ["official_pidnet", "fallback"]
        with (
            mock.patch.object(segmentation, "_official_pidnet_backend", return_value=official_result) as official_mock,
            mock.patch.object(segmentation, "_fallback_backend") as fallback_mock,
        ):
            result = segmentation.segment_frame(frame, cfg)
        self.assertIs(result, official_result)
        official_mock.assert_called_once()
        fallback_mock.assert_not_called()
        self.assertEqual(result.backend_attempts[0]["backend"], "official_pidnet")
        self.assertEqual(result.road_mask_stats()["backend_attempts"][0]["status"], "ok")

    def test_successful_backend_can_be_augmented_with_fallback_rois(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        official_result = segmentation.FrontendResult(
            road_mask=np.zeros((32, 32), dtype=bool),
            road_probability=np.zeros((32, 32), dtype=np.float32),
            analysis_mask=np.zeros((32, 32), dtype=bool),
            roi_candidates=[],
            backend="official_pidnet:pidnet-s",
            backend_attempts=[{"backend": "official_pidnet", "status": "ok"}],
        )
        fallback_roi = segmentation.RoiCandidate(
            roi_id="roi_000",
            bbox=[4, 4, 8, 8],
            area_px=64,
            touch_border=False,
            source="hybrid",
        )
        fallback_result = segmentation.FrontendResult(
            road_mask=np.zeros((32, 32), dtype=bool),
            road_probability=np.zeros((32, 32), dtype=np.float32),
            analysis_mask=segmentation.bbox_to_mask(frame.shape, fallback_roi.bbox),
            roi_candidates=[fallback_roi],
            backend="fallback",
        )
        cfg = self._auto_cfg(mode="hybrid")
        cfg["auto_backend_order"] = ["official_pidnet"]
        cfg["augment_with_fallback_rois"] = True
        with (
            mock.patch.object(segmentation, "_official_pidnet_backend", return_value=official_result),
            mock.patch.object(segmentation, "_fallback_backend", return_value=fallback_result),
        ):
            result = segmentation.segment_frame(frame, cfg)
        self.assertEqual(result.backend, "official_pidnet:pidnet-s")
        self.assertEqual(len(result.roi_candidates), 1)
        self.assertEqual(result.roi_candidates[0].source, "fallback_hybrid")
        self.assertEqual(result.backend_attempts[-1]["backend"], "fallback_roi_augmentation")

    def test_auto_backend_can_fail_on_fallback(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        fallback_result = segmentation.FrontendResult(
            road_mask=np.zeros((32, 32), dtype=bool),
            road_probability=np.zeros((32, 32), dtype=np.float32),
            analysis_mask=np.zeros((32, 32), dtype=bool),
            roi_candidates=[],
            backend="fallback",
        )
        cfg = self._auto_cfg(mode="hybrid")
        cfg["auto_backend_order"] = ["fallback"]
        cfg["fail_on_fallback"] = True
        with mock.patch.object(segmentation, "_fallback_backend", return_value=fallback_result):
            with self.assertRaises(RuntimeError):
                segmentation.segment_frame(frame, cfg)


if __name__ == "__main__":
    unittest.main()
