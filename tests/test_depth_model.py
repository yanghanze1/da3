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


depth_model = _optional_import("monocular_experiment.depth_model")


@unittest.skipIf(depth_model is None, "depth_model module not ready yet")
class DepthModelTests(unittest.TestCase):
    def test_mock_backend_keeps_392_upper_bound_resize_metadata(self) -> None:
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        result = depth_model.infer_relative_depth(
            frame,
            {
                "backend": "mock",
                "model_name": "DA3-SMALL",
                "process_res": 392,
                "process_res_method": "upper_bound_resize",
            },
        )
        self.assertEqual(result["backend"], "mock_da3")
        self.assertEqual(result["metadata"]["process_res"], 392)
        self.assertEqual(result["metadata"]["process_res_method"], "upper_bound_resize")
        self.assertEqual(result["relative_depth"].shape, frame.shape[:2])


if __name__ == "__main__":
    unittest.main()
