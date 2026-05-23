from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def _make_chessboard(rows: int, cols: int, square_px: int) -> np.ndarray:
    h = (rows + 1) * square_px
    w = (cols + 1) * square_px
    board = np.zeros((h, w), dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            color = 255 if (r + c) % 2 == 0 else 0
            board[r * square_px : (r + 1) * square_px, c * square_px : (c + 1) * square_px] = color
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_yaml_like_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))


def make_demo_sequence(dataset_root: str | Path, sequence_id: str, calibration_dir: str | Path) -> dict[str, Any]:
    """建立新版 smoke test 可直接跑通的 demo 序列。"""

    dataset_root = Path(dataset_root)
    sequence_dir = _ensure_dir(dataset_root / sequence_id)
    frames_dir = _ensure_dir(sequence_dir / "frames")
    calibration_dir = _ensure_dir(calibration_dir)

    width = 360
    height = 240
    frame_count = 12

    intrinsics = {
        "image_width": width,
        "image_height": height,
        "camera_matrix": [[260.0, 0.0, width / 2.0], [0.0, 260.0, height / 2.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "reprojection_error": 0.0,
    }
    extrinsics = {
        "rotation_matrix": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "translation_vector": [0.0, 0.85, 0.0],
        "reprojection_error_px": 0.0,
        "delta_z_m": 0.4,
    }
    _save_yaml_like_json(calibration_dir / "intrinsics_demo.yaml", intrinsics)
    _save_yaml_like_json(calibration_dir / "extrinsics_demo.yaml", extrinsics)

    intrinsic_images = _ensure_dir(calibration_dir / "intrinsic_images")
    board = _make_chessboard(rows=6, cols=9, square_px=28)
    for idx in range(5):
        canvas = np.full((height, width, 3), 127, dtype=np.uint8)
        angle = -10 + idx * 5
        mat = cv2.getRotationMatrix2D((board.shape[1] // 2, board.shape[0] // 2), angle, 1.0 - idx * 0.03)
        warped = cv2.warpAffine(board, mat, (board.shape[1], board.shape[0]), borderValue=(127, 127, 127))
        y0 = 20 + idx * 4
        x0 = 40 + idx * 10
        y1 = min(height, y0 + warped.shape[0])
        x1 = min(width, x0 + warped.shape[1])
        canvas[y0:y1, x0:x1] = warped[: y1 - y0, : x1 - x0]
        cv2.imwrite(str(intrinsic_images / f"board_{idx:02d}.png"), canvas)

    extrinsic_board = _make_chessboard(rows=6, cols=9, square_px=30)
    ext_canvas = np.full((height, width, 3), 127, dtype=np.uint8)
    ext_canvas[20 : 20 + extrinsic_board.shape[0], 30 : 30 + extrinsic_board.shape[1]] = extrinsic_board
    cv2.imwrite(str(calibration_dir / "extrinsic_board.png"), ext_canvas)

    motion_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    dt = 0.2
    forward_step = 0.08
    speed_mps = forward_step / dt

    for i in range(frame_count):
        frame = np.full((height, width, 3), 55, dtype=np.uint8)
        for row in range(height // 2, height):
            shade = int(70 + (row - height // 2) * 0.70)
            frame[row, :] = (shade, shade, shade)

        horizon = height // 2 + 10
        cv2.line(frame, (0, horizon), (width, horizon), (100, 100, 100), 2)
        cv2.line(frame, (128 - 2 * i, horizon), (94, height - 1), (170, 170, 170), 2)
        cv2.line(frame, (232 + 2 * i, horizon), (266, height - 1), (170, 170, 170), 2)

        pothole_y = 170 + i * 2
        pothole_h = 20
        cv2.rectangle(frame, (145, pothole_y), (215, pothole_y + pothole_h), (12, 12, 12), -1)
        cv2.rectangle(frame, (145, pothole_y), (215, pothole_y + pothole_h), (230, 230, 230), 1)

        curb_y = 138 + i
        curb_h = 16
        curb_x = 250
        cv2.rectangle(frame, (curb_x, curb_y), (332, curb_y + curb_h), (235, 235, 235), -1)
        cv2.rectangle(frame, (curb_x, curb_y), (332, curb_y + curb_h), (30, 30, 30), 1)

        for _ in range(80):
            x = int(rng.integers(0, width))
            y = int(rng.integers(height // 2, height))
            val = int(rng.integers(70, 125))
            frame[y, x] = (val, val, val)

        frame_id = f"frame_{i:03d}"
        cv2.imwrite(str(frames_dir / f"{frame_id}.png"), frame)

        timestamp = i * dt
        motion_rows.append(
            {
                "frame_id": frame_id,
                "timestamp_s": timestamp,
                "speed_mps": speed_mps if i > 0 else 0.0,
                "forward_displacement_m": forward_step if i > 0 else 0.0,
            }
        )

        gt_rows.append(
            {
                "frame_id": frame_id,
                "object_id": "neg_demo_0",
                "label": "negative",
                "risk_gt": "omega2",
                "h_gt_m": 0.08,
                "w_gt_m": 0.30,
                "d_gt_m": max(0.7, 2.6 - i * 0.15),
            }
        )
        gt_rows.append(
            {
                "frame_id": frame_id,
                "object_id": "pos_demo_0",
                "label": "positive",
                "risk_gt": "omega1",
                "h_gt_m": 0.05,
                "w_gt_m": 0.28,
                "d_gt_m": max(0.9, 3.1 - i * 0.12),
            }
        )

    pd.DataFrame(motion_rows).to_csv(sequence_dir / "motion.csv", index=False)
    pd.DataFrame(gt_rows).to_csv(sequence_dir / "gt_obstacles.csv", index=False)
    with (sequence_dir / "gt_plane.json").open("w", encoding="utf-8") as handle:
        json.dump({"normal": [1.0, 0.0, 0.0], "offset": 0.0}, handle, indent=2)

    return {
        "sequence_dir": str(sequence_dir),
        "calibration_dir": str(calibration_dir),
        "frame_count": frame_count,
    }
