from __future__ import annotations

# 匯入 Path，處理影像資料夾路徑。
from pathlib import Path
# 匯入 Any，讓標定結果字典可帶彈性欄位。
from typing import Any

# 匯入 OpenCV，執行棋盤格偵測與相機標定。
import cv2
# 匯入 NumPy，建立棋盤格三維點。
import numpy as np


# 定義可接受的校正影像副檔名。
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(directory: str | Path) -> list[Path]:
    """列出標定資料夾中所有合法影像檔。"""

    # 轉成 Path 物件。
    root = Path(directory)
    # 若資料夾不存在就直接報錯。
    if not root.exists():
        raise FileNotFoundError(f"Calibration directory does not exist: {root}")
    # 篩選合法影像並依檔名排序。
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])


def _board_object_points(rows: int, cols: int, square_size_m: float) -> np.ndarray:
    """
    建立位於 z=0 平面上的棋盤格角點三維座標。
    rows 與 cols 指的是內角點數，不是方格數。
    """

    # 預先建立 N x 3 座標陣列。
    points = np.zeros((rows * cols, 3), np.float32)
    # 生成棋盤格在平面上的規則網格。
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # 把 x、y 座標乘上實體方格尺寸，得到公尺尺度。
    points[:, :2] = grid * square_size_m
    # 回傳角點三維座標。
    return points


def calibrate_intrinsics(
    image_dir: str | Path,
    rows: int,
    cols: int,
    square_size_m: float,
) -> dict[str, Any]:
    """
    使用 Zhang 風格的棋盤格方法估計相機內參。
    回傳值為可直接序列化的字典。
    """

    # 列出所有標定影像。
    image_paths = _list_images(image_dir)
    # 若沒有影像則無法標定。
    if not image_paths:
        raise ValueError(f"No calibration images found in: {image_dir}")

    # 初始化所有影像對應的棋盤格三維點集合。
    object_points: list[np.ndarray] = []
    # 初始化所有影像對應的二維角點集合。
    image_points: list[np.ndarray] = []
    # 預先建立單張棋盤格的理論三維點。
    board_points = _board_object_points(rows, cols, square_size_m)
    # 記錄影像尺寸。
    image_shape = None

    # 逐張走訪標定影像。
    for image_path in image_paths:
        # 讀取彩色影像。
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        # 若讀不到檔案就跳過。
        if image is None:
            continue
        # 轉為灰階以便棋盤格偵測。
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 嘗試偵測內角點。
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        # 若此張未偵測到棋盤格，就略過。
        if not found:
            continue

        # 對偵測到的角點做亞像素精修。
        refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
        # 保存此張影像對應的三維棋盤格點。
        object_points.append(board_points.copy())
        # 保存此張影像對應的二維角點。
        image_points.append(refined)
        # 記住影像寬高，供標定函式使用。
        image_shape = gray.shape[::-1]

    # 若完全沒有有效偵測，無法做標定。
    if not object_points or image_shape is None:
        raise ValueError("No valid chessboard detections for intrinsic calibration.")

    # 呼叫 OpenCV 相機標定函式。
    reproj_error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_shape,
        None,
        None,
    )

    # 回傳可序列化結果。
    return {
        "image_width": int(image_shape[0]),
        "image_height": int(image_shape[1]),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "reprojection_error": float(reproj_error),
        "board": {"rows": rows, "cols": cols, "square_size_m": float(square_size_m)},
    }


def calibrate_extrinsics(
    image_path: str | Path,
    intrinsics: dict[str, Any],
    rows: int,
    cols: int,
    square_size_m: float,
    delta_z_m: float,
) -> dict[str, Any]:
    """
    使用棋盤格加上已知前向位移 delta-z 估計相機外參。
    核心方法是 solvePnP。
    """

    # 讀取外參標定影像。
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    # 若讀不到檔案就報錯。
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    # 轉灰階以利角點偵測。
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 偵測棋盤格角點。
    found, corners = cv2.findChessboardCorners(gray, (cols, rows))
    # 若找不到棋盤格就中止。
    if not found:
        raise ValueError("Chessboard was not detected in extrinsic calibration image.")

    # 以亞像素精修角點位置。
    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
    )

    # 建立棋盤格理論三維點。
    object_points = _board_object_points(rows, cols, square_size_m)
    # 將所有點沿 Z 軸加上已知前向位移。
    object_points[:, 2] += float(delta_z_m)

    # 讀取內參矩陣。
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    # 讀取畸變係數。
    dist_coeffs = np.asarray(intrinsics.get("dist_coeffs", []), dtype=np.float64)

    # 使用 solvePnP 解出旋轉與平移。
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    # 若 PnP 解算失敗，就中止。
    if not success:
        raise ValueError("solvePnP failed to estimate extrinsics.")

    # 把旋轉向量轉為旋轉矩陣。
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # 將三維點再投影回影像，用於計算重投影誤差。
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    # 攤平成 N x 2。
    projected = projected.reshape(-1, 2)
    # 把觀測角點也攤平成 N x 2。
    observed = corners.reshape(-1, 2)
    # 以每個角點的均方根投影誤差作為重投影誤差。
    reprojection_error_px = float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))

    # 回傳外參結果。
    return {
        "rotation_matrix": rotation_matrix.tolist(),
        "translation_vector": tvec.ravel().tolist(),
        "reprojection_error_px": reprojection_error_px,
        "delta_z_m": float(delta_z_m),
        "board": {"rows": rows, "cols": cols, "square_size_m": float(square_size_m)},
    }
