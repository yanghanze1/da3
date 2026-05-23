from __future__ import annotations

# 匯入 JSON 模組，供讀寫 JSON 與 JSONL 使用。
import json
# 匯入 Path 進行路徑處理。
from pathlib import Path
# 匯入型別工具。
from typing import Any, TYPE_CHECKING

# 匯入 YAML 模組。
import yaml

# 僅在型別檢查時匯入 pandas，避免執行期不必要依賴。
if TYPE_CHECKING:
    import pandas as pd

# 定義可視為影格輸入的副檔名集合。
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def ensure_dir(path: str | Path) -> Path:
    """確保指定資料夾存在，若不存在就建立。"""

    # 統一路徑型別。
    directory = Path(path)
    # 遞迴建立資料夾；若已存在則忽略。
    directory.mkdir(parents=True, exist_ok=True)
    # 回傳建立後的 Path 物件，方便鏈式使用。
    return directory


def list_frame_paths(frames_dir: str | Path) -> list[Path]:
    """列出影格資料夾中的所有合法影像檔，並依檔名排序。"""

    # 先取得影格資料夾路徑。
    root = Path(frames_dir)
    # 若資料夾不存在，直接回傳空清單。
    if not root.exists():
        return []
    # 篩出檔案型態正確的影像，並按名稱排序後回傳。
    return sorted([item for item in root.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES])


def read_image(path: str | Path):
    """以彩色模式讀取單張影像。"""

    # 延後匯入 cv2，減少模組初始負擔。
    import cv2

    # 讀取彩色影像。
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    # 若回傳 None，表示檔案不存在或格式不支援。
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    # 回傳 BGR 影像陣列。
    return image


def read_motion_csv(path: str | Path):
    """讀取 motion.csv 並檢查新版最小欄位契約。"""

    # 延後匯入 pandas。
    import pandas as pd

    # 讀取整張表格。
    frame = pd.read_csv(path)
    # 定義最小必要欄位集合。
    required = {"frame_id", "timestamp_s"}
    # 找出缺漏欄位。
    missing = required - set(frame.columns)
    # 若缺欄則中止。
    if missing:
        raise ValueError(f"motion.csv missing required columns: {sorted(missing)}")
    # 回傳已驗證的資料表。
    return frame


def read_gt_obstacles(path: str | Path):
    """讀取障礙物真值表並檢查新版欄位契約。"""

    # 延後匯入 pandas。
    import pandas as pd

    # 載入真值 CSV。
    frame = pd.read_csv(path)
    # 宣告必要欄位。
    required = {"frame_id", "object_id", "label", "risk_gt", "h_gt_m", "w_gt_m", "d_gt_m"}
    # 檢查是否缺漏。
    missing = required - set(frame.columns)
    # 若有缺漏就報錯。
    if missing:
        raise ValueError(f"gt_obstacles.csv missing required columns: {sorted(missing)}")
    # 回傳已驗證的表格。
    return frame


def read_gt_plane(path: str | Path) -> dict[str, Any]:
    """讀取地平面真值 JSON。"""

    # 開啟 JSON 檔。
    with Path(path).open("r", encoding="utf-8") as handle:
        # 解析並回傳字典。
        return json.load(handle)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """讀取 YAML 檔案。"""

    # 以 UTF-8 開啟檔案。
    with Path(path).open("r", encoding="utf-8") as handle:
        # 空檔則回傳空字典。
        return yaml.safe_load(handle) or {}


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """將字典寫成 YAML 檔。"""

    # 先取得輸出檔路徑。
    output = Path(path)
    # 確保父資料夾存在。
    ensure_dir(output.parent)
    # 以寫入模式開檔。
    with output.open("w", encoding="utf-8") as handle:
        # 依原欄位順序輸出 YAML。
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def save_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    """將物件寫成縮排 JSON。"""

    # 轉成 Path 物件。
    output = Path(path)
    # 確保父資料夾存在。
    ensure_dir(output.parent)
    # 開啟輸出檔。
    with output.open("w", encoding="utf-8") as handle:
        # 輸出人類可讀的 JSON。
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """將多筆字典資料逐列寫成 JSONL。"""

    # 轉成 Path 物件。
    output = Path(path)
    # 確保父資料夾存在。
    ensure_dir(output.parent)
    # 開啟輸出檔案。
    with output.open("w", encoding="utf-8") as handle:
        # 逐列走訪每筆資料。
        for row in rows:
            # 每列都寫成一行 JSON。
            handle.write(json.dumps(row, ensure_ascii=True))
            # 補上換行符號，形成 JSONL 格式。
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """讀取 JSONL 檔並回傳字典清單。"""

    # 準備接收所有資料列。
    rows: list[dict[str, Any]] = []
    # 開啟輸入檔案。
    with Path(path).open("r", encoding="utf-8") as handle:
        # 逐行讀取。
        for line in handle:
            # 去除前後空白。
            content = line.strip()
            # 只處理非空白行。
            if content:
                # 解析成字典並加入結果。
                rows.append(json.loads(content))
    # 回傳完整列表。
    return rows
