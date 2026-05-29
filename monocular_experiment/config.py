from __future__ import annotations

# 匯入路徑處理工具，用來解析設定檔位置。
from pathlib import Path
# 匯入 Any 以標註可接收多型別的設定內容。
from typing import Any

# 匯入 YAML 套件，用來讀取設定檔。
import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_payload(config_path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = config_path.resolve()
    if resolved in seen:
        raise ValueError(f"Circular config _base reference: {resolved}")
    seen.add(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    base_paths = payload.pop("_base", payload.pop("base_config", []))
    if isinstance(base_paths, (str, Path)):
        base_paths = [base_paths]
    merged: dict[str, Any] = {}
    for base_path in base_paths or []:
        base = Path(base_path)
        if not base.is_absolute():
            base = resolved.parent / base
        merged = _deep_merge(merged, _load_config_payload(base, seen))
    seen.remove(resolved)
    return _deep_merge(merged, payload)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """讀取 YAML 設定檔，並補上設定檔本身所在位置資訊。"""

    # 先把輸入路徑轉成絕對路徑，避免後續相對路徑解析混亂。
    resolved = Path(config_path).resolve()
    payload = _load_config_payload(resolved, set())
    # 記錄設定檔完整路徑，供其他模組後續查用。
    payload["_config_path"] = str(resolved)
    # 記錄設定檔所在資料夾，供相對路徑展開時使用。
    payload["_config_dir"] = str(resolved.parent)
    # 回傳補完後的設定內容。
    return payload


def nested_get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """安全地逐層讀取巢狀字典中的欄位。"""

    # 從最外層設定開始逐步往下取值。
    current: Any = payload
    # 依序走訪每一個鍵名。
    for key in keys:
        # 若目前節點不是字典或缺少目標鍵，就直接回傳預設值。
        if not isinstance(current, dict) or key not in current:
            return default
        # 進入下一層節點。
        current = current[key]
    # 成功找到最終值時回傳之。
    return current


def resolve_path(config: dict[str, Any], path_value: str | Path) -> Path:
    """依照設定檔位置，把相對路徑展開成專案中的絕對路徑。"""

    # 把輸入值轉成 Path 物件，便於後續判斷。
    candidate = Path(path_value)
    # 如果本來就是絕對路徑，直接回傳即可。
    if candidate.is_absolute():
        return candidate
    # 取出設定檔所在資料夾。
    config_dir = Path(config["_config_dir"])
    # 依專案結構，設定檔上一層視為 code 根目錄。
    code_root = config_dir.parent
    # 將相對路徑掛到 code 根目錄並轉成標準絕對路徑。
    return (code_root / candidate).resolve()
