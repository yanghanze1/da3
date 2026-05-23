from __future__ import annotations

# 匯入 dataclass，讓控制事件資料結構更精簡。
from dataclasses import dataclass
# 匯入 Path，便於寫出事件紀錄檔案。
from pathlib import Path


@dataclass
class ControlEvent:
    """描述單次風險控制事件的最小欄位集合。"""

    # 事件發生的影格編號。
    frame_id: str
    # 本次控制決策，例如 warning 或 intervention。
    decision: str
    # 對應的追蹤物件編號。
    object_id: str


class NoopControlBackend:
    """不做任何輸出的空白控制後端。"""

    def emit(self, _event: ControlEvent) -> None:
        """忽略控制事件，通常用於測試或停用紀錄時。"""

        # 明確回傳 None，表示此後端不產生任何副作用。
        return None


class LogControlBackend:
    """將控制事件以 CSV 形式附加寫入檔案的後端。"""

    def __init__(self, log_path: str | Path):
        """記住事件日誌輸出位置。"""

        # 將路徑統一轉為 Path 物件，方便後續建立資料夾與寫檔。
        self.log_path = Path(log_path)

    def emit(self, event: ControlEvent) -> None:
        """把單次控制事件附加到風險事件紀錄檔。"""

        # 若上層資料夾不存在就先建立。
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # 以附加模式開啟日誌檔。
        with self.log_path.open("a", encoding="utf-8") as handle:
            # 依固定欄位順序寫入一列 CSV。
            handle.write(f"{event.frame_id},{event.object_id},{event.decision}\n")
