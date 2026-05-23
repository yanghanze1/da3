# 從同套件的主程式模組匯入主函式。
from .main import main


# 當使用 `python -m monocular_experiment` 執行時，進入這個分支。
if __name__ == "__main__":
    # 將主函式的結果包成 SystemExit，作為命令列程式的結束代碼。
    raise SystemExit(main())
