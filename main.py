"""專案根入口：將命令列執行導向套件內真正的主程式。"""

# 從套件內匯入主函式。
from monocular_experiment.main import main


# 當此檔案被直接執行時，呼叫主函式並將回傳值作為結束代碼。
if __name__ == "__main__":
    # 使用 SystemExit 將 main() 的整數結果回傳給作業系統。
    raise SystemExit(main())
