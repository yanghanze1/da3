from __future__ import annotations

# 匯入 argparse，建立命令列介面。
import argparse
# 匯入 Path，處理輸入輸出路徑。
from pathlib import Path
# 匯入 Any，讓輸出 payload 型別更彈性。
from typing import Any

# 匯入設定載入與路徑展開工具。
from .config import load_config, resolve_path
# 匯入基礎 I/O 工具。
from .io_utils import ensure_dir, load_yaml, save_json, save_yaml


def _import_or_raise(module_name: str, symbol_name: str):
    """延後匯入指定模組與符號，避免 CLI 啟動時載入全部依賴。"""

    # 動態匯入目標模組。
    module = __import__(module_name, fromlist=[symbol_name])
    # 取出指定函式或類別並回傳。
    return getattr(module, symbol_name)


def build_parser() -> argparse.ArgumentParser:
    """建立新版論文對齊 CLI。"""

    # 建立最上層 parser。
    parser = argparse.ArgumentParser(description="Thesis-aligned monocular experiment CLI")
    # 建立子命令容器。
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 建立相機內參標定命令。
    intrinsic = subparsers.add_parser("calibrate-intrinsics", help="Calibrate camera intrinsics with chessboard images.")
    # 指定棋盤格影像資料夾。
    intrinsic.add_argument("--image-dir", required=True)
    # 指定內角點列數。
    intrinsic.add_argument("--rows", type=int, default=6)
    # 指定內角點行數。
    intrinsic.add_argument("--cols", type=int, default=9)
    # 指定棋盤格單格實體尺寸。
    intrinsic.add_argument("--square-size-m", type=float, default=0.025)
    # 指定輸出檔案。
    intrinsic.add_argument("--output", required=True)

    # 建立相機外參標定命令。
    extrinsic = subparsers.add_parser("calibrate-extrinsics", help="Calibrate camera extrinsics using chessboard and known delta-z.")
    # 指定外參標定影像。
    extrinsic.add_argument("--image-path", required=True)
    # 指定內參檔。
    extrinsic.add_argument("--intrinsics", required=True)
    # 指定內角點列數。
    extrinsic.add_argument("--rows", type=int, default=6)
    # 指定內角點行數。
    extrinsic.add_argument("--cols", type=int, default=9)
    # 指定棋盤格單格實體尺寸。
    extrinsic.add_argument("--square-size-m", type=float, default=0.025)
    # 指定已知前向位移 delta-z。
    extrinsic.add_argument("--delta-z-m", type=float, required=True)
    # 指定輸出檔案。
    extrinsic.add_argument("--output", required=True)

    # 建立 demo 資料序列生成命令。
    make_demo = subparsers.add_parser("make-demo-sequence", help="Generate thesis-style demo sequence.")
    # 指定設定檔。
    make_demo.add_argument("--config", required=True)
    # 指定資料集根目錄。
    make_demo.add_argument("--dataset-root", required=True)
    # 指定序列名稱。
    make_demo.add_argument("--sequence-id", required=True)

    # 建立執行整條管線命令。
    run = subparsers.add_parser("run-pipeline", help="Run the thesis-aligned pipeline for a sequence.")
    # 指定設定檔。
    run.add_argument("--config", required=True)
    # 指定資料集根目錄。
    run.add_argument("--dataset-root", required=True)
    # 指定序列名稱。
    run.add_argument("--sequence-id", required=True)
    # 指定輸出資料夾。
    run.add_argument("--output-dir", required=True)
    # 指定控制後端模式。
    run.add_argument("--control-backend", choices=["noop", "log"], default=None)

    # 建立評估命令。
    evaluate = subparsers.add_parser("evaluate-pipeline", help="Evaluate pipeline outputs against ground truth.")
    # 指定設定檔。
    evaluate.add_argument("--config", required=True)
    # 指定資料集根目錄。
    evaluate.add_argument("--dataset-root", required=True)
    # 指定序列名稱。
    evaluate.add_argument("--sequence-id", required=True)
    # 指定輸出資料夾。
    evaluate.add_argument("--output-dir", required=True)

    # 回傳設定完成的 parser。
    return parser


def _save_payload(path: str | Path, payload: dict[str, Any]) -> None:
    """依副檔名決定使用 JSON 或 YAML 寫出結果。"""

    # 若輸出路徑副檔名為 json，就寫成 JSON。
    if str(path).lower().endswith(".json"):
        save_json(path, payload)
    else:
        # 其他情況預設寫成 YAML。
        save_yaml(path, payload)


def _handle_calibrate_intrinsics(args: argparse.Namespace) -> int:
    """處理內參標定命令。"""

    # 動態匯入內參標定函式。
    calibrate_intrinsics = _import_or_raise("monocular_experiment.calibration", "calibrate_intrinsics")
    # 執行標定。
    payload = calibrate_intrinsics(args.image_dir, args.rows, args.cols, args.square_size_m)
    # 確保輸出資料夾存在。
    ensure_dir(Path(args.output).parent)
    # 將結果寫出。
    _save_payload(args.output, payload)
    # 成功回傳 0。
    return 0


def _handle_calibrate_extrinsics(args: argparse.Namespace) -> int:
    """處理外參標定命令。"""

    # 動態匯入外參標定函式。
    calibrate_extrinsics = _import_or_raise("monocular_experiment.calibration", "calibrate_extrinsics")
    # 先讀取內參檔。
    intrinsics = load_yaml(args.intrinsics)
    # 執行外參標定。
    payload = calibrate_extrinsics(
        image_path=args.image_path,
        intrinsics=intrinsics,
        rows=args.rows,
        cols=args.cols,
        square_size_m=args.square_size_m,
        delta_z_m=args.delta_z_m,
    )
    # 確保輸出資料夾存在。
    ensure_dir(Path(args.output).parent)
    # 將結果寫出。
    _save_payload(args.output, payload)
    # 成功回傳 0。
    return 0


def _handle_make_demo_sequence(args: argparse.Namespace) -> int:
    """處理 demo 資料序列生成命令。"""

    # 動態匯入 demo 序列生成函式。
    make_demo_sequence = _import_or_raise("monocular_experiment.demo_data", "make_demo_sequence")
    # 載入設定檔。
    config = load_config(args.config)
    # 由設定檔位置展開 calibration 目錄。
    calibration_dir = resolve_path(config, "calibration")
    # 生成 demo 序列與對應標定檔。
    make_demo_sequence(
        dataset_root=args.dataset_root,
        sequence_id=args.sequence_id,
        calibration_dir=calibration_dir,
    )
    # 成功回傳 0。
    return 0


def _handle_run_pipeline(args: argparse.Namespace) -> int:
    """處理執行完整管線命令。"""

    # 動態匯入管線執行函式。
    run_pipeline = _import_or_raise("monocular_experiment.pipeline", "run_pipeline")
    # 載入設定檔。
    config = load_config(args.config)
    # 優先採用命令列指定控制後端，否則回退設定檔值。
    backend = args.control_backend or config.get("runtime", {}).get("control_backend", "log")
    # 執行管線。
    run_pipeline(
        config=config,
        dataset_root=args.dataset_root,
        sequence_id=args.sequence_id,
        output_dir=args.output_dir,
        control_backend=backend,
    )
    # 成功回傳 0。
    return 0


def _handle_evaluate_pipeline(args: argparse.Namespace) -> int:
    """處理評估命令。"""

    # 動態匯入評估函式。
    evaluate_pipeline = _import_or_raise("monocular_experiment.evaluation", "evaluate_pipeline")
    # 載入設定檔。
    config = load_config(args.config)
    # 執行評估。
    evaluate_pipeline(
        dataset_dir=Path(args.dataset_root) / args.sequence_id,
        output_dir=args.output_dir,
        config=config,
    )
    # 成功回傳 0。
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。"""

    # 建立命令列 parser。
    parser = build_parser()
    # 解析命令列參數。
    args = parser.parse_args(argv)

    # 依子命令分派到對應處理函式。
    if args.command == "calibrate-intrinsics":
        return _handle_calibrate_intrinsics(args)
    # 外參標定命令。
    if args.command == "calibrate-extrinsics":
        return _handle_calibrate_extrinsics(args)
    # demo 序列生成命令。
    if args.command == "make-demo-sequence":
        return _handle_make_demo_sequence(args)
    # 管線執行命令。
    if args.command == "run-pipeline":
        return _handle_run_pipeline(args)
    # 評估命令。
    if args.command == "evaluate-pipeline":
        return _handle_evaluate_pipeline(args)
    # 理論上不會走到這裡；若走到代表有未知命令。
    parser.error(f"Unknown command: {args.command}")
    # 以標準 CLI 錯誤代碼回傳。
    return 2
