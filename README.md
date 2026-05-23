# `code2`：DA3 + PIDNet 論文對齊版單眼幾何管線

本專案依 [`new/第四章_v8.md`](D:/agent_lunwen/new/第四章_v8.md) 與 [`new/第五章_v6.md`](D:/agent_lunwen/new/第五章_v6.md) 重新整理為新版資料流：

```text
I_t
 -> DA3-SMALL relative depth
 -> Road ∪ Sidewalk mask
 -> scale alignment
 -> road point set S_t^road
 -> plane model Π_t
 -> candidate regions B_t / S_{t,k}^{cand}
 -> cross-frame summary M_t
 -> tracked objects O_k
 -> Risk(O_k)
```

系統不再使用舊版 `SfM / disparity / Z≈fB/d` 流程。深度來源改為 `Depth Anything 3`，語意來源維持 `PIDNet-S`，再由地平面、候選區域、跨幀延續與風險矩陣完成第五章的物件化與防碰撞判定。

## 專案結構

```text
code2/
  main.py
  README.md
  requirements.txt
  configs/
    default.yaml
  calibration/
  datasets/
  monocular_experiment/
  outputs/
  tests/
  third_party/
    PIDNet/
    Depth-Anything-3/
```

## 安裝

在 [`D:\agent_lunwen\code2`](D:/agent_lunwen/code2) 下安裝本專案依賴：

```powershell
python -m pip install -r requirements.txt
```

若要使用官方 DA3 後端，還需要讓本地 `Depth-Anything-3` repo 可被 Python 匯入，並準備好 `DA3-SMALL` 模型目錄：

1. 官方 repo 位置：[`third_party/Depth-Anything-3`](D:/agent_lunwen/code2/third_party/Depth-Anything-3)
2. 預設模型目錄：[`models/da3/DA3-SMALL`](D:/agent_lunwen/code2/models/da3/DA3-SMALL)
3. 官方 API 採用 `DepthAnything3.from_pretrained(...)`

本專案不做隱式下載。若 `repo_path` 或 `model_dir` 不存在，會直接報錯。

## 設定檔

主設定檔為 [`configs/default.yaml`](D:/agent_lunwen/code2/configs/default.yaml)，主要區塊如下：

- `calibration`
- `depth_model`
- `roi`
- `scale_alignment`
- `plane`
- `candidate_generation`
- `tracking`
- `risk`
- `evaluation`
- `runtime`
- `visualization`

其中 `depth_model` 的預設設定如下：

```yaml
depth_model:
  backend: tensorrt
  repo_path: third_party/Depth-Anything-3
  model_name: DA3-SMALL
  model_dir: models/da3/DA3-SMALL
  process_res: 392
  process_res_method: upper_bound_resize
  artifacts_dir: models/da3/artifacts
  build_engine_if_missing: true
  precision: fp16
  onnx_opset: 18
  workspace_size_gb: 4.0
```

測試與 smoke run 可把 `backend` 改成 `mock`，用來跳過大型模型依賴。

## TensorRT Backend

When `depth_model.backend` is set to `tensorrt`, the project:

- preprocesses the frame with the official local `Depth-Anything-3` Python API,
- exports a fixed-shape ONNX model for that DA3 input size,
- builds a TensorRT engine on first use,
- caches the generated `.onnx` and `.engine` files under `models/da3/artifacts`,
- reuses the cached engine for later frames with the same preprocessed shape.

If `onnx` is installed in a short custom path on Windows, place it under `third_party/python_pkgs` or add the folder through `depth_model.extra_python_paths`.

## PIDNet Acceleration

The ROI frontend now prefers `PIDNet-S` TensorRT inference and falls back to ONNX before using the heuristic fallback frontend.

Recommended config:

```yaml
roi:
  backend: auto
  official_pidnet_repo_path: third_party/PIDNet
  official_pidnet_weights: models/pidnet/PIDNet_S_Cityscapes_test.pt
  artifacts_dir: models/pidnet/artifacts
  build_engine_if_missing: true
  precision: fp16
  onnx_opset: 18
  workspace_size_gb: 2.0
  onnx_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
  input_size: [1024, 512]
```

Both the exported PIDNet ONNX file and the TensorRT engine are cached under `models/pidnet/artifacts`.

## 命令列介面

```powershell
python -m monocular_experiment --help
```

### 1. 內參標定

```powershell
python -m monocular_experiment calibrate-intrinsics `
  --image-dir calibration\intrinsic_images `
  --rows 6 `
  --cols 9 `
  --square-size-m 0.025 `
  --output calibration\intrinsics_demo.yaml
```

### 2. 外參標定

```powershell
python -m monocular_experiment calibrate-extrinsics `
  --image-path calibration\extrinsic_board.png `
  --intrinsics calibration\intrinsics_demo.yaml `
  --rows 6 `
  --cols 9 `
  --square-size-m 0.025 `
  --delta-z-m 0.4 `
  --output calibration\extrinsics_demo.yaml
```

### 3. 產生 demo 序列

```powershell
python -m monocular_experiment make-demo-sequence `
  --config configs\default.yaml `
  --dataset-root datasets `
  --sequence-id demo_sequence
```

### 4. 執行完整管線

```powershell
python -m monocular_experiment run-pipeline `
  --config configs\default.yaml `
  --dataset-root datasets `
  --sequence-id demo_sequence `
  --output-dir outputs\demo_run_v3 `
  --control-backend log
```

### 5. 評估管線

```powershell
python -m monocular_experiment evaluate-pipeline `
  --config configs\default.yaml `
  --dataset-root datasets `
  --sequence-id demo_sequence `
  --output-dir outputs\demo_run_v3
```

## 輸入資料契約

每段實驗序列採用：

```text
datasets/<sequence_id>/
  frames/
  motion.csv
  gt_obstacles.csv
  gt_plane.json
```

### `motion.csv`

必要欄位：

```text
frame_id,timestamp_s
```

選用欄位：

```text
speed_mps,forward_displacement_m
```

### `gt_obstacles.csv`

必要欄位：

```text
frame_id,object_id,label,risk_gt,h_gt_m,w_gt_m,d_gt_m
```

## 主要輸出契約

### `frame_states.jsonl`

每列至少包含：

- `frame_id`
- `depth_stats`
- `scale_alignment`
- `road_mask_stats`
- `plane_model`
- `roi_candidates`
- `candidate_clusters`
- `cross_frame_matches`
- `tracked_objects`
- `risk_events`
- `timing_ms`

### `evaluation_summary.json`

主要評估欄位：

- `detection_metrics`
- `geometry_metrics`
- `plane_metrics`
- `scale_metrics`
  - 其中包含 `scale_factor_stability`
- `tracking_metrics`
  - 其中包含 `tracking_continuity`
- `latency_metrics`

## 測試

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

smoke test 會把 `depth_model.backend` 改成 `mock`，因此不需要下載 DA3 權重也能驗證新資料流與新欄位契約。
