# code2 整體實驗流程說明

本文描述 `code2` 目前程式實作下的單眼深度、語意遮罩、幾何估計、候選障礙物、跨幀追蹤、風險評估與評估輸出的完整實驗流程。內容只以程式碼與最後完成的實驗輸出為依據。

最後完成的代表性實驗目錄：

```text
outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/
```

本文中的圖像路徑皆為建議放置位置；若圖片尚未產生，可先作為 placeholder，後續可從最後實驗的 `overlays/`、`plots/` 或重新繪製的說明圖補上。

![整體實驗流程](outputs/docs_figures/01_workflow_overview.png)

## 1. 整體資料流

目前實驗流程以逐幀影像 `I_t` 為起點，將單張影像、相機標定與車體運動資訊串接成下列資料流：

```text
影像幀 I_t
  -> 語意前端產生 road / analysis / obstacle analysis mask
  -> DA3 產生 relative depth
  -> 以 road mask 與相機高度做 scale alignment
  -> absolute depth + intrinsics/extrinsics 反投影為 3D points
  -> road points 擬合地平面
  -> residual / depth-gradient ROI 產生候選區
  -> ROI 內異常點 DBSCAN 聚類成 candidate observations
  -> 跨幀 matching / tracking / temporal snapshot
  -> risk assessment / risk event emission
  -> frame_states、overlays、summary、evaluation plots
```

步驟之間的連接關係如下：

1. `frames/` 提供影像，`motion.csv` 提供時間與前向運動資訊。
2. 語意前端從影像得到 road mask 與 obstacle analysis mask。
3. 深度模型從同一張影像得到 relative depth。
4. 尺度對齊使用 road mask、relative depth、相機內參與相機高度，把 relative depth 轉為 metric-like absolute depth。
5. 幾何反投影使用 absolute depth、內參與外參，把 road / candidate pixels 轉成 3D world points。
6. road world points 擬合地平面，地平面再提供 candidate residual 與 expected depth 的基準。
7. residual ROI、depth-gradient ROI 與 frontend ROI 合併後，成為 candidate generation 的搜尋區域。
8. candidate generation 將 ROI 內異常 3D 點聚類，產生 obstacle candidate。
9. tracking 將當前 candidates 與前一幀 candidates / tracks 連接成持續物件。
10. risk assessment 以 tracked object、道路點與車速計算風險，必要時輸出 risk event。
11. 每幀結果寫入 `frame_states.jsonl`，同時輸出 overlay 圖；整段序列再由 evaluation 產生 metrics 與 plots。

主要程式來源：

- `monocular_experiment/main.py`：命令列入口。
- `monocular_experiment/pipeline.py`：逐幀 pipeline 主流程。
- `monocular_experiment/evaluation.py`：評估流程與評估輸出。
- `monocular_experiment/models.py`：輸出資料結構。

## 2. 命令列入口與實驗階段

`monocular_experiment/main.py` 定義五個實驗入口。它們不是彼此都必須在每次實驗中重新執行，但形成完整實驗生命週期。

| 階段 | CLI 子命令 | 程式入口 | 主要輸入 | 主要輸出 |
|---|---|---|---|---|
| 內參標定 | `calibrate-intrinsics` | `_handle_calibrate_intrinsics()` | 棋盤格影像、rows/cols、square size | intrinsics YAML/JSON |
| 外參標定 | `calibrate-extrinsics` | `_handle_calibrate_extrinsics()` | 外參棋盤格影像、內參、`delta_z_m` | extrinsics YAML/JSON |
| demo 序列 | `make-demo-sequence` | `_handle_make_demo_sequence()` | config、dataset root、sequence id | demo frames、motion、GT、calibration |
| 執行管線 | `run-pipeline` | `_handle_run_pipeline()` -> `run_pipeline()` | sequence、config、calibration、output dir | `frame_states.jsonl`、overlays、summary、risk events |
| 評估管線 | `evaluate-pipeline` | `_handle_evaluate_pipeline()` -> `evaluate_pipeline()` | GT、`frame_states.jsonl`、config | `evaluation_summary.json`、plots、可選 tables/diagnostics |

連接方式：

1. 標定階段產生 intrinsics / extrinsics。
2. 資料序列階段提供 frames、motion 與 GT。
3. `run-pipeline` 同時讀取資料序列、標定與 config，產生逐幀預測與診斷。
4. `evaluate-pipeline` 讀取 `run-pipeline` 的輸出與 GT，產生整體評估。

## 3. 最後完成實驗輸出對照

代表性最後實驗目錄：

```text
outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/
  frame_states.jsonl
  pipeline_summary.json
  evaluation_summary.json
  risk_events.csv
  overlays/*.png
  plots/risk_confusion_matrix.png
  plots/geometry_error_scatter.png
  plots/latency_histogram.png
```

### 3.1 Run summary

`pipeline_summary.json` 記錄整段序列層級的結果：

| 欄位 | 最後實驗值 | 意義 |
|---|---:|---|
| `sequence_id` | `laf_raw_02hk_000000_020_190_eval` | 本次處理的 sequence id |
| `frames_processed` | `18` | 實際處理幀數 |
| `mean_latency_ms` | 約 `4907.00` | 平均逐幀 pipeline latency |
| `max_latency_ms` | 約 `17097.11` | 最大逐幀 pipeline latency |
| `risk_event_count` | `291` | 經 risk event emitter 過濾後輸出的事件數 |

### 3.2 Evaluation summary

`evaluation_summary.json` 記錄評估結果：

| 指標 | 最後實驗值 | 說明 |
|---|---:|---|
| detection precision | 約 `0.9444` | 被預測為 hazard 的幀中，有多少為正確 hazard |
| detection recall | `1.0` | GT hazard 中被偵測到的比例 |
| detection F1 | 約 `0.9714` | precision 與 recall 的 harmonic mean |
| TP / FP / FN / TN | `17 / 1 / 0 / 0` | frame-level detection confusion counts |
| scale factor count | `18` | 有效 scale alignment frame 數 |
| tracking continuity | 約 `0.7105` | continuing objects / active objects |
| temporal attempted / ok / applied | `0 / 0 / 0` | 最後實驗未實際套用 temporal measurement |

### 3.3 Frame-level diagnostics

`frame_states.jsonl` 是最重要的逐幀診斷檔。每一列是一個 frame state，包含：

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
- `temporal_measurements`
- `raw_risk_assessments`

最後實驗第一幀的摘要顯示：

- frame id：`02_Hanns_Klemm_Str_44_000000_000020`
- `roi_candidates`：19 個
- `candidate_clusters`：19 個
- `tracked_objects`：19 個
- `raw_risk_assessments`：19 個
- `risk_events`：0 個，表示第一幀雖有 raw assessment，但尚未通過 event emitter 輸出條件
- `depth_stats.backend` 使用 official DA3 / DA3-SMALL 類型後端
- `plane_model.status` 為 `ok`
- ROI source 以 depth-gradient contour / hybrid 結果為主要代表

### 3.4 最後實驗中沒有實際填充的產物

`evaluate_pipeline()` 程式在有資料時可產生 tables 與 ROI-to-candidate failure overlays；但最後實驗沒有 populated tables。原因可由 `evaluation_summary.json` 對照：

- `geometry_metrics.matched_pairs = 0`
- label `07` 的 `total_gt_frames = 0`
- temporal measurement attempted / ok / applied 皆為 0

因此本文後續提到 tables 或 temporal measurement 時，會區分「程式支援」與「最後實驗實際產出」。

## 4. 輸入資料契約

![輸入資料契約](outputs/docs_figures/02_input_dataset_contract.png)

`run_pipeline()` 以 `dataset_root` 與 `sequence_id` 定位資料序列：

```text
datasets/<sequence_id>/
  frames/
  motion.csv
  gt_obstacles.csv
  gt_plane.json
```

### 4.1 `frames/`

輸入：

- 逐幀影像檔。
- `list_frame_paths(frames_dir)` 會列出可用影像。
- `frame_path.stem` 會成為該幀的 `frame_id`。

輸出到下一步：

- BGR frame array，送入 segmentation、depth inference、overlay rendering。
- `frame_id`，用於所有輸出紀錄、tracking、risk event 與 overlay 命名。

### 4.2 `motion.csv`

輸入欄位：

- 必要：`frame_id`、`timestamp_s`。
- 可選：`speed_mps`、`forward_displacement_m`、`cumulative_distance_m`。

程式連接方式：

1. `read_motion_csv()` 讀入 motion table。
2. `run_pipeline()` 要求 motion row count 必須等於 frame 數。
3. 每一幀讀取當前 row 與前一幀 row。
4. 若 `speed_mps` 已提供且非 NaN，直接作為當前速度。
5. 若 `forward_displacement_m` 已提供且非 NaN，直接作為當前幀位移。
6. 若缺少其中之一，程式可由 timestamp 差與另一個量推導。
7. 若 `cumulative_distance_m` 存在，直接作為累積前進距離；否則累加非負的 `forward_displacement_m`。

中間產物：

- `speed_mps`
- `forward_displacement_m`
- `cumulative_forward_m`
- `timestamp_curr_s`
- `timestamp_prev_s`

這些中間產物會往後傳給 tracking、temporal measurement 與 risk assessment。

### 4.3 Calibration files

`configs/default.yaml` 的 `calibration` 區塊指定：

- `intrinsics_path`
- `extrinsics_path`

`resolve_path()` 會把相對路徑解析成實際檔案路徑。這兩個檔案在 pipeline 中被載入為：

- `intrinsics`：供 backprojection / projection 使用。
- `extrinsics`：供 camera-to-world / world-to-camera 轉換使用。

### 4.4 GT files

`evaluate_pipeline()` 在評估階段讀取：

- `gt_obstacles.csv`
- `gt_plane.json`

它們不參與 `run_pipeline()` 的預測流程，而是用於事後計算 detection、geometry、plane、risk confusion 等指標。

## 5. 設定檔如何控制實驗

`configs/default.yaml` 把 pipeline 各階段參數集中管理。每個區塊與實驗步驟的對應如下：

| Config 區塊 | 控制階段 | 主要作用 |
|---|---|---|
| `calibration` | 輸入載入、反投影、座標轉換 | 指定 intrinsics / extrinsics 路徑 |
| `depth_model` | relative depth inference | 指定 DA3/mock/TensorRT 後端、模型路徑、artifact cache、解析度 |
| `roi` | 語意前端與 processing ROI | 指定 PIDNet/fallback、road class、processing rect、ROI 門檻 |
| `scale_alignment` | 尺度對齊 | 指定 camera height、sampling stride、relative depth 門檻 |
| `plane` | 地平面估計 | 指定 RANSAC iterations、distance threshold、min inliers |
| `candidate_generation` | ROI 與 candidate 生成 | 指定 residual ROI、depth-gradient ROI、DBSCAN 與候選尺寸門檻 |
| `temporal_measurement` | temporal fusion | 指定 keyframe selection、baseline、support point fusion 門檻 |
| `tracking` | object tracking | 指定 anchor distance、IoU weight、miss/reactivation、match score |
| `risk` | 風險評估 | 指定 braking distance、clear width、height/width/risk 門檻 |
| `visualization` | 視覺化輸出 | 控制 overlay / plot 是否保存 |
| `evaluation` | 評估 | 指定距離分箱與 evaluation 輔助設定 |

最後實驗中，temporal measurement 的程式階段存在，但 `evaluation_summary.json` 顯示 attempted / ok / applied 均為 0，表示該次實驗沒有實際套用 temporal measurement 結果。

## 6. 逐幀 pipeline 詳細流程

以下順序對應 `run_pipeline()` 的 per-frame loop。每一幀都會依序完成這些步驟。

## 6.1 讀取 frame 與 motion

### 目的

把資料序列中的影像與車體運動資訊轉成後續可使用的 frame-level input。

### 輸入

- `frames/<frame_id>.*`
- `motion.csv` 當前 row
- 前一幀 motion row

### 處理與數學/程式實作

1. 讀取影像：
   - `read_image(frame_path)` 將影像檔讀成 BGR array。
2. 建立 frame id：
   - `frame_id = frame_path.stem`。
3. 讀取時間：
   - `timestamp_curr_s = motion_curr["timestamp_s"]`。
   - 若不是第一幀，讀取 `timestamp_prev_s`。
4. 推導速度：
   - 若 row 有有效 `speed_mps`，直接使用。
   - 否則可由 `forward_displacement_m / Δt` 推得。
5. 推導前向位移：
   - 若 row 有有效 `forward_displacement_m`，直接使用。
   - 否則可由 `speed_mps * Δt` 推得。
6. 更新累積前進距離：
   - 若有 `cumulative_distance_m`，直接使用。
   - 否則執行 `cumulative_forward_m += max(0, forward_displacement_m)`。

### 中間產物

- `frame`
- `frame_id`
- `timestamp_curr_s`
- `timestamp_prev_s`
- `speed_mps`
- `forward_displacement_m`
- `cumulative_forward_m`

### 輸出與連接

- `frame` 同時送入 segmentation、depth inference 與 overlay。
- `forward_displacement_m` 送入 cross-frame matching 與 tracking。
- `speed_mps` 送入 risk assessment。
- `cumulative_forward_m` 送入 temporal snapshot / keyframe logic。

## 6.2 語意前端與 ROI 初始候選

![PIDNet 語意前端與 ROI](outputs/docs_figures/03_segmentation_frontend.png)

### 目的

從影像中取得道路區域、分析區域與初始 ROI，限制後續幾何推論只在合理的路面/近路面區域中進行。

### 輸入

- BGR frame
- `roi` config

### 處理與程式實作

`segment_frame(frame, roi_cfg)` 回傳 `FrontendResult`。主要動作如下：

1. 選擇 segmentation backend：
   - 若設定為 auto，依序嘗試 TensorRT、ONNX、official PIDNet、fallback。
   - backend attempts 會記錄在 frame state 的 `road_mask_stats.backend_attempts`。
2. 產生 road probability / road mask：
   - PIDNet 類後端會由 road class indices 建立道路類別遮罩。
   - fallback 會以影像亮度、紋理、下半部區域與形態學處理近似 road/non-road。
3. 建立 processing ROI：
   - 可使用手動 bbox、normalized bbox 或 trapezoid 類型的 ROI。
   - processing ROI 可限制 road mask 或 analysis mask 的作用範圍。
4. 建立 analysis mask / obstacle analysis mask：
   - analysis mask 是後續候選點抽樣與 ROI 產生的主要 mask。
   - obstacle analysis mask 決定障礙物候選搜尋區。
5. 建立 frontend ROI candidates：
   - 由負空間、近道路非道路區、hybrid mode 等方式產生初始候選 bbox。

### 中間產物

- `road_mask`
- `road_probability`
- `analysis_mask`
- `obstacle_analysis_mask`
- `processing_roi`
- `frontend.roi_candidates`
- `backend_attempts`

### 輸出與連接

- `road_mask` 送入 scale alignment 與 road point sampling。
- `obstacle_analysis_mask` 送入 candidate point sampling。
- `frontend.roi_candidates` 與 depth-derived ROIs 一起送入 ROI selection。
- `road_mask_stats` 最後寫入 `frame_states.jsonl`。

### 最後實驗對照

最後實驗逐幀 `road_mask_stats` 包含：

- `backend`
- `backend_attempts`
- `road_area_px`
- `road_ratio`
- `analysis_area_px`
- `analysis_ratio`
- `processing_roi`
- `obstacle_analysis_mask_source`
- `frontend_roi_count`
- `depth_roi_count`
- `selected_roi_count`

## 6.3 DA3 相對深度推論與尺度對齊

![相對深度與尺度對齊](outputs/docs_figures/04_relative_depth_and_scale_alignment.png)

### 目的

DA3 輸出的是 relative depth，不能直接當成公尺尺度。此步驟先取得相對深度，再利用道路平面與相機高度推導尺度因子，轉成 metric-like absolute depth。

### 輸入

- BGR frame
- `depth_model` config
- `road_mask`
- camera intrinsics
- `scale_alignment.camera_height_m`

### 處理與程式實作

1. 相對深度推論：
   - `infer_relative_depth(frame, depth_cfg)` 輸出 `relative_depth`。
   - 若後端提供 confidence，也會輸出 `confidence`。
   - 最後實驗的 `depth_stats.backend` 顯示使用 official DA3 / DA3-SMALL 類後端。
2. road mask 內取樣：
   - 從 `relative_depth` 與 `road_mask` 中以 stride 取樣有效道路點。
3. 在相對深度空間反投影：
   - 對每個像素 `(u, v)` 與相對深度 `z`：

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = relative_depth(u, v)
```

4. RANSAC 擬合相對尺度下的道路平面：
   - 隨機取三點形成候選平面。
   - 用點到平面的距離挑選 inliers。
   - 以 inliers 做 SVD 精修平面。
5. 由相對平面取得虛擬相機高度：

```text
h_hat_cam = |offset| / ||normal||
```

6. 以真實相機高度對齊尺度：

```text
scale_factor = camera_height_m / h_hat_cam
absolute_depth = relative_depth * scale_factor
```

### 中間產物

- `relative_depth`
- `confidence`
- 相對深度空間 road 3D points
- relative plane normal / offset
- `h_hat_cam`
- `scale_factor`
- `absolute_depth`

### 輸出與連接

- `absolute_depth` 送入 road point sampling、candidate point sampling、depth ROI generation。
- `scale_alignment` 寫入 `frame_states.jsonl`。
- `depth_stats` 寫入 `frame_states.jsonl`。

### 最後實驗對照

`frame_states.jsonl` 中：

- `depth_stats` 記錄 backend、shape、relative depth min/max/mean、confidence mean。
- `scale_alignment` 記錄 scale factor、candidate count、selected count、status、metadata。
- `evaluation_summary.json` 中 `scale_metrics.scale_factor_stability.count = 18`。

## 6.4 道路點反投影與地平面估計

![地平面估計](outputs/docs_figures/05_ground_plane_fit.png)

### 目的

利用已尺度化的 depth 與 road mask，把道路像素轉為 3D world points，並擬合當前幀的地平面。該地平面是後續 residual、candidate height、risk 幾何量的基準。

### 輸入

- `absolute_depth`
- `road_mask`
- camera intrinsics
- camera extrinsics
- `plane` config

### 處理與數學/程式實作

1. road points 抽樣：
   - `sample_points_from_mask()` 在 road mask 中按 stride 取樣像素。
2. 像素反投影到 camera coordinates：

```text
X_cam = (u - cx) * Z / fx
Y_cam = (v - cy) * Z / fy
Z_cam = Z
```

3. camera coordinates 轉 world coordinates：

```text
X_world = R^{-1} * (X_cam - t)
```

其中 `R` 與 `t` 來自 extrinsics。

4. RANSAC 擬合地平面：
   - 每次隨機取三個 world points。
   - 用叉積建立候選法向量。
   - 平面形式為：

```text
n^T X + d = 0
```

   - 計算所有點到平面的距離。
   - 距離小於 `ransac_distance_threshold_m` 的點視為 inliers。
   - 選擇 inlier 數最多的平面。
5. SVD 精修：
   - 對最佳 inliers 去中心化。
   - SVD 最小奇異值對應的右奇異向量作為法向量。
   - 用 centroid 與 normal 求 offset。

### 中間產物

- road image points
- road world points
- RANSAC candidate plane
- inlier mask
- SVD refined plane

### 輸出與連接

- `PlaneEstimate`：normal、offset、inlier ratio、support count、source、status。
- 地平面送入 residual ROI generation、candidate height/distance estimation 與 evaluation。

### 最後實驗對照

`frame_states.jsonl` 的 `plane_model` 含：

- `normal`
- `offset`
- `inlier_ratio`
- `support_count`
- `source`
- `status`

`evaluation_summary.json` 的 `plane_metrics` 含：

- plane normal error mean 約 `6.8949` 度
- plane normal error max 約 `18.7566` 度
- plane inlier ratio mean 約 `0.5881`
- plane inlier ratio min 約 `0.4705`

## 6.5 障礙物 ROI、殘差與候選群集

![ROI 產生與候選群集](outputs/docs_figures/06_roi_generation_and_candidate_clustering.png)

### 目的

從 obstacle analysis mask、depth map 與地平面中找出可能的障礙物區域，再把 ROI 內的異常 3D 點聚類成 candidate observations。

### 輸入

- `obstacle_analysis_mask`
- `absolute_depth`
- `relative_depth`
- BGR frame
- `road_mask`
- `plane`
- intrinsics / extrinsics
- `candidate_generation` config
- `frontend.roi_candidates`

### 處理與數學/程式實作

#### 6.5.1 Candidate point sampling

`sample_points_from_mask()` 在 obstacle analysis mask 內取樣，產生：

- candidate image points
- candidate world points
- candidate depths

這些點作為 residual 與 DBSCAN 的基本資料。

#### 6.5.2 Residual ROI

`build_depth_residual_roi_candidates()` 以觀測深度與地平面預期深度的差來產生 ROI。

對每個 sampled pixel：

1. 由像素 `(u, v)` 與 intrinsics 建立 camera ray。
2. 用 extrinsics 將 ray 轉為 world ray。
3. 計算 ray 與 plane `n^T X + d = 0` 的交點。
4. 將交點投回 camera coordinates，取得地平面預期深度 `expected_depth`。
5. 計算 residual：

```text
residual = observed_depth - expected_depth
```

6. 依正/負 residual 門檻建立 anomaly mask。
7. 以 morphology 與 connected components / contours 轉成 ROI bbox。

含義：

- 正 residual / 負 residual 會依實作中的符號定義分別對應凸起或凹陷類候選。
- residual 是「目前量測深度相對地平面預期深度」的幾何偏差。

#### 6.5.3 Depth-gradient contour ROI

`build_depth_gradient_contour_roi_candidates()` 以深度邊界產生 ROI：

1. 選擇 depth source：absolute 或 relative。
2. 可轉成 inverse depth。
3. 用 percentile clip 抑制極端值。
4. 做 median / blur。
5. 計算 depth gradient magnitude。
6. 可合併 color gradient。
7. 以 road prior、mask dilation、morph open/close/dilate 清理 seed mask。
8. 從 connected components / contour 取得 bbox。

最後實驗名稱含 `depth_gradient_verify`，且 `frame_states.jsonl` 的 ROI diagnostics 顯示 depth-gradient contour / hybrid 是主要候選來源。

#### 6.5.4 ROI selection / filtering / deduplication

`_select_roi_candidates_for_obstacles()` 將 frontend ROI 與 depth-derived ROI 合併，並依設定：

- 過濾太小或邊界條件不符的 ROI。
- 限制 ROI 數量。
- 依 IoU / containment 去重。
- 保留 source 與 diagnostics。

中間產物：

- `frontend_roi_diagnostics`
- `depth_roi_diagnostics`
- `selected_roi_diagnostics`
- `candidate_roi_source_counts`

#### 6.5.5 Candidate observation generation

`build_candidate_observations()` 對每個 selected ROI：

1. 擷取 ROI 內 sampled points。
2. 計算每個點相對 plane 的 residual / signed delta。
3. 分別建立 positive / negative abnormal point mask。
4. 過濾 forward distance 太小的點。
5. 在 world `Y,Z` 平面上執行 DBSCAN：

```text
DBSCAN(eps = dbscan_eps_m, min_samples = dbscan_min_samples)
```

6. 對每個 cluster 計算：
   - bbox
   - padded bbox
   - anchor
   - distance percentile
   - height
   - width
   - z range
   - support points
   - abnormal ratio
7. 若點數、forward points、abnormal ratio、bbox area 等不合格，寫入 reject reason。
8. 合格 cluster 轉成 `CandidateObservation`。

#### 6.5.6 Candidate classification

`classify_all_candidates()` 依 candidate 幾何與 ROI / road mask 關係判斷 candidate type，例如：

- 高度與寬度是否在有效範圍。
- bbox aspect ratio 是否 elongated。
- 是否接觸道路邊界。
- 正/負 residual 類型。

### 中間產物

- `candidate_sample`
- `depth_roi_candidates`
- `roi_candidates_for_candidates`
- `roi_cluster_traces`
- `cluster_reject_reason_counts`
- abnormal point masks
- DBSCAN labels
- candidate support points

### 輸出與連接

- `candidate_clusters` 送入 cross-frame matching、tracking、overlay 與 frame state。
- ROI diagnostics 寫入 `road_mask_stats`。
- candidate geometry 後續用於 risk assessment。

### 最後實驗對照

最後實驗中：

- 第一幀有 19 個 `roi_candidates` 與 19 個 `candidate_clusters`。
- `road_mask_stats` 含 `candidate_roi_source_counts`、`selected_roi_cluster_traces`、`cluster_reject_reason_counts`。
- 這些欄位可用來追蹤一個 ROI 如何被接受、拒絕或轉成 candidate。

## 6.6 跨幀摘要、Temporal Snapshot 與 Tracking

![跨幀追蹤](outputs/docs_figures/07_tracking_across_frames.png)

### 目的

把單幀 candidate observations 連接成跨幀穩定物件，避免每幀都產生互不相關的障礙物 id。

### 輸入

- `prev_candidates`
- `candidate_clusters`
- `forward_displacement_m`
- `tracking` config
- `temporal_measurement` config
- frame timestamp / cumulative forward distance

### 處理與數學/程式實作

#### 6.6.1 Cross-frame candidate summary

`summarize_cross_frame_matches()` 比較前一幀 candidates 與目前 candidates，產生 `CrossFrameMatch`。配對依據包含：

- anchor distance
- centroid shift
- bbox IoU
- z-range overlap
- obstacle type consistency
- ego forward displacement compensation

輸出 `cross_frame_matches` 會作為 tracking continuity bonus。

#### 6.6.2 Track prediction

`ObjectTracker._predict_anchor()` 用車體前進距離更新舊 track 的預測位置：

```text
predicted_z = previous_anchor_z - forward_displacement_m
```

若預測後物件距離太小或不是有限值，該 prediction 無效。

#### 6.6.3 Track-candidate matching

`ObjectTracker._matching_metrics()` 對 track 與 candidate 計算 score：

```text
score = anchor_dist
        - bbox_iou_weight * iou
        - continuity_bonus
        + type_penalty
        + size_penalty
```

其中：

- `anchor_dist` 越小越好。
- `iou` 越大越好，因此以負項降低 score。
- `continuity_bonus` 來自 cross-frame match，越大越好。
- `type_penalty` 懲罰 obstacle type 不一致。
- `size_penalty` 懲罰 height / width 變化太大。

`_match_track_pool()` 會依 score 排序，貪婪選擇不衝突的 track-candidate pair。

#### 6.6.4 Track lifecycle

`ObjectTracker.update()` 負責：

- 更新 matched active tracks。
- reactivation retired tracks。
- 為 unmatched candidates 建立新 object id。
- 對 missed tracks 增加 miss count。
- 將超過 miss 門檻的 track retired。

輸出 tracked object state 可能包含：

- generated
- updated
- reactivated
- predicted
- retired

#### 6.6.5 Temporal measurement / snapshot

`TemporalMeasurementManager` 提供 temporal fusion：

1. 在 frame 尾端用 `add_snapshot()` 存入歷史 candidates。
2. 若啟用 measurement，`select_keyframes()` 依 baseline 與 time gap 選 keyframes。
3. 歷史點轉到目前座標系：

```text
history_point_z_current = history_point_z - baseline_m
```

4. `match_history_candidate()` 用 anchor distance、centroid shift、bbox IoU、z-range overlap、type score 計算 match。
5. 若品質足夠，可融合 support points 並更新 candidate geometry。

### 中間產物

- `cross_frame_matches`
- predicted anchors
- candidate-track match scores
- object id assignments
- temporal keyframe selections
- temporal measurement records
- temporal snapshots

### 輸出與連接

- `tracked_objects` 送入 risk assessment。
- `cross_frame_matches`、`tracked_objects`、`temporal_measurements` 寫入 frame state。
- `prev_candidates` 更新為目前 candidates，供下一幀使用。

### 最後實驗對照

最後實驗的 `evaluation_summary.json` 顯示：

- active object count：639
- continuing object count：454
- tracking continuity：約 0.7105
- temporal measurement attempted / ok / applied 都是 0

因此本文把 temporal measurement 描述為目前程式支援的階段，但最後實驗沒有實際套用成功的 temporal measurement 產物。

## 6.7 風險評估與控制事件

![風險評估邏輯](outputs/docs_figures/08_risk_assessment_logic.png)

### 目的

將 tracked objects 轉成風險層級與控制事件，回答「目前障礙物是否需要 warning 或 danger」。

### 輸入

- `tracked_objects`
- road world points
- `speed_mps`
- `risk` config

### 處理與數學/程式實作

#### 6.7.1 剎車距離

`safe_braking_distance_m()` 計算理論安全剎車距離：

```text
d_brake = v * reaction_time_s + v^2 / (2 * max_decel_mps2)
```

其中：

- `v` 是 `speed_mps`。
- `reaction_time_s` 來自 config。
- `max_decel_mps2` 來自 config。

#### 6.7.2 可通行寬度

`estimate_clear_path_width()` 在障礙物縱深附近取 road world points：

```text
near_mask = |road_point_z - object_distance_z| <= z_window_m
W_clear = max(Y_near) - min(Y_near)
```

若附近 road points 不足，使用 `default_clear_path_width_m`。

#### 6.7.3 橫向占用比例

`compute_lateral_occupancy_ratio()` 計算：

```text
lateral_occupancy = object_width / W_clear
```

若 `W_clear <= 0`，占用比例視為 1。

#### 6.7.4 風險權重

`classify_risk_weight()` 根據 obstacle type 選擇 positive / negative 高度門檻，並計算可繞行寬度：

```text
bypass_capacity = W_clear - vehicle_width_m - 2 * side_clearance_m
bypassable = object_width < bypass_capacity
```

風險權重：

- `omega0`：高度低於 warning threshold。
- `omega1`：高度未超過 danger threshold 且仍可繞行。
- `omega2`：高度或寬度條件使其不可安全通過。

#### 6.7.5 Safe / Warning / Danger decision

`decision_for_object()` 使用三類 danger 條件：

1. 高度超過 danger threshold。
2. 距離小於等於 `d_brake`。
3. 橫向占用比例超過 `width_occupancy_threshold`。

若任一 danger 條件成立，decision 為 `danger`。

若未達 danger，但高度、距離或寬度接近門檻，decision 為 `warning`。

否則 decision 為 `safe`。

#### 6.7.6 Event filtering

`_RiskEventEmitter.filter_events()` 不會把所有 raw risk assessment 都直接輸出。它還會檢查：

- consecutive hits
- support point count
- cooldown
- predicted object policy
- fallback / temporal measurement 條件
- safe-clear logic

通過後，才呼叫 control backend 輸出 `ControlEvent`。若 backend 為 log，事件寫入 `risk_events.csv`。

### 中間產物

- `d_brake_m`
- `w_clear_m`
- lateral occupancy ratio
- risk weight
- raw risk assessments
- filtered risk events

### 輸出與連接

- risk 被附回 `tracked_objects`。
- `raw_risk_assessments` 與 filtered `risk_events` 寫入 frame state。
- filtered events 寫入 `risk_events.csv`。

### 最後實驗對照

最後實驗：

- `pipeline_summary.json` 中 `risk_event_count = 291`。
- `risk_events.csv` 欄位是 `frame_id,object_id,decision`。
- `evaluation_summary.json` 的 `risk_confusion_matrix` 顯示 18 幀都被預測到 danger 類風險欄位。

## 6.8 FrameState、overlay 與 pipeline summary

![Pipeline overlay 範例](outputs/docs_figures/09_pipeline_overlay_example.png)

### 目的

把每幀所有中間結果與最終決策統一封裝，方便後續 evaluation、debug 與論文圖表整理。

### 輸入

- depth stats
- scale alignment
- road mask stats
- plane model
- ROI candidates
- candidate clusters
- cross-frame matches
- tracked objects
- risk events
- timing
- temporal measurements
- raw risk assessments

### 處理與程式實作

1. 建立 `FrameState`：
   - `FrameState(...).to_dict()` 轉成可序列化 dict。
2. 若 `visualization.save_overlays` 為 true，呼叫 `save_overlay()`：
   - 把 road mask 疊在原圖上。
   - 畫 road sampled points。
   - 畫 processing ROI。
   - 畫 depth ROI。
   - 畫 tracked object bbox。
   - 標註 object id、state、risk weight、decision、Z/H/W、tracking score、temporal quality。
3. 幀尾呼叫 `temporal_manager.add_snapshot()` 保存 history。
4. loop 結束後：
   - `write_jsonl(output_dir / "frame_states.jsonl", results)`
   - `save_json(output_dir / "pipeline_summary.json", summary)`

### 輸出

```text
outputs/<run>/
  frame_states.jsonl
  pipeline_summary.json
  overlays/<frame_id>.png
  risk_events.csv
```

### 最後實驗對照

最後實驗第一幀包含：

- `roi_candidates`：19 個
- `candidate_clusters`：19 個
- `tracked_objects`：19 個
- `raw_risk_assessments`：19 個

完整實驗：

- 18 幀
- overlays 目錄包含對應 frame 的 overlay PNG
- `pipeline_summary.json` 記錄整體 latency 與 risk event count

## 7. 中間視覺化圖層

`visualization.py` 支援中間過程視覺化命名，可用於把 pipeline 拆成連續圖層。建議路徑如下：

```text
outputs/<run>/intermediates/<frame_id>_01_original.png
outputs/<run>/intermediates/<frame_id>_02_da3_depth.png
outputs/<run>/intermediates/<frame_id>_03_absolute_depth.png
outputs/<run>/intermediates/<frame_id>_04_road_mask.png
outputs/<run>/intermediates/<frame_id>_05_analysis_mask.png
outputs/<run>/intermediates/<frame_id>_06_candidate_clusters.png
outputs/<run>/intermediates/<frame_id>_07_risk_decision.png
```

| 圖層 | 對應流程 | 建議說明 |
|---|---|---|
| `_01_original` | 原始輸入 | 顯示未處理 frame |
| `_02_da3_depth` | DA3 relative depth | 顯示相對深度分布 |
| `_03_absolute_depth` | scale alignment 後 depth | 顯示公尺尺度化後深度 |
| `_04_road_mask` | segmentation frontend | 顯示 road mask |
| `_05_analysis_mask` | ROI / candidate search | 顯示 obstacle analysis mask |
| `_06_candidate_clusters` | candidate generation | 顯示候選群集與 bbox |
| `_07_risk_decision` | risk assessment | 顯示 risk weight / decision |

最後實驗的主要實際視覺輸出是：

```text
outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/overlays/*.png
outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/plots/*.png
```

## 8. 評估流程與輸出

![評估輸出](outputs/docs_figures/10_evaluation_outputs.png)

### 目的

把 pipeline 預測結果與 GT 對齊，產生 detection、geometry、plane、scale、tracking、latency、temporal 與診斷指標。

### 輸入

- `outputs/<run>/frame_states.jsonl`
- `datasets/<sequence_id>/gt_obstacles.csv`
- `datasets/<sequence_id>/gt_plane.json`
- config

### 處理與數學/程式實作

`evaluate_pipeline()` 執行：

1. 讀取 frame states：
   - `read_jsonl(output_dir / "frame_states.jsonl")`
2. 讀取 GT：
   - `read_gt_obstacles(dataset_dir / "gt_obstacles.csv")`
   - `read_gt_plane(dataset_dir / "gt_plane.json")`
3. 依 frame 建立 GT 與 prediction index。
4. Detection metrics：
   - 逐幀判斷 GT 是否有 hazard、prediction 是否有 hazard。
   - 累計 TP / FP / FN / TN。
   - 計算 precision、recall、F1、accuracy。
5. Risk confusion matrix：
   - 依 `omega0/omega1/omega2` 建立 3x3 confusion matrix。
6. Geometry metrics：
   - 將 GT obstacle 與 predicted object 配對。
   - 計算 height / width / distance absolute error。
   - 輸出 MAE、RMSE、max error。
7. Plane metrics：
   - 將 predicted plane normal 與 GT normal 正規化。
   - 用 dot product 與 arccos 計算法向量角度誤差：

```text
angle_error = arccos(clip(dot(n_pred, n_gt), -1, 1))
```

8. Scale metrics：
   - 收集 `scale_alignment.status == ok` 的 scale factor。
   - 計算 mean、std、count。
9. Tracking metrics：
   - 統計 active objects 與 continuing objects。
   - `tracking_continuity = continuing_object_count / active_object_count`。
10. Latency metrics：
    - 收集每幀 `timing_ms.total`。
    - 計算 mean / max latency。
11. Temporal measurement metrics：
    - 統計 attempted、ok、applied、quality、method counts。
12. Label-layer diagnostics：
    - 檢查特定 label 在 frontend ROI、depth ROI、selected ROI、candidate、tracked layers 中的命中情況。

### 輸出

程式會建立或使用：

```text
outputs/<run>/
  evaluation_summary.json
  plots/risk_confusion_matrix.png
  plots/geometry_error_scatter.png
  plots/latency_histogram.png
```

在有資料時，也可能產生：

```text
outputs/<run>/tables/object_geometry_matches.csv
outputs/<run>/tables/label_07_layer_diagnostics.csv
outputs/<run>/tables/label_07_roi_to_candidate_failures.csv
outputs/<run>/roi_to_candidate_trace_overlays/*.png
```

### 最後實驗對照

最後實驗實際有三張 evaluation plot：

- `plots/risk_confusion_matrix.png`
- `plots/geometry_error_scatter.png`
- `plots/latency_histogram.png`

最後實驗沒有 populated tables，因為：

- `geometry_metrics.matched_pairs = 0`
- label `07` 的 `total_gt_frames = 0`

## 9. 動詞到實作對照表

| 動詞 | 程式位置 | 數學/程式實作摘要 | 最後實驗對應產物 |
|---|---|---|---|
| 讀取 | `io_utils.py`, `pipeline.py` | 讀影像/CSV/YAML，建立 frame、motion、calibration input | `frame_states.jsonl.frame_id` |
| 分割 | `segmentation.py` | PIDNet/fallback 產生 road mask、analysis mask、ROI | `road_mask_stats` |
| 推論深度 | `depth_model.py` | DA3/mock/TensorRT 輸出 relative depth | `depth_stats` |
| 對齊尺度 | `geometry.py` | road-plane + camera height 求 scale factor | `scale_alignment` |
| 反投影 | `geometry.py` | intrinsics/extrinsics 將像素與深度轉為 3D 點 | road/candidate support points 診斷 |
| 擬合平面 | `ground_plane.py` | RANSAC + SVD 估計 `n^T X + d = 0` | `plane_model`、`plane_metrics` |
| 產生 ROI | `obstacles.py`, `pipeline.py` | residual ROI、depth-gradient ROI、frontend ROI 合併與去重 | `roi_candidates`、ROI diagnostics |
| 聚類 | `obstacles.py` | DBSCAN 於 world Y/Z 聚類 support points | `candidate_clusters` |
| 分類 | `obstacles.py` | 高度、寬度、殘差比例、bbox 條件判定候選類型 | candidate metadata |
| 配對 | `obstacles.py`, `tracking.py` | IoU、anchor distance、centroid shift、z-range overlap | `cross_frame_matches` |
| 追蹤 | `tracking.py` | object id、track update、miss/predicted/reactivation | `tracked_objects`、tracking metrics |
| 融合 | `temporal_measurement.py` | keyframe selection 與 history support point fusion | 最後實驗 metrics 為 0，未實際套用 |
| 評估風險 | `risk.py` | braking distance、clear width、occupancy、threshold decision | `raw_risk_assessments`、`risk_events.csv` |
| 視覺化 | `visualization.py` | overlay、intermediate layers、plots | `overlays/*.png`、`plots/*.png` |
| 評估 | `evaluation.py` | GT 對齊、metrics、plots、可選 tables/diagnostics | `evaluation_summary.json` |

## 10. 建議補圖清單

以下路徑建議作為文件插圖的固定位置。可先保留 placeholder，之後從最後實驗輸出中挑選代表圖，或另行重繪流程示意圖。

```text
outputs/docs_figures/01_workflow_overview.png
outputs/docs_figures/02_input_dataset_contract.png
outputs/docs_figures/03_segmentation_frontend.png
outputs/docs_figures/04_relative_depth_and_scale_alignment.png
outputs/docs_figures/05_ground_plane_fit.png
outputs/docs_figures/06_roi_generation_and_candidate_clustering.png
outputs/docs_figures/07_tracking_across_frames.png
outputs/docs_figures/08_risk_assessment_logic.png
outputs/docs_figures/09_pipeline_overlay_example.png
outputs/docs_figures/10_evaluation_outputs.png
```

建議對應來源：

| 建議圖 | 可使用的實際來源 |
|---|---|
| `01_workflow_overview.png` | 依 `pipeline.py::run_pipeline()` 重繪流程圖 |
| `02_input_dataset_contract.png` | 依 `io_utils.py` 與 `pipeline.py` 重繪資料結構圖 |
| `03_segmentation_frontend.png` | 從 overlay 或中間圖標示 road / analysis / processing ROI |
| `04_relative_depth_and_scale_alignment.png` | 用 DA3 depth 與 scale factor metadata 重繪 |
| `05_ground_plane_fit.png` | 用 road sampled points 與 `plane_model` 重繪 |
| `06_roi_generation_and_candidate_clustering.png` | 用 `selected_roi_cluster_traces` 或 overlay 重繪 |
| `07_tracking_across_frames.png` | 用連續 frame 的 object id / cross-frame matches 重繪 |
| `08_risk_assessment_logic.png` | 依 `risk.py` 的公式與 threshold 重繪決策圖 |
| `09_pipeline_overlay_example.png` | 從 `outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/overlays/*.png` 選一張 |
| `10_evaluation_outputs.png` | 可組合 `plots/risk_confusion_matrix.png`、`geometry_error_scatter.png`、`latency_histogram.png` |

## 11. 實作來源索引

本文依據下列程式與最後實驗輸出整理：

- `configs/default.yaml`：實驗參數與各階段控制項。
- `monocular_experiment/main.py`：CLI 階段與命令入口。
- `monocular_experiment/io_utils.py`：資料讀寫與輸入檔格式讀取。
- `monocular_experiment/pipeline.py`：逐幀 pipeline 的權威執行順序。
- `monocular_experiment/segmentation.py`：road mask、analysis mask、ROI frontend。
- `monocular_experiment/depth_model.py`：DA3/mock/TensorRT 相對深度推論。
- `monocular_experiment/geometry.py`：尺度對齊、反投影、座標轉換。
- `monocular_experiment/ground_plane.py`：RANSAC + SVD 地平面估計。
- `monocular_experiment/obstacles.py`：residual ROI、depth-gradient ROI、DBSCAN 候選生成、分類、跨幀摘要。
- `monocular_experiment/tracking.py`：object tracking 與 assignment score。
- `monocular_experiment/temporal_measurement.py`：keyframe selection 與 temporal fusion。
- `monocular_experiment/risk.py`：剎車距離、可通行寬度、風險矩陣。
- `monocular_experiment/evaluation.py`：metrics、tables、plots、diagnostics。
- `monocular_experiment/visualization.py`：overlay、中間圖、評估圖輸出。
- `monocular_experiment/models.py`：FrameState、PlaneEstimate、CandidateObservation、TrackedObject、RiskAssessment schema。
- `outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/frame_states.jsonl`：最後實驗逐幀狀態。
- `outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/pipeline_summary.json`：最後實驗 pipeline summary。
- `outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/evaluation_summary.json`：最後實驗 evaluation summary。
- `outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/risk_events.csv`：最後實驗 risk event log。
- `outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/overlays/*.png`：最後實驗逐幀 overlay。
- `outputs/laf_raw_02hk_000000_020_190_depth_gradient_verify/plots/*.png`：最後實驗評估圖。
