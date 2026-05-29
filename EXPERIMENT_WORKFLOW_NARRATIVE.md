# code2 實驗流程程式流程簡述

本文參考「影像擷取與預處理 V-SLAM 位姿估計程式流程簡述」的寫法，把 `code2` 目前完成的單目深度、地平面、障礙物追蹤、單目時序測量與風險評估流程整理成 Step1 到 Step20 的程式流程說明。每一個 Step 先用自然段描述該階段在整體流程中的角色，再補充主要計算方式與輸出結果，最後以「圖」的形式標示可對應的示意圖或實際輸出位置。

最後完成的代表性實驗目錄為 `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/`，對應序列為 `laf_07ff_000003_000080_000340_verify27`。該次實驗共處理 27 幀 LostAndFound 影像；深度推論實際使用 `official_da3:DA3-SMALL`，道路語意前端實際使用 `official_pidnet:pidnet-s`，processing ROI 設定為 `foe_road_triangle`，實際逐幀結果為 `foe_road_triangle=25`、`trapezoid=2`。其中第一幀因無前一幀可供光流與 FOE 估計，必然退回 trapezoid；其餘大多數幀則使用 FOE 驅動的道路走廊 ROI。障礙物分析區域來源為 `processing_roi`，workflow figure 則固定選取倒數第二幀 `07_Festplatz_Flugfeld_000003_000330`（index 25）作為整份圖說的代表影格。

本次實驗整段輸出 30 個 frontend ROI、0 個 depth ROI、23 個 selected ROIs、21 個 candidate clusters、4 個 cross-frame matches、53 筆 tracked object frame records、46 筆 raw risk assessments，以及 5 筆 filtered risk events，且 5 筆皆為 `danger`。temporal measurement attempted 共 21 筆、`ok` 共 8 筆、`applied` 共 8 筆；overall mean quality 約為 0.3048，而成功 applied 的 8 筆 measurement quality 平均約為 0.8002，median 約為 0.8128。evaluation 顯示 detection precision=1.0、recall≈0.7778、F1≈0.8750；plane normal error mean≈6.93°、plane inlier ratio mean≈0.8075；tracking continuity≈0.7609；mean latency≈6680.44 ms，max latency≈14563.38 ms。

## Step1：讀取實驗序列並建立每一幀身份

Step1 使用資料序列作為整個實驗的起點。`run_pipeline()` 先根據 `dataset_root` 與 `sequence_id` 找到 `datasets/<sequence_id>/frames/`，再由 `list_frame_paths()` 列出影像檔，並用 `read_image()` 將每張影像讀成 BGR frame。每張影像的檔名 stem 會成為 `frame_id`，此 ID 會一路串接後續的深度結果、道路遮罩、候選物件、追蹤物件、時序測量、風險事件與 overlay 圖。

因此，Step1 不只是讀取影像，也是在建立整個 frame-level 實驗記錄的索引。最後實驗第一幀的 `frame_id` 為 `07_Festplatz_Flugfeld_000003_000080`，整段序列共有 27 幀，輸入影像來自 LostAndFound raw sequence 的 `07_Festplatz_Flugfeld_000003_000080_leftImg8bit.png` 到 `07_Festplatz_Flugfeld_000003_000340_leftImg8bit.png`。原始 LostAndFound 資料位於 `datasets_raw/LostAndFound/`，pipeline 實際讀取已整理好的 evaluation sequence `datasets/laf_07ff_000003_000080_000340_verify27/`。

圖 1：Step1 影像序列讀取與 frame identity 建立結果。輸出為：BGR frame、`frame_path`、`frame_id`、逐幀處理單位。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig01_frame_input.png`。

## Step2：讀取 motion 並推導速度與前向位移

Step2 讀取與每一幀對應的 `motion.csv` row，並同時保留目前幀與前一幀的 motion 資料。pipeline 會取得 `timestamp_s`，再依據可用欄位推導 `speed_mps`、`forward_displacement_m` 與 `cumulative_forward_m`。如果資料列已經提供有效速度或前向位移，程式直接採用；如果其中一項缺失，則使用時間差 `Δt = timestamp_curr_s - timestamp_prev_s` 進行換算。

在本次 07ff 驗證序列中，時間戳由 0.0 s 遞增到 5.2 s，除第一幀外，後續幀的 `speed_mps` 約為 0.75，`forward_displacement_m` 約為 0.15，`cumulative_distance_m` 最後累積到 3.9 m。這些 motion 量後續會用於跨幀候選摘要、tracking 預測、temporal snapshot keyframe 選擇，以及 risk assessment 的剎車距離計算。

圖 2：Step2 motion row 讀取與車體運動量推導。輸出為：`speed_mps`、`forward_displacement_m`、`cumulative_forward_m`。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig02_motion_profile.png`。

## Step3：載入相機內外參並建立 2D/3D 轉換基礎

Step3 從 config 的 `calibration` 區塊解析 intrinsics 與 extrinsics 路徑，再用 `load_yaml()` 載入相機內參與外參。內參提供 `fx`、`fy`、`cx`、`cy`，用於將像素與深度反投影到 camera coordinates；外參提供 rotation matrix 與 translation vector，用於 camera coordinates 與 local world coordinates 之間的轉換。

此步驟本身不產生障礙物結果，但它是後續 scale alignment、road point sampling、candidate point sampling、plane fitting、temporal support point fusion 與 risk width estimation 的共同幾何基礎。反投影公式為 `X_cam=(u-cx)Z/fx`、`Y_cam=(v-cy)Z/fy`、`Z_cam=Z`，world 轉換公式為 `X_world = R^{-1}(X_cam - t)`。

圖 3：Step3 相機 calibration 與座標轉換基礎。輸出為：camera matrix、rotation、translation、2D/3D transform basis。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig03_calibration_basis.png`。

## Step4：使用 PIDNet 產生道路遮罩，並以 FOE 建立動態 processing ROI

Step4 對每一幀呼叫 `segment_frame(frame, roi_cfg, apply_processing_roi=...)`，先由 `official_pidnet:pidnet-s` 產生道路遮罩，再視設定決定是否額外套用 processing ROI。因為本次 `roi.processing_rect.mode` 設為 `foe_road_triangle`，所以 pipeline 會先取得 `frontend.road_mask`，再呼叫 `build_foe_road_triangle_roi(previous_frame_bgr, current_frame_bgr, road_mask, config, previous_state)` 估計光流、FOE 與道路邊界，最後再由 `apply_processing_rect()` 把動態 ROI 寫回 frontend 結果。

本次實驗最重要的差異是紅框 processing ROI 不再是固定 trapezoid，而是以 FOE 與道路邊界動態生成的道路走廊。整段 27 幀中，有 25 幀實際使用 `foe_road_triangle`，2 幀退回 `trapezoid`；其中第一幀的 fallback 是預期行為，因為沒有 previous frame 可以估計光流。這一步輸出的 `road_mask` 會送入 scale alignment 與 road point sampling，而 `obstacle_analysis_mask` 來源設定為 `processing_roi`，因此後續候選點抽樣與障礙物分析都被限制在這個動態 ROI 內。

圖 4：Step4 PIDNet segmentation 與 FOE processing ROI 結果。輸出為：`FrontendResult`、`road_mask`、`analysis_mask`、`obstacle_analysis_mask`、processing ROI polygon/bbox。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig04_fallback_segmentation_roi.png`。

## Step5：使用 DA3-SMALL 產生相對深度

Step5 對每一幀呼叫 `infer_relative_depth(frame, depth_cfg)`，使用 DA3-SMALL 產生與原圖對齊的 relative depth map。本次實驗中，27 幀的 `depth_stats.backend` 皆為 `official_da3:DA3-SMALL`，表示深度後端在整段序列中沒有切換或退回其他備援實作。

此時的 depth map 只保留畫面中的遠近關係，尚未對齊到公尺尺度，因此不能直接用來計算障礙物高度、寬度或剎車距離。`relative_depth` 接下來會與 `road_mask` 一起進入 scale alignment，而 `depth_stats` 會被寫入 `frame_states.jsonl`，內容包含 backend、shape、valid count、relative depth 統計與 confidence diagnostics。

圖 5：Step5 DA3-SMALL 相對深度推論結果。輸出為：`relative_depth`、`depth_stats`、depth backend diagnostics。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig05_relative_depth_da3.png`。

## Step6：使用道路遮罩與相機高度將相對深度對齊到公尺尺度

Step6 將 DA3-SMALL 的 relative depth 轉成 absolute depth。pipeline 呼叫 `estimate_scale_factor_from_relative_depth_plane(relative_depth_map, road_mask, intrinsics, camera_height_m, ...)`，先在 `road_mask` 中取樣有效像素，對每個像素 `(u,v)` 與相對深度 `z` 做相對尺度反投影：`x=(u-cx)z/fx`、`y=(v-cy)z/fy`、`z=relative_depth(u,v)`。

接著程式在這些相對 3D 道路點上用 RANSAC 擬合平面，計算相機原點到相對平面的距離 `h_hat_cam = |offset| / ||normal||`，最後用已知相機高度求 `scale_factor = camera_height_m / h_hat_cam`，並將整張圖轉為 `absolute_depth = relative_depth * scale_factor`。本次實驗 evaluation 的 `scale_factor_stability` 統計為 mean≈9.3634、std≈1.5099、count=27，表示整段序列都成功完成尺度對齊，且每一幀都有可用的 scale factor。

圖 6：Step6 relative depth 與道路平面尺度對齊結果。輸出為：`scale_factor`、`absolute_depth`、`scale_alignment`。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig06_scale_alignment_absolute_depth.png`。

## Step7：從道路遮罩抽樣 3D 道路點並擬合地平面

Step7 使用 `sample_points_from_mask(depth_map_m=absolute_depth, mask=road_mask, intrinsics=intrinsics, extrinsics=extrinsics, ...)`，將道路像素轉成 3D world points。對每個取樣像素 `(u,v)` 與深度 `Z`，程式先用相機內參反投影到 camera coordinates，再用外參轉換到 local world coordinates。

取得 road world points 後，`estimate_ground_plane(road_world_points, plane_cfg)` 會用 RANSAC 與 SVD 擬合地平面 `n^T X + d = 0`。evaluation 顯示本次實驗的 `plane_normal_error_deg.mean≈6.9291`、`max≈12.4355`，`plane_inlier_ratio.mean≈0.8075`、`min≈0.529`，表示用 PIDNet 道路遮罩配合尺度對齊後，整體地平面估計品質明顯優於先前使用 fallback 前端的版本。

圖 7：Step7 道路 3D 點抽樣與地平面 RANSAC 擬合結果。輸出為：road world points、RANSAC inliers、`plane_model`。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig07_ground_plane_ransac.png`。

## Step8：在 processing ROI 內抽樣候選點

Step8 將障礙物分析區域從 2D mask 轉成可進行幾何分析的 3D 點集。本次實驗中，障礙物分析區域來源為 `processing_roi`，因此 candidate point sampling 實際是在 FOE 道路走廊 ROI（必要時退回 trapezoid）內進行。pipeline 先由 `_resolve_obstacle_analysis_mask(frontend)` 取得 `obstacle_analysis_mask`，再用 `sample_points_from_mask()` 根據 `absolute_depth`、intrinsics 與 extrinsics 將 mask 中的像素轉成 candidate image points、candidate world points 與 candidate depths。

這些 candidate sampled points 還不是最終障礙物，而是後續 ROI 過濾、DBSCAN 聚類與 temporal support point fusion 共同使用的候選點集。因為本次紅框 ROI 已經集中到道路前向區域，後續障礙物搜尋區域也同步跟著 FOE 走廊收斂。

圖 8：Step8 processing ROI 內的 candidate point sampling 結果。輸出為：candidate image points、candidate world points、candidate depths。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig08_candidate_point_sampling.png`。

## Step9：保留 depth-derived ROI 能力，但本次實驗未形成有效 depth ROI

Step9 的程式能力仍包含兩種幾何 ROI 線索。第一種是地平面 residual：`build_depth_residual_roi_candidates()` 可對每個 candidate sampled pixel 建立 camera ray，計算 observed depth 與 plane expected depth 的差異，再將超過 residual 門檻的點轉為 residual ROI。第二種是 depth-gradient contour：`build_depth_gradient_contour_roi_candidates()` 使用 `absolute_depth` 或 `relative_depth` 的深度梯度、color gradient、road prior 與 morphology 找出深度邊界候選框。

但在本次 07ff FOE rerun 中，evaluation 的 label-layer diagnostics 對 `07` 類別顯示 `depth_roi` 命中數為 0，整段實驗的 depth ROI 累計也為 0。這代表程式雖然保留 depth-derived ROI 支線，本次實驗真正送入後續候選搜尋的區域主要仍由 frontend 與 processing ROI 約束，而不是由 depth-gradient 額外擴增出新的 ROI。

圖 9：Step9 depth-derived ROI 生成結果。輸出為：depth-gradient map、color/depth edge support、depth-derived ROI candidates（本次實驗有效命中為 0）。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig09_depth_roi_generation.png`。

## Step10：合併、過濾與去重 ROI，形成 candidate generation 搜尋框

Step10 使用 `_select_roi_candidates_for_obstacles()` 將 frontend ROI 與 depth-derived ROI 合併，並依照最後實驗設定中的選擇策略、面積條件、邊界條件與 IoU/containment 去重邏輯，得到真正送入 candidate generation 的 selected ROI list。

這一步的選擇不是語意分類，而是將多來源 ROI 整理成一組較可靠、較少重複且保留來源資訊的候選框。本次實驗整段累計 frontend ROI 為 30、selected ROI 為 23，而 depth ROI 為 0，因此 selected ROI 幾乎完全由道路前端與 FOE ROI 約束後的可疑區域組成。對 label `07` 而言，27 個 GT frames 中 `frontend_roi` 命中 10 次、`selected_roi` 命中 8 次，顯示 selected ROI 這一步已經進一步收斂但也帶來部分召回損失。

圖 10：Step10 多來源 ROI 合併、過濾與去重結果。輸出為：selected ROIs、`candidate_roi_source_counts`、`selected_roi_diagnostics`。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig10_selected_roi_generation.png`。

## Step11：在 selected ROI 內使用 DBSCAN 聚類異常點

Step11 由 `build_candidate_observations()` 將 selected ROI 轉成 `CandidateObservation`。程式會先切出每個 ROI 內的 candidate sampled points，再計算這些點相對地平面的 signed residual 或 delta，分別建立 positive 與 negative abnormal point masks。接著過濾前向距離太小的點，並在 world `Y,Z` 平面上執行 `DBSCAN(eps=dbscan_eps_m, min_samples=dbscan_min_samples)`。

DBSCAN 會將空間上密度足夠的異常點群標成 cluster，孤立點標成 noise。每個有效 cluster 會被轉成 candidate，並計算 bbox、padded bbox、anchor、distance percentile、height、width、z range、support points 與 abnormal ratio。本次實驗整段共形成 21 個 candidate clusters；對 label `07` 而言，27 個 GT frames 中有 7 次進入 candidate layer，表示 selected ROI 到 candidate clustering 之間仍存在一小段召回落差。

圖 11：Step11 selected ROI 內的異常點 DBSCAN 聚類結果。輸出為：abnormal point masks、DBSCAN labels、`CandidateObservation`、candidate clusters。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig11_dbscan_candidate_clusters.png`。

## Step12：根據候選幾何與道路關係分類 candidate type

Step12 使用 `classify_all_candidates()`，依據每個 candidate 的幾何特徵與道路遮罩關係進行候選類型判定。最後實驗沿用 candidate 的 positive/negative obstacle type、height、width、bbox aspect ratio、道路邊界接觸情況與 residual 分布來補充 candidate metadata。

此步驟不是再執行一個影像分類模型，而是根據幾何門檻與形態條件，把已生成的候選障礙物描述得更完整，方便後續 tracking score、temporal match 與 risk rule 使用。分類後的 candidates 會送入跨幀摘要與 object tracker。

圖 12：Step12 candidate geometry 與道路關係分類結果。輸出為：candidate type、obstacle type、type-related metadata。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig12_candidate_type_classification.png`。

## Step13：建立跨幀候選摘要，連接上一幀與目前幀

Step13 使用 `summarize_cross_frame_matches(previous_candidates, current_candidates, ...)` 比較上一幀 candidates 與目前 candidates。因為車體向前移動，同一個靜態障礙物在車體座標中的 `Z` 應該減少，所以配對時會使用 `forward_displacement_m` 進行補償。

配對特徵包含 anchor distance、centroid shift、bbox IoU、z-range overlap 與 obstacle type consistency。這一步輸出的 `CrossFrameMatch` 還不是最終 track，但會在 `ObjectTracker` 中成為 continuity bonus，幫助 tracker 維持穩定 object id，也會間接提高 Step15 時序測量找到歷史候選的機率。本次實驗整段共產生 4 個 cross-frame matches，顯示這個 07ff 驗證序列的連續候選重疊程度相對有限。

圖 13：Step13 previous/current candidates 跨幀配對摘要結果。輸出為：pairwise match metrics、`cross_frame_matches`。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig13_cross_frame_matches.png`。

## Step14：使用 ObjectTracker 將 candidates 轉成持續 tracked objects

Step14 由 `ObjectTracker.update()` 將單幀 candidate 轉成跨幀持續物件。tracker 先用 `_predict_anchor()` 預測既有 track 在目前幀的位置，公式為 `predicted_z = previous_anchor_z - forward_displacement_m`。接著 `_matching_metrics()` 對每個 track-candidate pair 計算分數：`score = anchor_dist - bbox_iou_weight * iou - continuity_bonus + type_penalty + size_penalty`。

分數越低代表越適合配對，因為 anchor 越近越好、IoU 越高越好、continuity bonus 越高越好，而 type mismatch 與 size change 會增加懲罰。程式排序所有 pair 後進行 greedy matching，更新 matched tracks，為 unmatched candidates 建立新 object id，並處理 missed、predicted、retired 與 reactivated 等 lifecycle state。本次實驗整段產生 53 筆 tracked object frame records；evaluation 中 `active_object_count=46`、`continuing_object_count=35`，tracking continuity≈0.7609。

圖 14：Step14 ObjectTracker 配對、預測與 lifecycle 更新結果。輸出為：predicted anchors、matching scores、tracked objects。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig14_tracker_lifecycle.png`。

## Step15：套用單目時序測量修正候選幾何

Step15 使用 `TemporalMeasurementManager` 保存歷史 snapshot，並根據 baseline、time gap、object id / geometry match 與支撐點品質做 temporal measurement。每幀尾端的 `add_snapshot()` 會保存當前 frame、timestamp、`cumulative_forward_m` 與 candidates；下一幀或後續幀進行 measurement 時，歷史點會先依前向 baseline 轉換到目前座標，公式為 `history_point_z_current = history_point_z - baseline_m`。

在本次 07ff FOE rerun 中，`measure_candidates()` 全部使用 `support_point_fusion` 方法，attempted 共 21 筆、`ok` 共 8 筆、`applied` 共 8 筆。evaluation 的 overall mean quality 約為 0.3048；若只看成功 applied 的 8 筆，quality mean 約為 0.8002，median 約為 0.8128。status counts 為 `no_candidate_match=12`、`ok=8`、`insufficient_current_support=1`；method counts 為 `support_point_fusion=21`。這證明本次實驗中的單目時序測量已實際介入最終幾何估計，而不是只保留模組卻未套用。

在 workflow figure 固定選取的代表影格 `07_Festplatz_Flugfeld_000003_000330` 中，可見框線來自 tracked objects，其中至少兩個可見物件已套用 temporal measurement：`obj_007` 的 `temporal_applied=true`、`quality≈0.8536`、`height_m≈0.6219`、`width_m≈1.2349`、`distance_m≈10.8855`；`obj_008` 的 `temporal_applied=true`、`quality≈0.7732`、`height_m≈0.2074`、`width_m≈1.0949`、`distance_m≈9.8993`。這也是本次實驗在代表圖中可以直接觀察到時序測量結果的關鍵例子。

圖 15：Step15 單目時序測量與 temporal snapshot history。輸出為：temporal snapshot history、measurement status、applied count、quality。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig15_temporal_snapshot_history.png`。

## Step16：根據 tracked objects、道路點與速度計算風險

Step16 由 `assess_tracked_objects()` 對 tracked objects 進行風險評估。對每個 tracked object，程式先用目前速度計算剎車距離：`d_brake = v * reaction_time_s + v^2 / (2 * max_decel_mps2)`。接著用 road world points 在障礙物前向距離附近估計可通行寬度：`near_mask = |road_point_z - object_distance_z| <= z_window_m`，`W_clear = max(Y_near) - min(Y_near)`。

取得道路可通行寬度後，程式計算橫向占用比例 `lateral_occupancy = object_width / W_clear`，並依 obstacle type 的高度門檻與可繞行寬度 `bypass_capacity = W_clear - vehicle_width_m - 2 * side_clearance_m` 判斷 `omega0`、`omega1` 或 `omega2`。由於 Step15 已將成功的 temporal measurement 寫回 candidate 幾何量，這一步使用的 tracked object 規格已包含時序修正後的高度、寬度與距離。本次實驗整段共有 46 筆 raw risk assessments。

圖 16：Step16 tracked object 風險評估結果。輸出為：braking distance、clear width、risk weight、raw risk assessments。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig16_raw_risk_assessment.png`。

## Step17：過濾 raw risk assessments 並輸出 risk events

Step17 將 raw risk assessment 轉成真正輸出的 risk events。pipeline 透過 `_RiskEventEmitter.filter_events()` 檢查 consecutive hits、support point count、cooldown、predicted object policy 與 safe-clear logic，只有通過過濾的事件才會被送到 control backend。

本次實驗使用 log 後端，因此通過的事件會寫入 `risk_events.csv`，欄位為 `frame_id,object_id,decision`。最後實驗共輸出 5 筆 filtered risk events，且 5 筆皆為 `danger`；`pipeline_summary.json` 的 `risk_event_count` 也為 5。此步驟把逐幀風險推論收斂成可記錄、可評估的事件層輸出。

圖 17：Step17 raw risk filtering 與 risk event emission 結果。輸出為：filtered risk events、`risk_events.csv`。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig17_filtered_risk_events.png`。

## Step18：保存 FrameState、輸出 overlay，並固定 workflow figure 到倒數第二幀

Step18 在每幀處理結束後建立 `FrameState`。pipeline 會把 `depth_stats`、`scale_alignment`、`road_mask_stats`、`plane_model`、`roi_candidates`、`candidate_clusters`、`cross_frame_matches`、`tracked_objects`、`risk_events`、`timing_ms`、`temporal_measurements` 與 `raw_risk_assessments` 統一寫成 dict，再追加到 `frame_states.jsonl`。

若 `visualization.save_overlays` 啟用，程式會呼叫 `save_overlay()`，在原始 frame 上疊加 road mask、road sampled points、processing ROI、depth ROI、tracked object bbox、object id、state、risk weight、decision 以及 Z/H/W 等幾何資訊。這裡要特別注意：overlay 與 workflow figure 中的障礙物框來自 `tracked_objects`，而不是單純的 `candidate_clusters`；紅色 ROI 則是 Step4 寫回的 processing ROI。由於本次 config 設定 `workflow_figure_frame_index=-2` 且 `workflow_figure_selection_mode=fixed_index`，最後選中的代表影格是倒數第二幀 `07_Festplatz_Flugfeld_000003_000330`。

圖 18：Step18 單幀 FrameState 與 overlay 可視化輸出。輸出為：`frame_states.jsonl` 單列、overlay PNG。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig18_frame_state_overlay.png`。

## Step19：整段序列結束後寫出 pipeline summary

Step19 在 27 幀全部處理完成後，呼叫 `write_jsonl(output_dir / "frame_states.jsonl", results)` 寫出所有 frame states，並用 `save_json(output_dir / "pipeline_summary.json", summary)` 寫出整段序列 summary。summary 中包含 `sequence_id`、`frames_processed`、mean latency、max latency、risk event count 與 workflow figure selection 結果。

本次實驗的 `sequence_id` 為 `laf_07ff_000003_000080_000340_verify27`，`frames_processed=27`，`mean_latency_ms≈6680.44`，`max_latency_ms≈14563.38`，`risk_event_count=5`。workflow figure selection summary 對應固定選中的 `frame_id=07_Festplatz_Flugfeld_000003_000330`、`frame_index=25`。此步驟將逐幀 pipeline 結果收束成整段實驗的總結。

圖 19：Step19 sequence-level pipeline summary 輸出。輸出為：`pipeline_summary.json`、完整 `frame_states.jsonl`。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig19_pipeline_summary.png`。

## Step20：使用 frame states 與 GT 進行 evaluation

Step20 由 `evaluate_pipeline()` 讀取 `outputs/<run>/frame_states.jsonl`、`gt_obstacles.csv` 與 `gt_plane.json`，將預測結果與 GT 依 frame 對齊。程式先計算 detection metrics：TP、FP、FN、TN，再得到 `precision = TP/(TP+FP)`、`recall = TP/(TP+FN)`、`F1 = 2PR/(P+R)` 與 accuracy。本次實驗得到 `precision=1.0`、`recall≈0.7778`、`F1≈0.8750`、`accuracy≈0.7778`，TP/FP/FN/TN 為 21/0/6/0。

接著 evaluation 建立 risk confusion matrix，並計算 plane metrics、scale metrics、tracking metrics、latency metrics 與 temporal measurement metrics。tracking continuity 使用 `continuing_object_count / active_object_count`，本次為 `35/46≈0.7609`。另外，geometry metrics 的 `matched_pairs=0`，因此 `height/width/distance` 的 MAE/RMSE 欄位維持 0.0，這代表沒有形成可比較的 geometry 配對，而不是代表真實幾何誤差為零。對 label `07` 的 layer diagnostics 而言，`total_gt_frames=27`，`frontend_roi=10`、`selected_roi=8`、`candidate=7`、`tracked=8`，可作為本次 07ff 驗證序列中各層召回表現的摘要。最後實驗產生三張 plots：`risk_confusion_matrix.png`、`geometry_error_scatter.png`、`latency_histogram.png`。

圖 20：Step20 evaluation 指標與圖表輸出。輸出為：`evaluation_summary.json`、`plots/risk_confusion_matrix.png`、`plots/geometry_error_scatter.png`、`plots/latency_histogram.png`。圖片路徑：`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig20_evaluation_summary.png`。

## 圖檔路徑總表

| 圖號 | 圖示內容 | 圖片路徑 |
|---|---|---|
| 圖 1 | frame 與輸入影像 | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig01_frame_input.png` |
| 圖 2 | motion profile | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig02_motion_profile.png` |
| 圖 3 | calibration basis | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig03_calibration_basis.png` |
| 圖 4 | PIDNet segmentation 與 FOE ROI | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig04_fallback_segmentation_roi.png` |
| 圖 5 | DA3 relative depth | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig05_relative_depth_da3.png` |
| 圖 6 | scale alignment 與 absolute depth | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig06_scale_alignment_absolute_depth.png` |
| 圖 7 | ground plane RANSAC | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig07_ground_plane_ransac.png` |
| 圖 8 | candidate point sampling | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig08_candidate_point_sampling.png` |
| 圖 9 | depth ROI generation | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig09_depth_roi_generation.png` |
| 圖 10 | selected ROI generation | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig10_selected_roi_generation.png` |
| 圖 11 | DBSCAN candidate clusters | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig11_dbscan_candidate_clusters.png` |
| 圖 12 | candidate type classification | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig12_candidate_type_classification.png` |
| 圖 13 | cross-frame matches | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig13_cross_frame_matches.png` |
| 圖 14 | tracker lifecycle | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig14_tracker_lifecycle.png` |
| 圖 15 | temporal snapshot history | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig15_temporal_snapshot_history.png` |
| 圖 16 | raw risk assessment | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig16_raw_risk_assessment.png` |
| 圖 17 | filtered risk events | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig17_filtered_risk_events.png` |
| 圖 18 | FrameState overlay | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig18_frame_state_overlay.png` |
| 圖 19 | pipeline summary | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig19_pipeline_summary.png` |
| 圖 20 | evaluation summary | `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig20_evaluation_summary.png` |

## 主要程式與最後實驗產物

主要程式來源為 `configs/verify_laf_07ff.yaml`、`configs/default.yaml`、`monocular_experiment/main.py`、`monocular_experiment/io_utils.py`、`monocular_experiment/pipeline.py`、`monocular_experiment/foe_roi.py`、`monocular_experiment/segmentation.py`、`monocular_experiment/depth_model.py`、`monocular_experiment/geometry.py`、`monocular_experiment/ground_plane.py`、`monocular_experiment/obstacles.py`、`monocular_experiment/tracking.py`、`monocular_experiment/temporal_measurement.py`、`monocular_experiment/risk.py`、`monocular_experiment/evaluation.py`、`monocular_experiment/visualization.py`、`monocular_experiment/models.py`。

最後實驗產物為 `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/frame_states.jsonl`、`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/pipeline_summary.json`、`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/evaluation_summary.json`、`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/risk_events.csv`、`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig01_frame_input.png` 至 `outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/figures/fig20_evaluation_summary.png`、`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/overlays/*.png`、`outputs/verify_laf_07ff_000003_000080_000340_27_foe_rerun_20260525/plots/*.png`。
