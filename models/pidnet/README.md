# PIDNet 官方權重放置位置

此資料夾用來放置官方 PIDNet 預訓練權重。

目前本研究預設使用：

- `PIDNet_S_Cityscapes_test.pt`

建議檔名與完整路徑如下：

```text
models/pidnet/PIDNet_S_Cityscapes_test.pt
```

選擇原因：

- `PIDNet-S` 是官方較輕量、即時性較高的版本，適合本研究的車載與逐幀處理場景。
- `Cityscapes` 是道路場景語意分割資料集，直接包含 `road` 與 `sidewalk` 類別，最符合本研究的可通行區域偵測需求。
- 官方 README 在自訂影像推論範例中，也以 Cityscapes 權重作為道路場景應用示例。

官方來源：

- PIDNet 官方倉庫：<https://github.com/XuJiacong/PIDNet>
- 權重下載說明請見官方 README 的 `Models` 章節
