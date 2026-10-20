# 👋 手勢與制服警示偵測系統

## 🌟 專案概述

本專案是一個基於 **YOLOv8** (假設使用) 模型和 **OpenCV** 的即時影像分析解決方案。它能同時執行 **人體姿態估計** (用於手勢偵測) 和 **物件分類** (用於制服/非制服人員識別)，並提供可自定義的 **感興趣區域 (ROI)** 篩選功能，以專注於特定監控範圍。

### 核心功能

* **手勢警示偵測 (Hand Gesture Alert)**：偵測人員是否做出特定警示手勢 (例如：雙手舉過肩膀並持續超過預設時間，預設為 $3.0$ 秒)。
* **制服合規性偵測 (Uniform Compliance)**：識別人員是否為非制服人員 (Class ID 1)，並在連續多個影格 (預設 $30$ 幀) 偵測到時觸發警報。
* **即時/影片/圖片處理**：支援 Webcam 即時串流、影片檔案和圖片批次處理。
* **ROI 區域篩選**：允許使用者通過 GUI 或手動輸入框選偵測區域，**僅處理區域內**的目標，提高效率和準確性。

## 🛠️ 環境與依賴

您需要安裝以下 Python 套件：

```bash
pip install opencv-python ultralytics numpy torch torchvision

```

* `ultralytics`: 用於 YOLO 模型。
* `opencv-python`: 用於影像處理、Webcam 存取和 GUI 顯示。
* `torch`, `torchvision`: 用於處理 NMS (非極大值抑制) 邏輯。

## 🚀 使用方法

### 步驟 1: 配置模型路徑

在檔案末尾的 `main()` 函式中，請將 `CUSTOM_MODEL_PATH` 變數設定為您訓練好的 **YOLO 模型** (例如：`last.pt`) 的絕對路徑。

```python
# ==============================================================================
# 主程式 (這裡設定您的模型路徑)
# ==============================================================================
def main():
    """主程式"""
    
    # 🎯 核心修改: 設定您的自定義模型路徑！
    CUSTOM_MODEL_PATH = r'C:\Users\hti_intern4\Desktop\gesture_uniform\last.pt' 
    # ... (其餘程式碼)

```

### 步驟 2: 執行程式

運行 Python 腳本：

```bash
python your_script_name.py

```

### 步驟 3: 選擇模式與設定

程式執行後，會要求您選擇一種運行模式：

1. **使用攝影機即時偵測 (`run_webcam`)**
2. **處理影片檔案 (`run_video`)**
3. **處理單張圖片 (`process_image`)**
4. **批次處理多張圖片 (`process_image_batch`)**

選擇模式後，程式將進入 **ROI 選擇階段**：

* **GUI 框選**：畫面會彈出，請使用滑鼠拖曳來框選感興趣的區域 (ROI)。
* **`ENTER` / `SPACE**`: 確認選取。
* **`C`**: 取消選取。


* **手動輸入**：如果 GUI 框選失敗或被取消，您可以手動輸入座標，格式為：`x_min,y_min,x_max,y_max`。

## ⚙️ 偵測邏輯詳解

### 1. 手勢偵測 (`check_hands_above_shoulders`)

此函式使用姿態估計的 **關鍵點** 來判斷手勢：

* **關鍵點依據**：程式碼依賴 COCO 格式的關鍵點索引 (肩、肘、腕)。
* **核心判斷**：
1. 計算左右肩膀的平均 Y 座標 (`shoulder_y`) 作為基準線。
2. 檢查手腕 (Wrist) 的 Y 座標是否**小於**基準線 (Y 座標越小越靠上)。
3. **排除平舉**：引入距離限制。計算手腕到肩膀的實際距離 (`D_actual`)，並與手臂長度 (肩膀到手肘的距離 `L_arm` 乘上一個係數 `MAX_DISTANCE_FACTOR = 1.5`) 比較，防止將手臂平舉視為舉手。



### 2. 制服/非制服人員偵測

* **模型輸出分類**：模型輸出預計有兩個類別 (Class)：
* **Class 0**: 制服人員 (Uniformed Person)
* **Class 1**: 非制服人員 (Non-Uniformed Person, `NON_UNIFORMED_ID = 1`)


* **連續偵測機制**：
* 使用計數器 `self.non_uniform_counter`。
* 當連續偵測到非制服人員達到閾值 (`NON_UNIFORM_THRESHOLD = 30` 幀) 時，觸發 **非制服警報** (`self.is_non_uniform_alerting = True`)。



### 3. 目標篩選與處理

* **ROI 篩選**：僅處理其中心點落在設定 ROI 內的邊界框。
* **NMS (非極大值抑制) 清理**：在 ROI 篩選後，使用 PyTorch 的 `torchvision.ops.nms` 函式來進一步清理高度重疊的邊界框 (IoU 閾值為 $0.75$)，確保即使有重疊偵測，每個目標也只被處理一次，特別是在多人場景中。

## 💡 參數調整 (在 `GestureDetector` 類別中)

| 參數名稱 | 程式碼位置 | 預設值 | 說明 |
| --- | --- | --- | --- |
| `alert_duration` | `__init__` (in `main` setup) | `3.0` 秒 | 手勢被判定為警示所需的最小持續時間。 |
| `NON_UNIFORM_THRESHOLD` | `GestureDetector` Class | `30` 幀 | 觸發非制服警報所需的連續偵測影格數。 |
| `NON_UNIFORMED_ID` | `GestureDetector` Class | `1` | 非制服人員的類別 ID。 |
| `confidence_threshold` | `check_hands_above_shoulders` | `0.6` | 判斷關鍵點是否可信的信心度閾值。 |
| `MAX_DISTANCE_FACTOR` | `check_hands_above_shoulders` | `1.5` | 用於排除平舉手勢的距離容忍因子。 |
| `ALERT_PERSON_CONFIDENCE_THRESHOLD` | `process_frame` | `0.5` | YOLO 偵測的初始信心度閾值。 |

## 🖼️ 輸出視覺化

* **邊界框**：根據人員類別 (制服/非制服) 和狀態 (警報/警告/正常) 變色。
* **警報訊息**：在畫面頂部顯示即時狀態：
* **紅色**：已觸發手勢或非制服警報。
* **橙色**：手勢偵測中 (計時器運行)。


* **ROI 框**：用青色框線 (255, 255, 0) 標示選定的偵測區域。
* **警報標籤**：在觸發警報的目標中心點繪製紅色感嘆號三角形標記。

---

請問您是否需要我針對程式碼中的**關鍵點判斷邏輯**（例如：手腕、肩膀的座標索引）提供一個**示意圖**，以幫助您更好地理解 `check_hands_above_shoulders` 函式的運作？
