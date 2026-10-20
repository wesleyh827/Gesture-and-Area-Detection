import cv2
from ultralytics import YOLO
import time
import numpy as np
import os
import torch
from torchvision.ops import nms
import sys

# ==============================================================================
# ROI 選擇輔助函數
# ==============================================================================

def _select_roi_gui(frame):
    """從給定的影格上框選單一偵測區域。"""
    print("\n📍 請在畫面上框選感興趣的區域 (ROI)：")
    print("  1. 用滑鼠拖曳框選矩形區域。")
    print("  2. 按 ENTER 或 SPACE 確認選取。")
    print("  3. 按 C 取消選取。\n")

    window_name = 'Select ROI - Press ENTER to confirm'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # selectROI 返回 (x, y, w, h)
    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # 確保視窗關閉

    x, y, w, h = roi

    if w > 0 and h > 0:
        # 轉換為 [x_min, y_min, x_max, y_max] 格式
        h_frame, w_frame = frame.shape[:2]
        x_max = min(int(x + w), w_frame)
        y_max = min(int(y + h), h_frame)
        return [int(x), int(y), x_max, y_max]
    else:
        print("⚠ ROI 框選已取消或無效。")
        return None


def _prompt_manual_roi_input(source_type="mode"):
    """提示使用者輸入單一 ROI 座標。"""
    print(f"\n📍 請手動輸入 ROI 座標 (像素) 用於當前 {source_type}：")
    print("  格式: x_min,y_min,x_max,y_max")

    roi_input = input("  ROI 座標: ").strip()

    try:
        coords = [int(c.strip()) for c in roi_input.split(',')]
        if len(coords) == 4 and coords[0] < coords[2] and coords[1] < coords[3]:
            return coords
        else:
            print("  ⚠ 輸入格式錯誤或順序錯誤，將忽略 ROI。")
    except ValueError:
        print("  ⚠ 座標必須為逗號分隔的整數，將忽略 ROI。")

    return None


# ==============================================================================
# 輔助函式 (檢查點是否在 ROI 內)
# ==============================================================================
def _is_point_in_roi(x, y, roi_coords):
    """檢查點 (x, y) 是否在 ROI [xmin, ymin, xmax, ymax] 內"""
    if roi_coords is None or len(roi_coords) != 4:
        return True  # 如果沒有設定 ROI，則視為在內
    x_min, y_min, x_max, y_max = roi_coords
    return int(x) >= x_min and int(x) <= x_max and int(y) >= y_min and int(y) <= y_max


# ==============================================================================
# 類別：手勢偵測器 (Gesture Detector)
# ==============================================================================

class GestureDetector:

    # 連續偵測相關常數
    NON_UNIFORM_THRESHOLD = 30  # 必須連續偵測到的影格數
    NON_UNIFORMED_ID = 1        # 假設非制服人員的 Class ID 為 1

    def __init__(self, model_path, alert_duration=3.0, debug=False, roi_coords=None):
        """
        初始化手勢偵測器
        """
        print(f"載入自定義模型: {model_path}...")
        self.model = YOLO(model_path)
        self.alert_threshold = alert_duration
        self.alert_start_time = None
        self.is_alerting = False

        # 類別名稱 (從模型中讀取，用於繪製標籤)
        self.class_names = self.model.names if hasattr(self.model, 'names') else {0: 'person', 1: 'person'}

        self.debug = debug
        self.roi_coords = roi_coords
        if self.roi_coords:
            print(f"✓ 已設定 ROI 區域: {self.roi_coords}")

        # 非制服人員連續偵測計數器
        self.non_uniform_counter = 0
        self.is_non_uniform_alerting = False

    def check_hands_above_shoulders(self, keypoints, person_id=0):
        """
        檢查雙手是否都在肩膀以上，並加入距離限制排除平舉。
        """
        # COCO keypoints 索引 (y 座標越小越靠上)
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        confidence_threshold = 0.6

        is_l_shoulder_valid = left_shoulder[2] >= confidence_threshold
        is_r_shoulder_valid = right_shoulder[2] >= confidence_threshold
        if not is_l_shoulder_valid and not is_r_shoulder_valid:
            if self.debug:
                print(f"Person #{person_id+1}: ❌ 兩邊肩膀都無法偵測")
            return False

        # 1. 計算肩膀基準 Y 座標
        if is_l_shoulder_valid and is_r_shoulder_valid:
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        elif is_l_shoulder_valid:
            shoulder_y = left_shoulder[1]
        else:
            shoulder_y = right_shoulder[1]

        # 2. 距離和高度判定參數
        tolerance = 10
        MAX_DISTANCE_FACTOR = 1.5

        # ------------------------ 左手判定 ------------------------
        left_hand_up = False
        if left_wrist[2] >= confidence_threshold:
            left_y = left_wrist[1]
            is_y_up = left_y < (shoulder_y + tolerance)
            is_distance_ok = True

            if left_elbow[2] >= confidence_threshold and is_y_up:
                L_arm = np.sqrt((left_shoulder[0] - left_elbow[0])**2 + (left_shoulder[1] - left_elbow[1])**2)
                D_actual = np.sqrt((left_shoulder[0] - left_wrist[0])**2 + (left_shoulder[1] - left_wrist[1])**2)

                max_allowed_distance = L_arm * MAX_DISTANCE_FACTOR
                is_distance_ok = D_actual < max_allowed_distance

                if self.debug and is_y_up and not is_distance_ok:
                    print(f"Person #{person_id+1}: ❌ 左手平舉被排除 (D_actual={D_actual:.1f} > Max={max_allowed_distance:.1f})")

            left_hand_up = is_y_up and is_distance_ok

        # ------------------------ 右手判定 ------------------------
        right_hand_up = False
        if right_wrist[2] >= confidence_threshold:
            right_y = right_wrist[1]
            is_y_up = right_y < (shoulder_y + tolerance)
            is_distance_ok = True

            if right_elbow[2] >= confidence_threshold and is_y_up:
                R_arm = np.sqrt((right_shoulder[0] - right_elbow[0])**2 + (right_shoulder[1] - right_elbow[1])**2)
                D_actual = np.sqrt((right_shoulder[0] - right_wrist[0])**2 + (right_shoulder[1] - right_wrist[1])**2)

                max_allowed_distance = R_arm * MAX_DISTANCE_FACTOR
                is_distance_ok = D_actual < max_allowed_distance

                if self.debug and is_y_up and not is_distance_ok:
                    print(f"Person #{person_id+1}: ❌ 右手平舉被排除 (D_actual={D_actual:.1f} > Max={max_allowed_distance:.1f})")

            right_hand_up = is_y_up and is_distance_ok

        # ------------------------ 最終判定 ------------------------
        detected_hands = 0
        raised_hands = 0

        if left_wrist[2] >= confidence_threshold:
            detected_hands += 1
            if left_hand_up:
                raised_hands += 1

        if right_wrist[2] >= confidence_threshold:
            detected_hands += 1
            if right_hand_up:
                raised_hands += 1

        result = (detected_hands > 0) and (detected_hands == raised_hands)

        if self.debug:
            print(f"Person #{person_id+1}: 偵測到手:{detected_hands}, 舉起手:{raised_hands} -> {'✅ ALERT' if result else '❌ NORMAL'}")

        return result

    def draw_info(self, frame, hands_up, duration=0, is_static=False, person_info=""):
        """在畫面上繪製資訊"""
        # 兩種警報：舉手警報 (hands_up) 和非制服警報 (self.is_non_uniform_alerting)
        is_alert_active = hands_up or self.is_non_uniform_alerting

        if is_alert_active:
            # 優先顯示非制服警報 (如果有)
            if self.is_non_uniform_alerting:
                text = f"Alert: Non-uniformed Person (Consecutive {self.non_uniform_counter}/{self.NON_UNIFORM_THRESHOLD} frames)"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
                cv2.putText(frame, text, (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)

            # 如果是非制服警報，則不再處理舉手警報的計時器顯示
            elif hands_up:
                if is_static:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
                    cv2.putText(frame, f"WARNING: ALERT POSE DETECTED!{person_info}", (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
                elif duration >= self.alert_threshold:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
                    cv2.putText(frame, f"WARNING: ALERT DETECTED!{person_info}", (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
                else:
                    progress = (duration / self.alert_threshold) * 100
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 165, 255), -1)
                    cv2.putText(frame, f"Hands Up{person_info}: {duration:.1f}s / {self.alert_threshold:.1f}s ({progress:.0f}%)",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # 正常狀態，顯示非制服人員計數
            text = f"Status: Normal (Non-uniform Count: {self.non_uniform_counter}/{self.NON_UNIFORM_THRESHOLD} frames)"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    # ==============================================================================
    # process_frame (整合連續偵測、ROI 篩選和 NMS 清理)
    # ==============================================================================
    def process_frame(self, frame, is_static=False):

        ALERT_PERSON_CONFIDENCE_THRESHOLD = 0.5

        # 執行 YOLO 推理。iou=0.8 是 YOLO 內建的 NMS，後面再手動處理衝突框。
        results = self.model(frame, conf=ALERT_PERSON_CONFIDENCE_THRESHOLD, iou=0.8, verbose=False)

        hands_up = False
        duration = 0
        alert_person_id = -1
        person_statuses = []

        annotated_frame = frame.copy()

        # --- 步驟 1: 確保 ROI 框無條件顯示 ---
        if self.roi_coords:
            x1, y1, x2, y2 = self.roi_coords
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # --- 獲取所有原始數據 (包含信心度 confs) ---
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.array([])
        keypoints_data = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None and results[0].keypoints.data is not None else np.array([])
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else np.array([])
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else np.array([])

        if len(boxes) == 0 or len(keypoints_data) == 0:
            if not hands_up and not is_static and self.alert_start_time is not None:
                self.alert_start_time = None
                self.is_alerting = False

            self.non_uniform_counter = 0
            self.is_non_uniform_alerting = False

            return annotated_frame, hands_up

        # --- 步驟 2: ROI 篩選與初步數據收集 ---
        targets_to_check_raw = []

        for i in range(len(boxes)):
            if i >= len(keypoints_data) or keypoints_data[i].shape[0] < 17 or i >= len(confs):
                continue

            box = boxes[i]
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2

            is_in_roi = _is_point_in_roi(x_center, y_center, self.roi_coords)

            if is_in_roi:
                targets_to_check_raw.append({
                    'original_index': i,
                    'keypoints': keypoints_data[i],
                    'class_id': classes[i],
                    'conf': confs[i]
                })
            elif self.debug:
                cv2.putText(annotated_frame, "OUT_ROI", (int(box[0]), int(box[1]) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # --- 步驟 3: NMS 清理 (強制單一目標追蹤) ---
        targets_to_check = targets_to_check_raw

        if len(targets_to_check) > 1:
            # 1. 提取通過 ROI 篩選的目標數據
            roi_boxes_np = np.array([boxes[t['original_index']] for t in targets_to_check])
            roi_confs_np = np.array([t['conf'] for t in targets_to_check])

            # 2. 轉換為 PyTorch 張量
            boxes_tensor = torch.tensor(roi_boxes_np, dtype=torch.float32)
            confs_tensor = torch.tensor(roi_confs_np, dtype=torch.float32)

            # 3. 執行 NMS 清理，高 IoU 閾值 (0.75) 用於強制合併重疊的框
            nms_indices_tensor = nms(boxes_tensor, confs_tensor, iou_threshold=0.75)

            # 4. 轉換回 Numpy 索引並過濾 targets_to_check
            nms_indices = nms_indices_tensor.cpu().numpy().astype(int)
            targets_to_check = [targets_to_check[i] for i in nms_indices]

        # --- 步驟 4: 更新連續計數器 ---
        current_found_non_uniform = any(int(t['class_id']) == self.NON_UNIFORMED_ID for t in targets_to_check)

        if current_found_non_uniform:
            self.non_uniform_counter = min(self.non_uniform_counter + 1, self.NON_UNIFORM_THRESHOLD)
        else:
            self.non_uniform_counter = 0

        # 觸發非制服警報狀態
        if self.non_uniform_counter == self.NON_UNIFORM_THRESHOLD:
            self.is_non_uniform_alerting = True
            if self.debug:
                print(f"🚨 非制服警報觸發！連續 {self.NON_UNIFORM_THRESHOLD} 幀。")
        elif self.non_uniform_counter == 0:
            self.is_non_uniform_alerting = False

        # --- 步驟 5: 舉手警報邏輯與狀態彙總 ---
        current_person_id = 0

        if len(targets_to_check) > 0:
            for target in targets_to_check:
                original_idx = target['original_index']
                person_keypoints = target['keypoints']
                person_class = int(target['class_id'])
                person_class_name = self.class_names.get(person_class, "Unknown")

                # 舉手警報只對制服人員（Class 0）生效
                is_alert = self.check_hands_above_shoulders(person_keypoints, current_person_id) if person_class == 0 else False

                person_statuses.append({
                    'id': original_idx,
                    'visual_id': current_person_id + 1,
                    'is_alert': is_alert,
                    'class_name': person_class_name,
                    'keypoints': person_keypoints,
                })

                if is_alert:
                    hands_up = True
                    alert_person_id = original_idx

                    if not is_static:
                        if self.alert_start_time is None:
                            self.alert_start_time = time.time()

                        duration = time.time() - self.alert_start_time

                        if duration >= self.alert_threshold and not self.is_alerting:
                            self.is_alerting = True
                            if self.debug:
                                print("\n🚨 舉手警示觸發！")

                current_person_id += 1

        # 如果沒有舉手警報
        if not hands_up:
            if not is_static and self.alert_start_time is not None:
                self.alert_start_time = None
            self.is_alerting = False

        # 繪製邊界框、骨架和頂部資訊
        annotated_frame = self.draw_person_labels(annotated_frame, person_statuses, results[0], duration, is_static)

        info_text = ""
        if hands_up:
            matched = [p for p in person_statuses if p['id'] == alert_person_id]
            if matched:
                info_text = f" (Person #{matched[0]['visual_id']})"
        annotated_frame = self.draw_info(annotated_frame, hands_up, duration, is_static, info_text)

        return annotated_frame, hands_up

    # ==============================================================================
    # draw_person_labels (使用 visual_id 和原始索引)
    # ==============================================================================
    def draw_person_labels(self, frame, person_statuses, result, duration=0, is_static=False):
        """在每個人身上繪製編號和狀態標籤"""
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
        else:
            return frame

        for person_status in person_statuses:
            # person_id 是原始索引 (original_idx)
            person_id = person_status['id']
            is_alert = person_status['is_alert']
            class_name = person_status['class_name']
            visual_id = person_status['visual_id']  # 自定義編號 (#1, #2, ...)

            # 使用原始索引來取得正確的邊界框座標
            try:
                x1, y1, x2, y2 = boxes[person_id][:4].astype(int)
            except IndexError:
                continue

            x_min, y_min = x1, y1
            x_max, y_max = x2, y2

            # 警報狀態判斷
            show_non_uniform_alert = self.is_non_uniform_alerting and (class_name == self.class_names.get(self.NON_UNIFORMED_ID, 'non-uniformed_person'))

            show_alert = is_alert and (is_static or duration >= self.alert_threshold)
            show_warning = is_alert and not is_static and duration < self.alert_threshold

            # 根據制服和警報狀態設定顏色
            if class_name == self.class_names.get(0, 'uniformed_person'):  # Class 0: 制服人員
                base_color = (255, 165, 0)  # 橙色
            else:  # Class 1: 非制服人員
                base_color = (0, 255, 255)  # 黃色

            if show_alert or show_non_uniform_alert:
                color = (0, 0, 255)  # 警報時為紅色
                label_bg_color = (0, 0, 200)
            elif show_warning:
                color = (0, 165, 255)  # 警告時為橙色
                label_bg_color = (0, 140, 220)
            else:
                color = base_color
                label_bg_color = tuple(int(c * 0.8) for c in base_color)

            thickness = 3 if show_alert or show_non_uniform_alert else 2

            # 繪製自定義邊界框
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

            # 使用 visual_id 作為標籤編號
            label = f"#{visual_id} {class_name}"
            if show_alert:
                label += " [HAND ALERT!]"
            elif show_non_uniform_alert:
                label += " [NON-UNIFORM ALERT!]"
            elif show_warning:
                label += f" [Warning {duration:.1f}s]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            label_y = y_min - 10
            if label_y < text_height + 10:
                label_y = y_min + text_height + 10

            cv2.rectangle(
                frame,
                (x_min, label_y - text_height - 10),
                (x_min + text_width + 10, label_y + 5),
                label_bg_color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x_min + 5, label_y - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )

            # 繪製警報三角形 (簡略)
            if show_alert or show_non_uniform_alert:
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                triangle_size = 40
                pts = np.array([
                    [center_x, center_y - triangle_size],
                    [center_x - triangle_size, center_y + triangle_size],
                    [center_x + triangle_size, center_y + triangle_size]
                ], np.int32)

                cv2.fillPoly(frame, [pts], (0, 0, 255))
                cv2.polylines(frame, [pts], True, (255, 255, 255), 3)

                cv2.putText(
                    frame,
                    "!",
                    (center_x - 8, center_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    3
                )

        return frame

    def _setup_roi_in_execution(self, source, is_webcam=False):
        """
        內部函數：在執行開始時獲取第一幀並設定 ROI。
        """
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"❌ 無法開啟影像來源: {source}")
            return None, None

        # 設定解析度 (Webcam only)
        if is_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 讀取第一幀
        ret, first_frame = cap.read()

        if not ret:
            print("❌ 無法讀取影像第一幀。")
            cap.release()
            return None, None

        # 1. 執行 ROI 設定
        if self.roi_coords is None:
            # 嘗試使用 GUI 進行視覺框選
            self.roi_coords = _select_roi_gui(first_frame.copy())

            if self.roi_coords is None:
                # 如果 GUI 框選失敗或被取消，切換到手動輸入
                self.roi_coords = _prompt_manual_roi_input(source_type="即時偵測")

        # 釋放/重置 cap (Webcam 必須釋放並重新開啟)
        if is_webcam:
            cap.release()
        else:
            # 影片模式下，將播放頭重置到第一幀
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return first_frame, cap

    def run_webcam(self, camera_id=0):
        """
        使用攝影機即時偵測
        """

        print(f"\n開始使用攝影機 {camera_id}")

        # 執行 ROI 設定
        first_frame, cap = self._setup_roi_in_execution(camera_id, is_webcam=True)

        # 如果 ROI 設置階段失敗 (例如無法開啟 Webcam)，則退出
        if cap is None:
            return

        # 重新開啟 Webcam，確保串流是持續的
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ 無法開啟攝影機 {camera_id}")
            return

        print("請雙手舉過肩膀3秒以上來測試")
        print("按 'q' 離開\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取影像")
                break

            processed_frame, _ = self.process_frame(frame, is_static=False)

            cv2.imshow('Gesture Detection - Webcam', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("程式結束")

    def run_video(self, video_path, save_output=True, output_dir='output'):
        """
        處理影片檔案
        """
        print(f"\n處理影片: {video_path}")

        # 執行 ROI 設定
        first_frame, cap = self._setup_roi_in_execution(video_path, is_webcam=False)

        if cap is None:
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"影片資訊: {width}x{height} @ {fps}fps, 總影格數: {total_frames}")

        out = None
        output_path = None
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(video_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_result{ext}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 將儲存結果至: {output_path}")

        alert_events = []
        frame_count = 0

        print("\n開始處理影片...")
        print("按 'q' 提前結束，按 'p' 暫停")

        while True:
            ret, frame = cap.read()
            if not ret:
                # 如果是最後一幀，嘗試將最後的警報事件結束
                if alert_events and not alert_events[-1]['end']:
                    alert_events[-1]['end_frame'] = frame_count
                    alert_events[-1]['end_time'] = frame_count / fps
                    alert_events[-1]['end'] = True
                break

            frame_count += 1

            processed_frame, hands_up = self.process_frame(frame, is_static=False)

            # 判斷是舉手警報還是非制服警報
            is_any_alert_active = self.is_alerting or self.is_non_uniform_alerting

            if is_any_alert_active and (not alert_events or alert_events[-1]['end']):
                alert_events.append({
                    'start_frame': frame_count,
                    'start_time': frame_count / fps,
                    'end': False,
                    'type': 'Non-Uniform' if self.is_non_uniform_alerting else 'Hand-Gesture'
                })
            elif not is_any_alert_active and alert_events and not alert_events[-1]['end']:
                alert_events[-1]['end_frame'] = frame_count
                alert_events[-1]['end_time'] = frame_count / fps
                alert_events[-1]['end'] = True

            if out is not None:
                out.write(processed_frame)

            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"\r處理中... {progress:.1f}% ({frame_count}/{total_frames})", end='')

            cv2.imshow('Gesture Detection - Video', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n\n⚠ 使用者中斷處理")
                break
            elif key == ord('p'):
                print("\n暫停中，按任意鍵繼續...")
                cv2.waitKey(0)

        print("\n\n影片處理完成！")

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        print(f"\n{'=' * 50}")
        print("處理統計:")
        print(f"總影格數: {frame_count}")
        print(f"影片長度: {frame_count / fps:.2f} 秒")
        print(f"偵測到警示事件: {len(alert_events)} 次")

        if alert_events:
            print("\n警示事件詳情:")
            for i, event in enumerate(alert_events, 1):
                start_time = event['start_time']
                end_time = event.get('end_time', frame_count / fps)
                duration = end_time - start_time
                print(f" 事件 {i} ({event['type']}): {start_time:.2f}s - {end_time:.2f}s (持續 {duration:.2f}s)")

        if save_output and output_path:
            print(f"\n💾 結果影片已儲存至: {output_path}")

        print(f"{'=' * 50}\n")

    def process_image_batch(self, image_folder, save_output=True, output_dir='output'):
        """批次處理多張圖片"""
        print(f"\n批次處理資料夾: {image_folder}")

        if not os.path.exists(image_folder):
            print(f"❌ 找不到資料夾: {image_folder}")
            return

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, file))

        if not image_files:
            print("❌ 資料夾中沒有找到圖片檔案")
            return

        print(f"找到 {len(image_files)} 張圖片\n")

        # ROI 設置 (批次模式)
        if self.roi_coords is None:
            first_frame = cv2.imread(image_files[0])

            if first_frame is not None:
                self.roi_coords = _select_roi_gui(first_frame.copy())

            if self.roi_coords is None:
                self.roi_coords = _prompt_manual_roi_input(source_type="批次處理")

        alert_count = 0

        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] ", end="")

            frame = cv2.imread(image_path)
            if frame is None:
                print(f"⚠ 跳過無法讀取的圖片: {image_path}")
                continue

            annotated_frame, hands_up = self.process_frame(frame, is_static=True)

            # 檢查兩種警報狀態
            is_alert_active = hands_up or self.is_non_uniform_alerting

            if is_alert_active:
                alert_count += 1
                alert_type = 'Non-Uniform' if self.is_non_uniform_alerting else 'Hand-Gesture'
                print(f"✅ {os.path.basename(image_path)} - 偵測到警示 ({alert_type})")
            else:
                print(f"❌ {os.path.basename(image_path)} - 未偵測到警示")

            if save_output:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_result{ext}")
                cv2.imwrite(output_path, annotated_frame)

        print(f"\n{'=' * 50}")
        print("處理完成！")
        print(f"總圖片數: {len(image_files)}")
        print(f"偵測到警示事件: {alert_count} 張")
        print(f"未偵測到: {len(image_files) - alert_count} 張")
        if save_output:
            print(f"💾 結果已儲存至: {output_dir}/")
        print(f"{'=' * 50}")

    def process_image(self, image_path, save_output=True, output_dir='output'):
        """處理單張圖片"""
        print(f"\n處理圖片: {image_path}")

        if not os.path.exists(image_path):
            print(f"❌ 找不到圖片: {image_path}")
            return False

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ 無法讀取圖片: {image_path}")
            return False

        # ROI 設置 (單張圖片模式)
        if self.roi_coords is None:
            self.roi_coords = _select_roi_gui(frame.copy())

            if self.roi_coords is None:
                self.roi_coords = _prompt_manual_roi_input(source_type="單張圖片")

        # 處理
        annotated_frame, hands_up = self.process_frame(frame, is_static=True)

        is_alert_active = hands_up or self.is_non_uniform_alerting

        print(f"{'✅ 偵測到警示姿勢！' if is_alert_active else '❌ 未偵測到警示姿勢'}")

        output_path = None
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_result{ext}")
            cv2.imwrite(output_path, annotated_frame)
            print(f"💾 結果已儲存至: {output_path}")

        cv2.imshow('Gesture Detection - Image', annotated_frame)
        print("按任意鍵繼續...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return hands_up


# ==============================================================================
# 主程式 (這裡設定您的模型路徑)
# ==============================================================================
def main():
    """主程式"""

    # 設定您的自定義模型路徑
    CUSTOM_MODEL_PATH = r'C:\Users\hti_intern4\Desktop\gesture_uniform\last.pt'

    if CUSTOM_MODEL_PATH == 'YOUR_BEST_PT_PATH_HERE' or not os.path.exists(CUSTOM_MODEL_PATH):
        print("=" * 50)
        print("❌ 錯誤：請在 main() 函式中設定正確的 CUSTOM_MODEL_PATH！")
        print(f"目前設定: {CUSTOM_MODEL_PATH}")
        print("=" * 50)
        return

    print("=" * 50)
    print("手勢警示偵測系統")
    print("=" * 50)

    # 建立偵測器實例，傳入自定義模型路徑
    detector = GestureDetector(
        model_path=CUSTOM_MODEL_PATH,
        alert_duration=3.0,
        debug=True
    )

    # 選擇模式
    print("\n請選擇模式:")
    print("1. 使用攝影機即時偵測")
    print("2. 處理影片檔案")
    print("3. 處理單張圖片")
    print("4. 批次處理多張圖片")

    choice = input("\n請輸入選項 (1-4): ").strip()

    # 執行模式
    if choice == '1':
        detector.run_webcam(camera_id=0)

    elif choice == '2':
        video_path = input("請輸入影片路徑: ").strip()
        save = input("是否儲存結果影片? (y/n, 預設:y): ").strip().lower()
        save_output = save != 'n'
        detector.run_video(video_path, save_output=save_output)
    elif choice == '3':
        image_path = input("請輸入圖片路徑: ").strip()
        detector.process_image(image_path, save_output=True)
    elif choice == '4':
        folder_path = input("請輸入圖片資料夾路徑: ").strip()
        detector.process_image_batch(folder_path, save_output=True)
    else:
        print("❌ 無效的選項")


if __name__ == "__main__":
    main()
