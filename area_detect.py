import cv2
from ultralytics import YOLO
import time
import numpy as np
import os

class AreaPersonDetector:
    def __init__(self, model_size='n', stay_duration=3.0):
        """
        初始化區域人員偵測器
        
        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
            stay_duration: 需要停留的時間（秒）
        """
        print(f"載入 YOLOv8{model_size} 模型...")
        self.model = YOLO(f'yolov8{model_size}.pt')  # 使用物件偵測模型
        self.stay_threshold = stay_duration
        
        # 偵測區域（初始為 None，需要用戶設定）
        self.detection_area = None
        
        # 追蹤每個人的資訊
        self.tracked_people = {}  # {person_id: {'enter_time': timestamp, 'counted': bool}}
        self.total_count = 0  # 總計數
        self.next_person_id = 1
        
        print(f"✓ 模型載入完成")
        print(f"設定：需在區域內停留 {stay_duration} 秒才計數")
    
    def set_detection_area(self, video_path):
        """
        讓使用者在影片第一幀上框選偵測區域
        
        Args:
            video_path: 影片路徑
        
        Returns:
            bool: 是否成功設定區域
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ 無法讀取影片第一幀")
            return False
        
        print("\n📍 請在畫面上框選偵測區域：")
        print("   1. 用滑鼠拖曳框選矩形區域")
        print("   2. 按 'r' 重新框選")
        print("   3. 按 'c' 確認區域")
        print("   4. 按 'q' 取消\n")
        
        # 使用 OpenCV 的 selectROI
        cv2.namedWindow('Select Detection Area', cv2.WINDOW_NORMAL)
        roi = cv2.selectROI('Select Detection Area', frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:  # 檢查是否有效
            self.detection_area = {
                'x': int(roi[0]),
                'y': int(roi[1]),
                'width': int(roi[2]),
                'height': int(roi[3])
            }
            print(f"✓ 偵測區域已設定: ({self.detection_area['x']}, {self.detection_area['y']}) "
                  f"寬={self.detection_area['width']}, 高={self.detection_area['height']}")
            return True
        else:
            print("❌ 未選擇有效區域")
            return False
    
    def is_person_in_area(self, bbox):
        """
        判斷人是否在偵測區域內
        
        Args:
            bbox: [x1, y1, x2, y2] 人的邊界框
        
        Returns:
            bool: 是否在區域內
        """
        if self.detection_area is None:
            return False
        
        # 計算人的中心點
        person_center_x = (bbox[0] + bbox[2]) / 2
        person_center_y = (bbox[1] + bbox[3]) / 2
        
        # 檢查中心點是否在區域內
        area_x1 = self.detection_area['x']
        area_y1 = self.detection_area['y']
        area_x2 = area_x1 + self.detection_area['width']
        area_y2 = area_y1 + self.detection_area['height']
        
        return (area_x1 <= person_center_x <= area_x2 and 
                area_y1 <= person_center_y <= area_y2)
    
    def get_person_id(self, bbox, current_time):
        """
        簡單的人員追蹤（根據位置相近度）
        
        Args:
            bbox: [x1, y1, x2, y2] 人的邊界框
            current_time: 當前時間戳
        
        Returns:
            int: 人員ID
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # 尋找最接近的已追蹤人員（簡單的追蹤邏輯）
        min_distance = float('inf')
        matched_id = None
        
        for pid, info in self.tracked_people.items():
            if 'last_position' in info:
                last_x, last_y = info['last_position']
                distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                
                # 如果距離小於閾值，認為是同一個人
                if distance < 100 and distance < min_distance:
                    min_distance = distance
                    matched_id = pid
        
        if matched_id is None:
            # 新的人
            matched_id = self.next_person_id
            self.next_person_id += 1
            self.tracked_people[matched_id] = {
                'enter_time': None,
                'counted': False,
                'last_seen': current_time
            }
        
        # 更新位置和最後出現時間
        self.tracked_people[matched_id]['last_position'] = (center_x, center_y)
        self.tracked_people[matched_id]['last_seen'] = current_time
        
        return matched_id
    
    def clean_old_tracks(self, current_time, timeout=2.0):
        """清除超過一定時間未出現的追蹤"""
        to_remove = []
        for pid, info in self.tracked_people.items():
            if current_time - info['last_seen'] > timeout:
                to_remove.append(pid)
        
        for pid in to_remove:
            del self.tracked_people[pid]
    
    def draw_detection_area(self, frame):
        """在畫面上繪製偵測區域"""
        if self.detection_area is None:
            return frame
        
        x = self.detection_area['x']
        y = self.detection_area['y']
        w = self.detection_area['width']
        h = self.detection_area['height']
        
        # 繪製半透明區域
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # 繪製邊框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # 繪製區域標籤
        cv2.putText(frame, "Detection Area", (x + 5, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def draw_info(self, frame):
        """繪製統計資訊"""
        # 背景框
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        
        # 統計資訊
        cv2.putText(frame, f"Total Count: {self.total_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 當前在區域內的人數
        current_in_area = sum(1 for info in self.tracked_people.values() 
                            if info['enter_time'] is not None)
        cv2.putText(frame, f"Current in Area: {current_in_area}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 追蹤中的人數
        cv2.putText(frame, f"Tracking: {len(self.tracked_people)}", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame, current_time):
        """
        處理單一影格
        
        Args:
            frame: 輸入影格
            current_time: 當前時間戳
        
        Returns:
            annotated_frame: 標註後的影格
        """
        # 執行人員偵測
        results = self.model(frame, verbose=False)
        
        # 繪製偵測區域
        annotated_frame = self.draw_detection_area(frame.copy())
        
        # 處理偵測到的人
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                # 只處理 person (class 0) 且信心度 > 0.5
                if int(cls) == 0 and conf > 0.5:
                    # 取得或分配人員ID
                    person_id = self.get_person_id(box, current_time)
                    
                    # 判斷是否在區域內
                    in_area = self.is_person_in_area(box)
                    
                    if in_area:
                        # 在區域內
                        if self.tracked_people[person_id]['enter_time'] is None:
                            # 剛進入區域
                            self.tracked_people[person_id]['enter_time'] = current_time
                            print(f"Person #{person_id} 進入區域")
                        
                        # 計算停留時間
                        stay_duration = current_time - self.tracked_people[person_id]['enter_time']
                        
                        # 檢查是否達到計數條件
                        if stay_duration >= self.stay_threshold and not self.tracked_people[person_id]['counted']:
                            self.tracked_people[person_id]['counted'] = True
                            self.total_count += 1
                            print(f"✓ Person #{person_id} 已計數！(停留 {stay_duration:.1f}秒) - 總計: {self.total_count}")
                        
                        # 繪製邊界框（根據狀態選擇顏色）
                        if self.tracked_people[person_id]['counted']:
                            color = (0, 255, 0)  # 綠色：已計數
                            status = "COUNTED"
                        else:
                            color = (0, 165, 255)  # 橘色：計時中
                            status = f"{stay_duration:.1f}s"
                    else:
                        # 不在區域內
                        if self.tracked_people[person_id]['enter_time'] is not None:
                            # 離開區域，重置
                            print(f"Person #{person_id} 離開區域")
                            self.tracked_people[person_id]['enter_time'] = None
                            self.tracked_people[person_id]['counted'] = False
                        
                        color = (128, 128, 128)  # 灰色：區域外
                        status = "Outside"
                    
                    # 繪製邊界框
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 繪製標籤
                    label = f"P{person_id} [{status}]"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 清理舊的追蹤
        self.clean_old_tracks(current_time)
        
        # 繪製統計資訊
        annotated_frame = self.draw_info(annotated_frame)
        
        return annotated_frame
    
    def process_video(self, video_path, save_output=True, output_dir='output', show_video=True):
        """
        處理影片
        
        Args:
            video_path: 影片路徑
            save_output: 是否儲存結果
            output_dir: 輸出資料夾
            show_video: 是否顯示即時畫面
        """
        print(f"\n處理影片: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 無法開啟影片")
            return
        
        # 取得影片資訊
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"影片資訊: {width}x{height} @ {fps}fps, 總長度: {total_frames/fps/60:.1f} 分鐘")
        
        # 準備輸出
        out = None
        output_path = None
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(video_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_area_result{ext}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 將儲存結果至: {output_path}")
        
        print("\n開始處理...")
        if show_video:
            print("按 'q' 結束，按 'p' 暫停")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps
            
            # 處理影格
            processed_frame = self.process_frame(frame, current_time)
            
            # 儲存
            if out is not None:
                out.write(processed_frame)
            
            # 顯示進度
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                eta = (total_frames - frame_count) / fps_actual / 60
                print(f"\r進度: {progress:.1f}% | 速度: {fps_actual:.1f} fps | "
                      f"剩餘: {eta:.1f} 分 | 計數: {self.total_count}", end='')
            
            # 顯示畫面
            if show_video:
                cv2.imshow('Area Person Detection', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n\n⚠ 使用者中斷")
                    break
                elif key == ord('p'):
                    print("\n暫停...")
                    cv2.waitKey(0)
        
        # 清理
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # 顯示結果
        print(f"\n\n{'='*60}")
        print(f"處理完成！")
        print(f"總影格數: {frame_count}")
        print(f"處理時間: {(time.time() - start_time)/60:.1f} 分鐘")
        print(f"✓ 在區域內停留超過 {self.stay_threshold} 秒的人數: {self.total_count}")
        
        if save_output and output_path:
            print(f"\n💾 結果已儲存至: {output_path}")
        
        print(f"{'='*60}\n")


def main():
    """主程式"""
    print("=" * 60)
    print("區域人員停留偵測系統")
    print("=" * 60)
    
    # 取得影片路徑
    video_path = input("\n請輸入影片路徑: ").strip()
    
    if not os.path.exists(video_path):
        print("❌ 找不到影片檔案")
        return
    
    # 設定停留時間
    stay_duration = input("請輸入需要停留的秒數 (預設: 3): ").strip()
    try:
        stay_duration = float(stay_duration) if stay_duration else 3.0
    except:
        stay_duration = 3.0
    
    # 建立偵測器
    detector = AreaPersonDetector(model_size='n', stay_duration=stay_duration)
    
    # 設定偵測區域
    print("\n正在開啟影片以設定偵測區域...")
    if not detector.set_detection_area(video_path):
        print("❌ 未設定偵測區域，程式結束")
        return
    
    # 選擇是否儲存
    save = input("\n是否儲存結果影片? (y/n, 預設:y): ").strip().lower()
    save_output = save != 'n'
    
    # 選擇是否顯示即時畫面
    show = input("是否顯示即時處理畫面? (y/n, 預設:y): ").strip().lower()
    show_video = show != 'n'
    
    # 開始處理
    detector.process_video(video_path, save_output=save_output, show_video=show_video)


if __name__ == "__main__":
    main()