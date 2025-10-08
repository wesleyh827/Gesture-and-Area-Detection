import cv2
from ultralytics import YOLO
import time
import numpy as np
import os

class GestureDetector:
    def __init__(self, model_size='n', alert_duration=3.0):
        """
        初始化手勢偵測器
        
        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
            alert_duration: 警示動作持續時間（秒）
        """
        print(f"載入 YOLOv8-{model_size}-pose 模型...")
        self.model = YOLO(f'yolov8{model_size}-pose.pt')
        self.alert_threshold = alert_duration
        self.alert_start_time = None
        self.is_alerting = False
        self.gui_available = self._check_gui_support()
        
    def _check_gui_support(self):
        """檢查是否支援 GUI 顯示"""
        try:
            # 嘗試創建一個測試視窗
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('test', test_img)
            cv2.destroyAllWindows()
            return True
        except:
            print("⚠ 偵測到系統不支援 GUI 顯示，將只儲存結果圖片")
            return False
        
    def check_hands_above_shoulders(self, keypoints):
        """
        檢查雙手是否都在肩膀以上（或接近）
        
        Args:
            keypoints: [17, 3] 陣列，包含 (x, y, confidence)
        
        Returns:
            bool: 雙手是否在肩膀以上
        """
        # COCO keypoints 索引
        # 5: 左肩, 6: 右肩, 7: 左手肘, 8: 右手肘, 9: 左手腕, 10: 右手腕
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        
        # 檢查confidence（信心度）- 降低閾值讓偵測更容易
        confidence_threshold = 0.4
        
        # 計算平均肩膀高度（y座標越小越上方）
        if left_shoulder[2] < confidence_threshold or right_shoulder[2] < confidence_threshold:
            return False
        
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # 容差值（像素）- 手腕接近肩膀也算舉手
        # 調整這個值來改變靈敏度：數字越大越容易觸發
        tolerance = 60
        
        # 左手判定：優先用手腕，如果手腕信心度不夠則用手肘
        left_hand_up = False
        if left_wrist[2] >= confidence_threshold:
            left_hand_up = left_wrist[1] < (shoulder_y + tolerance)
        elif left_elbow[2] >= confidence_threshold:
            # 如果手腕偵測不到，手肘在肩膀附近也算
            left_hand_up = left_elbow[1] < (shoulder_y + tolerance * 0.5)
        
        # 右手判定：同上
        right_hand_up = False
        if right_wrist[2] >= confidence_threshold:
            right_hand_up = right_wrist[1] < (shoulder_y + tolerance)
        elif right_elbow[2] >= confidence_threshold:
            right_hand_up = right_elbow[1] < (shoulder_y + tolerance * 0.5)
        
        return left_hand_up and right_hand_up
    
    def draw_info(self, frame, hands_up, duration=0, is_static=False):
        """在畫面上繪製資訊"""
        # 繪製狀態資訊
        if hands_up:
            if is_static or duration >= self.alert_threshold:
                # 警示觸發（圖片模式或達到時間）
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
                cv2.putText(frame, "WARNING: ALERT POSE DETECTED!", (50, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
            else:
                # 偵測到舉手但未達3秒
                cv2.putText(frame, f"Hands Up: {duration:.1f}s / {self.alert_threshold}s", 
                          (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # 未偵測到警示姿勢
            if is_static:
                cv2.putText(frame, "No Alert Pose Detected", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def process_frame(self, frame, is_static=False):
        """
        處理單一影格
        
        Args:
            frame: 輸入影格
            is_static: 是否為靜態圖片（不需計時）
        
        Returns:
            annotated_frame: 標註後的影格
            hands_up: 是否偵測到警示姿勢
        """
        # 執行姿態估計
        results = self.model(frame, verbose=False)
        
        hands_up = False
        duration = 0
        
        # 檢查是否有偵測到人
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # 只處理第一個人（可擴展為多人）
            person_keypoints = results[0].keypoints.data[0]
            
            # 檢查是否雙手舉起
            if self.check_hands_above_shoulders(person_keypoints):
                hands_up = True
                
                if not is_static:
                    if self.alert_start_time is None:
                        self.alert_start_time = time.time()
                    
                    duration = time.time() - self.alert_start_time
                    
                    # 檢查是否達到警示時間
                    if duration >= self.alert_threshold and not self.is_alerting:
                        self.is_alerting = True
                        print(f"\n🚨 警示觸發！持續時間: {duration:.2f}秒")
            else:
                # 重置計時器
                if not is_static and self.alert_start_time is not None:
                    print(f"重置計時器（持續了 {time.time() - self.alert_start_time:.2f}秒）")
                self.alert_start_time = None
                self.is_alerting = False
        
        # 繪製姿態骨架
        annotated_frame = results[0].plot()
        
        # 繪製資訊
        annotated_frame = self.draw_info(annotated_frame, hands_up, duration, is_static)
        
        return annotated_frame, hands_up
    
    def process_image(self, image_path, save_output=True, output_dir='output'):
        """
        處理單張圖片
        
        Args:
            image_path: 圖片路徑
            save_output: 是否儲存結果
            output_dir: 輸出資料夾
        
        Returns:
            bool: 是否偵測到警示姿勢
        """
        print(f"\n處理圖片: {image_path}")
        
        # 讀取圖片
        if not os.path.exists(image_path):
            print(f"❌ 找不到圖片: {image_path}")
            return False
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ 無法讀取圖片: {image_path}")
            return False
        
        # 處理圖片（靜態模式）
        annotated_frame, hands_up = self.process_frame(frame, is_static=True)
        
        # 顯示結果
        print(f"{'✅ 偵測到警示姿勢！' if hands_up else '❌ 未偵測到警示姿勢'}")
        
        # 儲存結果
        output_path = None
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_result{ext}")
            cv2.imwrite(output_path, annotated_frame)
            print(f"💾 結果已儲存至: {output_path}")
        
        # 只在支援 GUI 時顯示圖片
        if self.gui_available:
            try:
                cv2.imshow('Gesture Detection - Image', annotated_frame)
                print("按任意鍵繼續...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"⚠ 無法顯示圖片: {e}")
        else:
            print(f"💡 請開啟 {output_path} 查看結果")
        
        return hands_up
    
    def process_image_batch(self, image_folder, save_output=True, output_dir='output'):
        """
        批次處理多張圖片
        
        Args:
            image_folder: 圖片資料夾路徑
            save_output: 是否儲存結果
            output_dir: 輸出資料夾
        """
        print(f"\n批次處理資料夾: {image_folder}")
        
        if not os.path.exists(image_folder):
            print(f"❌ 找不到資料夾: {image_folder}")
            return
        
        # 支援的圖片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        # 取得所有圖片
        image_files = []
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, file))
        
        if not image_files:
            print("❌ 資料夾中沒有找到圖片檔案")
            return
        
        print(f"找到 {len(image_files)} 張圖片\n")
        
        # 統計結果
        alert_count = 0
        
        # 處理每張圖片
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] ", end="")
            
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"⚠ 跳過無法讀取的圖片: {image_path}")
                continue
            
            # 處理圖片
            annotated_frame, hands_up = self.process_frame(frame, is_static=True)
            
            if hands_up:
                alert_count += 1
                print(f"✅ {os.path.basename(image_path)} - 偵測到警示姿勢")
            else:
                print(f"❌ {os.path.basename(image_path)} - 未偵測到警示姿勢")
            
            # 儲存結果
            if save_output:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_result{ext}")
                cv2.imwrite(output_path, annotated_frame)
        
        # 顯示統計
        print(f"\n{'='*50}")
        print(f"處理完成！")
        print(f"總圖片數: {len(image_files)}")
        print(f"偵測到警示姿勢: {alert_count} 張")
        print(f"未偵測到: {len(image_files) - alert_count} 張")
        if save_output:
            print(f"💾 結果已儲存至: {output_dir}/")
        print(f"{'='*50}")
    
    def run_webcam(self, camera_id=0):
        """使用攝影機即時偵測"""
        if not self.gui_available:
            print("❌ 系統不支援 GUI 顯示，無法使用攝影機模式")
            print("💡 請使用圖片處理模式")
            return
            
        print(f"\n開始使用攝影機 {camera_id}")
        print("請雙手舉過肩膀3秒以上來測試")
        print("按 'q' 離開\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ 無法開啟攝影機 {camera_id}")
            return
        
        # 設定解析度（可選）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取影像")
                break
            
            # 處理影格
            processed_frame, _ = self.process_frame(frame, is_static=False)
            
            # 顯示結果
            cv2.imshow('Gesture Detection - Webcam', processed_frame)
            
            # 按 'q' 離開
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("程式結束")
    
    def run_video(self, video_path, save_output=True, output_dir='output'):
        """
        處理影片檔案
        
        Args:
            video_path: 影片路徑
            save_output: 是否儲存結果影片
            output_dir: 輸出資料夾
        """
        print(f"\n處理影片: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 無法開啟影片 {video_path}")
            return
        
        # 取得影片資訊
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"影片資訊: {width}x{height} @ {fps}fps, 總影格數: {total_frames}")
        
        # 準備輸出影片
        out = None
        output_path = None
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(video_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_result{ext}")
            
            # 使用 mp4v 編碼器（相容性較好）
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 將儲存結果至: {output_path}")
        
        # 記錄警示事件
        alert_events = []
        frame_count = 0
        
        print("\n開始處理影片...")
        if self.gui_available:
            print("按 'q' 提前結束，按 'p' 暫停")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 處理影格
            processed_frame, hands_up = self.process_frame(frame, is_static=False)
            
            # 記錄警示事件
            if self.is_alerting and (not alert_events or not alert_events[-1]['end']):
                if not alert_events or alert_events[-1]['end']:
                    # 新的警示事件
                    alert_events.append({
                        'start_frame': frame_count,
                        'start_time': frame_count / fps,
                        'end': False
                    })
            elif not self.is_alerting and alert_events and not alert_events[-1]['end']:
                # 警示結束
                alert_events[-1]['end_frame'] = frame_count
                alert_events[-1]['end_time'] = frame_count / fps
                alert_events[-1]['end'] = True
            
            # 儲存影格
            if out is not None:
                out.write(processed_frame)
            
            # 顯示進度
            if frame_count % 30 == 0:  # 每30幀顯示一次
                progress = (frame_count / total_frames) * 100
                print(f"\r處理中... {progress:.1f}% ({frame_count}/{total_frames})", end='')
            
            # 顯示結果（如果支援GUI）
            if self.gui_available:
                cv2.imshow('Gesture Detection - Video', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n\n⚠ 使用者中斷處理")
                    break
                elif key == ord('p'):
                    print("\n暫停中，按任意鍵繼續...")
                    cv2.waitKey(0)
        
        print("\n\n影片處理完成！")
        
        # 釋放資源
        cap.release()
        if out is not None:
            out.release()
        if self.gui_available:
            cv2.destroyAllWindows()
        
        # 顯示結果統計
        print(f"\n{'='*50}")
        print(f"處理統計:")
        print(f"總影格數: {frame_count}")
        print(f"影片長度: {frame_count/fps:.2f} 秒")
        print(f"偵測到警示事件: {len(alert_events)} 次")
        
        if alert_events:
            print(f"\n警示事件詳情:")
            for i, event in enumerate(alert_events, 1):
                start_time = event['start_time']
                end_time = event.get('end_time', frame_count / fps)
                duration = end_time - start_time
                print(f"  事件 {i}: {start_time:.2f}s - {end_time:.2f}s (持續 {duration:.2f}s)")
        
        if save_output and output_path:
            print(f"\n💾 結果影片已儲存至: {output_path}")
        
        print(f"{'='*50}\n")


def main():
    """主程式"""
    print("=" * 50)
    print("手勢警示偵測系統")
    print("=" * 50)
    
    # 建立偵測器
    detector = GestureDetector(
        model_size='n',  # 使用 nano 模型（最快）
        alert_duration=3.0  # 3秒警示
    )
    
    # 選擇模式
    print("\n請選擇模式:")
    print("1. 使用攝影機即時偵測")
    print("2. 處理影片檔案")
    print("3. 處理單張圖片")
    print("4. 批次處理多張圖片")
    
    choice = input("\n請輸入選項 (1-4): ").strip()
    
    if choice == '1':
        # 使用攝影機
        detector.run_webcam(camera_id=0)
    elif choice == '2':
        # 使用影片檔案
        video_path = input("請輸入影片路徑: ").strip()
        save = input("是否儲存結果影片? (y/n, 預設:y): ").strip().lower()
        save_output = save != 'n'
        detector.run_video(video_path, save_output=save_output)
    elif choice == '3':
        # 處理單張圖片
        image_path = input("請輸入圖片路徑: ").strip()
        detector.process_image(image_path, save_output=True)
    elif choice == '4':
        # 批次處理圖片
        folder_path = input("請輸入圖片資料夾路徑: ").strip()
        detector.process_image_batch(folder_path, save_output=True)
    else:
        print("❌ 無效的選項")


if __name__ == "__main__":
    main()