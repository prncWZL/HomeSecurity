import sys
import time
import cv2
import numpy as np
import os
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QStatusBar,
    QFileDialog, QMessageBox, QSlider
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from deepface import DeepFace
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device


class DetectionThread(QThread):
    """后台检测线程"""
    update_frame = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, bool, bool)
    status_update = pyqtSignal(str, str)
    video_info = pyqtSignal(str, int)
    video_progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.paused = False
        self.cap = None
        self.fire_model = None
        self.face_params = None
        self.imgsz = None
        self.fire_device = None
        self.fire_half = None
        self.fire_opt = None
        self.video_path = None
        self.total_frames = 0
        self.current_frame = 0


        self.playback_speed = 1.0  # 默认正常速度

        # 目标跟踪记录
        self.face_tracking = {}
        self.fire_smoke_tracking = {}

        # 报警状态
        self.fire_alarm_active = False
        self.intrusion_alarm_active = False
        self.alarm_enabled = True

        # 统计信息
        self.face_count = 0
        self.fire_count = 0
        self.smoke_count = 0
        self.unknown_count = 0
        self.alarm_history = []

        # 帧率计算
        self.frame_count = 0
        self.start_time = time.time()

    def set_playback_speed(self, speed):
        self.playback_speed = speed
        self.status_update.emit("info", f"播放速度设置为: {speed}x")

    def init_system(self):
        """初始化检测系统"""
        try:
            # 初始化视频源
            if self.video_path:
                # 使用视频文件
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    self.status_update.emit("error", f"无法打开视频文件: {self.video_path}")
                    return False

                # 获取视频信息
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_info.emit(os.path.basename(self.video_path), self.total_frames)
                self.status_update.emit("info", f"已加载视频: {os.path.basename(self.video_path)}")
            else:
                # 使用摄像头
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.status_update.emit("error", "无法打开摄像头")
                    return False
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.status_update.emit("info", "使用摄像头作为视频源")

            # 初始化火焰/烟雾检测模型
            try:
                self.fire_opt = {
                    'weights': 'best.pt',
                    'img_size': 640,
                    'conf_thres': 0.4,
                    'iou_thres': 0.5,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'agnostic_nms': False,
                    'augment': False
                }

                self.fire_device = select_device(self.fire_opt['device'])
                self.fire_half = self.fire_device.type != 'cpu'

                self.fire_model = attempt_load(self.fire_opt['weights'], map_location=self.fire_device)
                self.imgsz = check_img_size(self.fire_opt['img_size'], s=self.fire_model.stride.max())

                if self.fire_half:
                    self.fire_model.half()

                # 预热模型
                img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.fire_device)
                _ = self.fire_model(img.half() if self.fire_half else img) if self.fire_device.type != 'cpu' else None

                self.status_update.emit("info", f"火焰/烟雾模型加载成功 (设备: {self.fire_opt['device']})")
            except Exception as e:
                self.status_update.emit("error", f"火焰/烟雾模型加载失败: {str(e)}")
                return False

            # 初始化人脸检测参数
            self.face_params = {
                'detector': 'opencv',  # opencv
                'align': False,
                'db_path': '../../database',
                'model_name': 'VGG-Face',
                'distance_metric': 'cosine',
                'threshold': 0.6  # 0.6
            }

            # 检查人脸数据库
            if not os.path.exists(self.face_params['db_path']):
                self.status_update.emit("warning", f"人脸数据库路径不存在: {self.face_params['db_path']}")
            else:
                self.status_update.emit("info", f"使用人脸数据库: {self.face_params['db_path']}")

            return True
        except Exception as e:
            self.status_update.emit("error", f"初始化失败: {str(e)}")
            return False

    def load_video(self, video_path):
        """加载视频文件"""
        self.video_path = video_path
        if self.cap:
            self.cap.release()
        if not self.init_system():
            return False
        return True

    def set_video_position(self, position):
        """设置视频播放位置"""
        if self.cap and self.video_path:
            # 计算实际帧位置
            frame_pos = int((position / 100.0) * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_frame = frame_pos
            self.status_update.emit("info", f"跳转到帧: {frame_pos}/{self.total_frames}")

    def toggle_pause(self):
        """切换暂停状态"""
        self.paused = not self.paused
        status = "暂停" if self.paused else "继续"
        self.status_update.emit("info", f"视频已{status}")

    def process_fire_smoke(self, frame):
        """处理火焰/烟雾检测"""
        fire_detected = False
        smoke_detected = False
        fire_frame = frame.copy()
        current_time = time.time()

        if self.fire_model is None:
            return fire_frame, False

        try:
            # 预处理
            img = cv2.resize(fire_frame, (self.imgsz, self.imgsz))
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.fire_device)
            img = img.half() if self.fire_half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 推理
            pred = self.fire_model(img, augment=self.fire_opt['augment'])[0]

            # NMS
            pred = non_max_suppression(
                pred,
                self.fire_opt['conf_thres'],
                self.fire_opt['iou_thres'],
                classes=None,
                agnostic=self.fire_opt['agnostic_nms']
            )

            # 处理结果
            det = pred[0]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], fire_frame.shape).round()

                for *xyxy, conf, cls in det:
                    label = f'{self.fire_model.names[int(cls)]} {conf:.2f}'
                    color = (0, 0, 255) if self.fire_model.names[int(cls)] == 'fire' else (255, 0, 0)

                    # 绘制边界框
                    xyxy = [int(x) for x in xyxy]
                    cv2.rectangle(fire_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(fire_frame, label, (xyxy[0], xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # 更新检测状态
                    if self.fire_model.names[int(cls)] == 'fire':
                        fire_detected = True
                        self.fire_count += 1
                    elif self.fire_model.names[int(cls)] == 'smoke':
                        smoke_detected = True
                        self.smoke_count += 1

                    # 目标跟踪
                    box = tuple(xyxy)
                    if box not in self.fire_smoke_tracking:
                        self.fire_smoke_tracking[box] = {
                            'start_time': current_time,
                            'last_time': current_time,
                            'alarmed': False
                        }
                    else:
                        self.fire_smoke_tracking[box]['last_time'] = current_time

            # 清理旧目标
            to_remove = []
            for box, info in self.fire_smoke_tracking.items():
                if current_time - info['last_time'] > 10:
                    to_remove.append(box)
            for box in to_remove:
                del self.fire_smoke_tracking[box]

        except Exception as e:
            self.status_update.emit("error", f"火焰检测错误: {str(e)}")

        return fire_frame, fire_detected or smoke_detected

    def process_faces(self, frame):
        """处理人脸检测"""
        face_frame = frame.copy()
        unknown_face_detected = False
        current_time = time.time()

        try:
            rgb_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            faces = DeepFace.extract_faces(
                img_path=rgb_frame,
                detector_backend=self.face_params['detector'],
                enforce_detection=False,
                align=False
            )

            confidence_threshold = 0.8  # 0.8
            filtered_faces = [f for f in faces if f.get('confidence', 0) >= confidence_threshold]
            self.face_count += len(filtered_faces)

            for face in filtered_faces:
                area = face.get('facial_area', {})
                x, y, w, h = area.get('x', 0), area.get('y', 0), area.get('w', 0), area.get('h', 0)
                x, y = max(0, x), max(0, y)
                w, h = min(w, face_frame.shape[1] - x), min(h, face_frame.shape[0] - y)
                face_roi = face_frame[y:y + h, x:x + w]

                if face_roi.size == 0:
                    continue

                try:
                    face_img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    embedding = DeepFace.represent(
                        img_path=face_img,
                        model_name=self.face_params['model_name'],
                        enforce_detection=False
                    )[0]['embedding']

                    df = DeepFace.find(
                        img_path=face_img,
                        db_path=self.face_params['db_path'],
                        model_name=self.face_params['model_name'],
                        distance_metric=self.face_params['distance_metric'],
                        enforce_detection=False,
                        silent=True
                    )

                    recognized = False
                    if df is not None and len(df[0]) > 0:
                        best_match = df[0].iloc[0]
                        if best_match['distance'] <= self.face_params['threshold']:
                            identity = os.path.basename(os.path.dirname(best_match['identity']))
                            label = f"{identity} ({1 - best_match['distance']:.2f})"
                            recognized = True
                            cv2.rectangle(face_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(face_frame, label, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if not recognized:
                        self.unknown_count += 1
                        unknown_face_detected = True
                        cv2.rectangle(face_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(face_frame, "Unknown", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        # 目标跟踪
                        face_box = (x, y, x + w, y + h)
                        if face_box not in self.face_tracking:
                            self.face_tracking[face_box] = {
                                'start_time': current_time,
                                'last_time': current_time,
                                'alarmed': False
                            }
                        else:
                            self.face_tracking[face_box]['last_time'] = current_time

                except Exception as e:
                    self.status_update.emit("warning", f"人脸识别错误: {str(e)}")

            # 清理旧目标
            to_remove = []
            for box, info in self.face_tracking.items():
                if current_time - info['last_time'] > 10:
                    to_remove.append(box)
            for box in to_remove:
                del self.face_tracking[box]

        except Exception as e:
            self.status_update.emit("error", f"人脸检测错误: {str(e)}")

        return face_frame, unknown_face_detected

    def run(self):
        """主检测循环"""
        if not self.init_system():
            self.status_update.emit("error", "系统初始化失败，无法启动检测")
            return

        self.status_update.emit("info", "检测系统启动成功")
        self.start_time = time.time()
        self.frame_count = 0
        self.current_frame = 0

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                # 视频结束处理
                if self.video_path:
                    self.status_update.emit("info", "视频播放完成")
                    self.video_progress.emit(100)  # 发送完成信号
                    break
                else:
                    self.status_update.emit("error", "无法从摄像头读取帧")
                    time.sleep(1)
                    continue

            # 更新当前帧计数
            if self.video_path:
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                # 每5帧更新一次进度
                if self.current_frame % 5 == 0:
                    progress = int((self.current_frame / self.total_frames) * 100)
                    self.video_progress.emit(progress)

            # 处理检测
            fire_frame, fire_danger = self.process_fire_smoke(frame.copy())
            face_frame, unknown_face = self.process_faces(frame.copy())

            current_time = time.time()
            fire_alarm = False
            intrusion_alarm = False

            # 检查火焰/烟雾报警
            for box, info in self.fire_smoke_tracking.items():
                if not info['alarmed'] and current_time - info['start_time'] >= 3:
                    fire_alarm = True
                    info['alarmed'] = True
                    if self.alarm_enabled:
                        self.alarm_history.append(f"{time.strftime('%H:%M:%S')} - 检测到火焰/烟雾")

            # 检查陌生人报警
            for box, info in self.face_tracking.items():
                if not info['alarmed'] and current_time - info['start_time'] >= 5:
                    intrusion_alarm = True
                    info['alarmed'] = True
                    if self.alarm_enabled:
                        self.alarm_history.append(f"{time.strftime('%H:%M:%S')} - 检测到陌生人")

            # 更新帧率统计
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                fps = self.frame_count / (current_time - self.start_time)
                self.status_update.emit("stats", f"帧率: {fps:.1f} FPS")

            # 发送结果
            self.update_frame.emit(frame, face_frame, fire_frame, fire_alarm, intrusion_alarm)

            # 控制处理速度 - 根据播放速度调整
            if self.video_path:
                # 视频文件模式：根据播放速度调整
                sleep_time = 0.03 / self.playback_speed
                time.sleep(max(0.001, sleep_time))  # 确保不会睡眠负数时间
            else:
                # 摄像头模式：固定速度
                time.sleep(0.03)

    def stop(self):
        """停止检测线程"""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.quit()
        self.wait()

    def toggle_alarm(self):
        """切换报警状态"""
        self.alarm_enabled = not self.alarm_enabled
        status = "启用" if self.alarm_enabled else "禁用"
        self.status_update.emit("info", f"报警系统已{status}")


class SecuritySystemUI(QMainWindow):
    """安全监控系统主界面"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("安全监控系统")
        self.setWindowIcon(QIcon("security_icon.png"))
        self.setGeometry(100, 100, 1400, 850)  # 增加高度以适应新控件

        # 创建主部件和布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        # 创建左侧视频区域
        self.video_layout = self.create_video_layout()
        self.main_layout.addLayout(self.video_layout, 70)  # 70%宽度

        # 创建右侧信息面板
        self.info_panel = self.create_info_panel()
        self.main_layout.addWidget(self.info_panel, 30)  # 30%宽度

        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("系统准备就绪")
        self.status_bar.addWidget(self.status_label)

        # 创建检测线程
        self.detection_thread = DetectionThread()
        self.detection_thread.update_frame.connect(self.update_frames)
        self.detection_thread.status_update.connect(self.update_status)
        self.detection_thread.video_info.connect(self.update_video_info)  # 连接视频信息信号
        self.detection_thread.video_progress.connect(self.update_video_progress)  # 连接视频进度信号

        # 初始化UI状态
        self.fire_alarm_active = False
        self.intrusion_alarm_active = False
        self.system_running = False
        self.video_loaded = False

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
            }
            QLabel {
                color: #E0E0E0;
            }
            QGroupBox {
                color: #E0E0E0;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #3F3F46;
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #2D2D30;
            }
            QPushButton#alarmBtn {
                background-color: #D32F2F;
                font-weight: bold;
            }
            QPushButton#alarmBtn:checked {
                background-color: #388E3C;
            }
            QStatusBar {
                background-color: #252526;
                color: #E0E0E0;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #3F3F46;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #1976D2;
                border: 1px solid #0D47A1;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #1976D2;
                border-radius: 4px;
            }
        """)

    def create_video_layout(self):
        """创建视频显示区域布局"""
        layout = QVBoxLayout()

        # 标题
        title = QLabel("安全监控系统")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #E0E0E0; padding: 10px 0;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 状态指示器
        status_layout = QHBoxLayout()

        # 系统状态
        self.system_status = QLabel("系统已停止")
        self.system_status.setFont(QFont("Arial", 10))
        self.system_status.setStyleSheet("color: #FF5252;")
        status_layout.addWidget(self.system_status)

        # 视频源信息
        self.video_source_label = QLabel("视频源: 摄像头")
        self.video_source_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.video_source_label)

        # 报警状态
        alarm_layout = QHBoxLayout()

        self.fire_status = QLabel("火焰检测")
        self.fire_status.setStyleSheet("""
            background-color: #3F3F46; 
            border-radius: 10px; 
            padding: 8px;
            font-weight: bold;
        """)
        self.fire_status.setAlignment(Qt.AlignCenter)
        alarm_layout.addWidget(self.fire_status)

        self.intrusion_status = QLabel("入侵检测")
        self.intrusion_status.setStyleSheet("""
            background-color: #3F3F46; 
            border-radius: 10px; 
            padding: 8px;
            font-weight: bold;
        """)
        self.intrusion_status.setAlignment(Qt.AlignCenter)
        alarm_layout.addWidget(self.intrusion_status)

        status_layout.addLayout(alarm_layout)
        layout.addLayout(status_layout)

        # 视频进度条（新增）
        self.video_progress_layout = QVBoxLayout()
        self.video_progress_layout.setContentsMargins(5, 0, 5, 5)

        self.video_progress_label = QLabel("视频进度: 0%")
        self.video_progress_label.setFont(QFont("Arial", 9))
        self.video_progress_layout.addWidget(self.video_progress_label)

        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setRange(0, 100)
        self.video_slider.setValue(0)
        self.video_slider.setEnabled(False)
        self.video_slider.sliderReleased.connect(self.on_slider_released)
        self.video_progress_layout.addWidget(self.video_slider)

        layout.addLayout(self.video_progress_layout)

        # 视频显示区域
        video_grid = QGridLayout()

        # 原始视频
        original_group = QGroupBox("原始视频")
        original_layout = QVBoxLayout(original_group)
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.original_label.setStyleSheet("background-color: black;")
        original_layout.addWidget(self.original_label)
        video_grid.addWidget(original_group, 0, 0)

        # 人脸检测
        face_group = QGroupBox("人脸检测")
        face_layout = QVBoxLayout(face_group)
        self.face_label = QLabel()
        self.face_label.setAlignment(Qt.AlignCenter)
        self.face_label.setMinimumSize(400, 300)
        self.face_label.setStyleSheet("background-color: black;")
        face_layout.addWidget(self.face_label)
        video_grid.addWidget(face_group, 1, 0)

        # 火焰检测
        fire_group = QGroupBox("火焰/烟雾检测")
        fire_layout = QVBoxLayout(fire_group)
        self.fire_label = QLabel()
        self.fire_label.setAlignment(Qt.AlignCenter)
        self.fire_label.setMinimumSize(400, 300)
        self.fire_label.setStyleSheet("background-color: black;")
        fire_layout.addWidget(self.fire_label)
        video_grid.addWidget(fire_group, 1, 1)

        # 报警区域
        alarm_group = QGroupBox("报警信息")
        alarm_layout = QVBoxLayout(alarm_group)
        self.alarm_label = QLabel("无报警信息")
        self.alarm_label.setAlignment(Qt.AlignCenter)
        self.alarm_label.setMinimumSize(400, 300)
        self.alarm_label.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            color: #4CAF50; 
            background-color: #1E1E1E;
        """)
        alarm_layout.addWidget(self.alarm_label)
        video_grid.addWidget(alarm_group, 0, 1)

        # 设置行列比例
        video_grid.setRowStretch(0, 1)
        video_grid.setRowStretch(1, 1)
        video_grid.setColumnStretch(0, 1)
        video_grid.setColumnStretch(1, 1)

        layout.addLayout(video_grid, 80)  # 调整高度比例

        # 控制按钮
        control_layout = QHBoxLayout()

        # 播放速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("播放速度:"))

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 4)  # 1-4倍速
        self.speed_slider.setValue(1)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self.change_playback_speed)
        speed_layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_label)
        control_layout.addLayout(speed_layout)

        self.start_btn = QPushButton("启动系统")
        self.start_btn.setStyleSheet("background-color: #388E3C; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止系统")
        self.stop_btn.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold;")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        self.alarm_btn = QPushButton("禁用报警")
        self.alarm_btn.setObjectName("alarmBtn")
        self.alarm_btn.clicked.connect(self.toggle_alarm)
        self.alarm_btn.setCheckable(True)
        control_layout.addWidget(self.alarm_btn)

        # 新增：视频控制按钮
        self.video_btn = QPushButton("打开视频")
        self.video_btn.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold;")
        self.video_btn.clicked.connect(self.open_video)
        control_layout.addWidget(self.video_btn)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)

        layout.addLayout(control_layout, 10)  # 10%高度

        return layout

    def create_info_panel(self):
        """创建右侧信息面板"""
        panel = QGroupBox("系统信息")
        layout = QVBoxLayout(panel)

        # 统计信息
        stats_group = QGroupBox("检测统计")
        stats_layout = QVBoxLayout(stats_group)

        self.face_stats = QLabel("检测到人脸: 0")
        self.face_stats.setFont(QFont("Arial", 10))
        stats_layout.addWidget(self.face_stats)

        self.fire_stats = QLabel("检测到火焰: 0")
        self.fire_stats.setFont(QFont("Arial", 10))
        stats_layout.addWidget(self.fire_stats)

        self.smoke_stats = QLabel("检测到烟雾: 0")
        self.smoke_stats.setFont(QFont("Arial", 10))
        stats_layout.addWidget(self.smoke_stats)

        self.unknown_stats = QLabel("未知人员: 0")
        self.unknown_stats.setFont(QFont("Arial", 10))
        stats_layout.addWidget(self.unknown_stats)

        layout.addWidget(stats_group)

        # 系统信息
        system_group = QGroupBox("系统状态")
        system_layout = QVBoxLayout(system_group)

        self.camera_status = QLabel("摄像头: 未连接")
        self.camera_status.setFont(QFont("Arial", 10))
        system_layout.addWidget(self.camera_status)

        self.detection_status = QLabel("火焰检测模型: 未加载")
        self.detection_status.setFont(QFont("Arial", 10))
        system_layout.addWidget(self.detection_status)

        self.recognition_status = QLabel("人脸识别模型: 未加载")
        self.recognition_status.setFont(QFont("Arial", 10))
        system_layout.addWidget(self.recognition_status)

        self.system_uptime = QLabel("运行时间: 00:00:00")
        self.system_uptime.setFont(QFont("Arial", 10))
        system_layout.addWidget(self.system_uptime)

        self.fps_counter = QLabel("帧率: 0 FPS")
        self.fps_counter.setFont(QFont("Arial", 10))
        system_layout.addWidget(self.fps_counter)

        # 新增：视频信息
        self.video_info_label = QLabel("视频: 无")
        self.video_info_label.setFont(QFont("Arial", 10))
        system_layout.addWidget(self.video_info_label)

        layout.addWidget(system_group)

        # 报警记录
        alarm_group = QGroupBox("报警记录")
        alarm_layout = QVBoxLayout(alarm_group)

        self.alarm_list = QLabel("无报警记录")
        self.alarm_list.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.alarm_list.setStyleSheet("background-color: #1E1E1E; color: #E0E0E0; padding: 10px; border-radius: 5px;")
        self.alarm_list.setFont(QFont("Consolas", 9))
        self.alarm_list.setMinimumHeight(250)
        alarm_layout.addWidget(self.alarm_list)

        # 报警记录控制按钮
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("清空记录")
        self.clear_btn.setStyleSheet("font-size: 10px; padding: 3px;")
        self.clear_btn.clicked.connect(self.clear_alarms)
        btn_layout.addWidget(self.clear_btn)

        self.save_btn = QPushButton("保存记录")
        self.save_btn.setStyleSheet("font-size: 10px; padding: 3px;")
        self.save_btn.clicked.connect(self.save_alarms)
        btn_layout.addWidget(self.save_btn)
        alarm_layout.addLayout(btn_layout)

        layout.addWidget(alarm_group)

        # 添加伸缩空间使布局填满
        layout.addStretch(1)

        return panel

    def open_video(self):
        """打开视频文件"""
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "打开视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
            options=options
        )

        if fileName:
            # 停止当前检测（如果有）
            if self.system_running:
                self.stop_detection()

            # 加载视频文件
            if self.detection_thread.load_video(fileName):
                self.video_loaded = True
                self.status_label.setText(f"已加载视频: {os.path.basename(fileName)}")
                self.video_source_label.setText(f"视频源: {os.path.basename(fileName)}")
                self.pause_btn.setEnabled(True)
                self.video_slider.setEnabled(True)
            else:
                QMessageBox.critical(self, "错误", "无法加载视频文件")

    def toggle_pause(self):
        """切换暂停状态"""
        self.detection_thread.toggle_pause()
        if self.detection_thread.paused:
            self.pause_btn.setText("继续")
        else:
            self.pause_btn.setText("暂停")

    def update_video_info(self, filename, total_frames):
        """更新视频信息"""
        self.video_info_label.setText(f"视频: {filename}, 总帧数: {total_frames}")
        self.video_slider.setRange(0, 100)
        self.video_slider.setValue(0)

    def update_video_progress(self, progress):
        """更新视频进度"""
        self.video_progress_label.setText(f"视频进度: {progress}%")
        self.video_slider.setValue(progress)

    def on_slider_released(self):
        """当用户释放进度条滑块时"""
        if self.video_loaded:
            position = self.video_slider.value()
            self.detection_thread.set_video_position(position)

    def change_playback_speed(self, value):
        """改变播放速度"""
        # 将滑块值转换为速度 (1=0.5x, 2=1.0x, 3=1.5x, 4=2.0x)
        speed_map = {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0}
        speed = speed_map.get(value, 1.0)
        self.speed_label.setText(f"{speed}x")
        self.detection_thread.set_playback_speed(speed)

    def update_frames(self, original, face, fire, fire_alarm, intrusion_alarm):
        """更新视频帧显示"""
        # 更新原始视频
        self.display_frame(self.original_label, original)

        # 更新人脸检测结果
        self.display_frame(self.face_label, face)

        # 更新火焰检测结果
        self.display_frame(self.fire_label, fire)

        # 更新报警状态
        self.update_fire_alarm(fire_alarm)
        self.update_intrusion_alarm(intrusion_alarm)

        # 更新报警信息显示
        alarm_text = ""
        if fire_alarm:
            alarm_text += "🔥 火焰/烟雾报警！\n"
            self.alarm_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #FF5252; background-color: #330000;")
        if intrusion_alarm:
            alarm_text += "🚨 入侵报警！\n"
            self.alarm_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #FF5252; background-color: #330000;")

        if not alarm_text:
            alarm_text = "系统运行正常"
            self.alarm_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #4CAF50; background-color: #003300;")

        self.alarm_label.setText(alarm_text)

    def display_frame(self, label, frame):
        """在QLabel中显示OpenCV图像"""
        # 将OpenCV图像转换为Qt图像
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # 缩放图像以适应标签
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 设置图像
        label.setPixmap(pixmap)

    def update_fire_alarm(self, active):
        """更新火焰报警状态"""
        self.fire_alarm_active = active
        if active:
            self.fire_status.setText("火焰报警!")
            self.fire_status.setStyleSheet(
                "background-color: #D32F2F; color: white; border-radius: 10px; padding: 8px; font-weight: bold;")
        else:
            self.fire_status.setText("火焰检测")
            self.fire_status.setStyleSheet(
                "background-color: #3F3F46; border-radius: 10px; padding: 8px; font-weight: bold;")

    def update_intrusion_alarm(self, active):
        """更新入侵报警状态"""
        self.intrusion_alarm_active = active
        if active:
            self.intrusion_status.setText("入侵报警!")
            self.intrusion_status.setStyleSheet(
                "background-color: #D32F2F; color: white; border-radius: 10px; padding: 8px; font-weight: bold;")
        else:
            self.intrusion_status.setText("入侵检测")
            self.intrusion_status.setStyleSheet(
                "background-color: #3F3F46; border-radius: 10px; padding: 8px; font-weight: bold;")

    def update_status(self, status_type, message):
        """更新系统状态信息"""
        if status_type == "info":
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: #4CAF50;")
        elif status_type == "warning":
            self.status_label.setText("警告: " + message)
            self.status_label.setStyleSheet("color: #FFC107;")
        elif status_type == "error":
            self.status_label.setText("错误: " + message)
            self.status_label.setStyleSheet("color: #D32F2F;")
        elif status_type == "stats":
            self.fps_counter.setText(message)

        # 更新特定状态信息
        if "摄像头" in message:
            self.camera_status.setText(message)
        elif "火焰" in message and "模型" in message:
            self.detection_status.setText(message)
        elif "人脸数据库" in message:
            self.recognition_status.setText(message)

    def start_detection(self):
        """开始检测"""
        if not self.system_running:
            self.detection_thread.start()
            self.system_running = True
            self.system_status.setText("系统运行中")
            self.system_status.setStyleSheet("color: #4CAF50;")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.start_time = time.time()
            self.timer_id = self.startTimer(1000)  # 每秒更新一次运行时间

    def stop_detection(self):
        """停止检测"""
        if self.system_running:
            self.detection_thread.running = False
            self.system_running = False
            self.system_status.setText("系统已停止")
            self.system_status.setStyleSheet("color: #D32F2F;")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            if hasattr(self, 'timer_id'):
                self.killTimer(self.timer_id)

    def timerEvent(self, event):
        """定时器事件，更新运行时间"""
        if self.system_running:
            elapsed = time.time() - self.start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.system_uptime.setText(f"运行时间: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def toggle_alarm(self):
        """切换报警开关"""
        self.detection_thread.toggle_alarm()
        if self.detection_thread.alarm_enabled:
            self.alarm_btn.setText("禁用报警")
            self.alarm_btn.setStyleSheet("background-color: #388E3C; color: white; font-weight: bold;")
        else:
            self.alarm_btn.setText("启用报警")
            self.alarm_btn.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold;")

    def clear_alarms(self):
        """清空报警记录"""
        self.detection_thread.alarm_history = []
        self.alarm_list.setText("报警记录已清空")

    def save_alarms(self):
        """保存报警记录"""
        if self.detection_thread.alarm_history:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"alarms_{timestamp}.txt"
            with open(filename, "w") as f:
                for alarm in self.detection_thread.alarm_history:
                    f.write(alarm + "\n")
            self.status_label.setText(f"报警记录已保存到 {filename}")
        else:
            self.status_label.setText("没有报警记录可保存")

    def update_statistics(self):
        """更新统计信息"""
        if not hasattr(self.detection_thread, 'face_count'):
            return

        self.face_stats.setText(f"检测到人脸: {self.detection_thread.face_count}")
        self.fire_stats.setText(f"检测到火焰: {self.detection_thread.fire_count}")
        self.smoke_stats.setText(f"检测到烟雾: {self.detection_thread.smoke_count}")
        self.unknown_stats.setText(f"未知人员: {self.detection_thread.unknown_count}")

        # 更新报警记录
        if hasattr(self.detection_thread, 'alarm_history') and self.detection_thread.alarm_history:
            alarm_text = "\n".join(self.detection_thread.alarm_history[-10:])  # 显示最近10条
            self.alarm_list.setText(alarm_text)
        else:
            self.alarm_list.setText("无报警记录")

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        self.stop_detection()
        if self.detection_thread.isRunning():
            self.detection_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SecuritySystemUI()
    window.show()

    # 设置定时器更新统计信息
    timer = QTimer()
    timer.timeout.connect(window.update_statistics)
    timer.start(1000)  # 每秒更新一次

    sys.exit(app.exec_())