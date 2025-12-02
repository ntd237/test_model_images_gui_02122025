"""
Main Window cho YOLO Model Testing Tool
UI chính với layout Master-Detail
"""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QTextEdit, QGroupBox, QFileDialog, QMessageBox,
    QHeaderView, QLabel
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import cv2
import numpy as np

from src.gui.widgets import ImageLabel, InfoPanel, SliderWithLabel, StatusLabel
from src.core.model_loader import ModelLoader
from src.core.inference import InferenceEngine
from src.utils.image_utils import (
    load_image_cv, cv_to_qpixmap, validate_image_format, get_image_info
)


class InferenceThread(QThread):
    """
    Thread riêng để chạy inference không block UI
    (Separate thread to run inference without blocking UI)
    """
    
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(
        self,
        model,
        image,
        conf_threshold,
        iou_threshold,
        device='cpu'
    ):
        super().__init__()
        self.model = model
        self.image = image
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.engine = InferenceEngine()
    
    def run(self):
        """Run inference"""
        try:
            result = self.engine.run_inference(
                self.model,
                self.image,
                self.conf_threshold,
                self.iou_threshold,
                self.device
            )
            
            if result is not None:
                self.finished.emit(result)
            else:
                self.error.emit("Inference thất bại")
                
        except Exception as e:
            self.error.emit(f"Lỗi inference: {str(e)}")


class MainWindow(QMainWindow):
    """Main Window của application"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Model Testing Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.model_loader = ModelLoader()
        self.current_image = None
        self.current_image_path = None
        self.inference_thread = None
        
        # Setup UI
        self.setup_ui()
        
        print("✅ Application initialized successfully")
    
    def setup_ui(self):
        """Setup main UI layout"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # === LEFT PANEL: Control Panel ===
        left_panel = self.create_control_panel()
        left_panel.setMaximumWidth(320)
        main_layout.addWidget(left_panel)
        
        # === RIGHT PANEL: Main Content ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(16)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image display area
        image_splitter = self.create_image_display()
        right_layout.addWidget(image_splitter, stretch=3)
        
        # Detection results area
        results_panel = self.create_results_panel()
        right_layout.addWidget(results_panel, stretch=1)
        
        main_layout.addWidget(right_panel, stretch=1)
        
        # Status bar
        self.statusBar().showMessage("Sẵn sàng | Ready")
    
    def create_control_panel(self) -> QWidget:
        """
        Tạo control panel bên trái
        (Create left control panel)
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        
        # === TITLE ===
        title = QLabel("YOLO Model Tester")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # === IMAGE SECTION ===
        image_group = QGroupBox("Ảnh Input")
        image_layout = QVBoxLayout(image_group)
        
        self.btn_load_image = QPushButton("📁 Load Ảnh")
        self.btn_load_image.clicked.connect(self.load_image)
        image_layout.addWidget(self.btn_load_image)
        
        self.image_info_panel = InfoPanel("Thông tin ảnh")
        image_layout.addWidget(self.image_info_panel)
        
        layout.addWidget(image_group)
        
        # === MODEL SECTION ===
        model_group = QGroupBox("Model YOLO")
        model_layout = QVBoxLayout(model_group)
        
        self.btn_load_model = QPushButton("🧠 Load Model")
        self.btn_load_model.clicked.connect(self.load_model)
        model_layout.addWidget(self.btn_load_model)
        
        self.model_info_panel = InfoPanel("Thông tin model")
        model_layout.addWidget(self.model_info_panel)
        
        layout.addWidget(model_group)
        
        # === SETTINGS SECTION ===
        settings_group = QGroupBox("Cài Đặt Inference")
        settings_layout = QVBoxLayout(settings_group)
        
        # Confidence threshold slider
        self.conf_slider = SliderWithLabel(
            "Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            default_value=0.25,
            decimals=2
        )
        settings_layout.addWidget(self.conf_slider)
        
        # IOU threshold slider
        self.iou_slider = SliderWithLabel(
            "IOU Threshold:",
            min_value=0.0,
            max_value=1.0,
            default_value=0.45,
            decimals=2
        )
        settings_layout.addWidget(self.iou_slider)
        
        layout.addWidget(settings_group)
        
        # === INFERENCE BUTTON ===
        self.btn_run_inference = QPushButton("▶ Chạy Inference")
        self.btn_run_inference.setObjectName("successButton")
        self.btn_run_inference.setEnabled(False)
        self.btn_run_inference.clicked.connect(self.run_inference)
        self.btn_run_inference.setMinimumHeight(50)
        font = self.btn_run_inference.font()
        font.setPointSize(14)
        font.setBold(True)
        self.btn_run_inference.setFont(font)
        layout.addWidget(self.btn_run_inference)
        
        # === STATUS ===
        self.status_label = StatusLabel()
        self.status_label.set_status("Load ảnh và model để bắt đầu", "info")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # === SAVE BUTTON ===
        self.btn_save_result = QPushButton("💾 Lưu Kết Quả")
        self.btn_save_result.setEnabled(False)
        self.btn_save_result.clicked.connect(self.save_result)
        layout.addWidget(self.btn_save_result)
        
        return panel
    
    def create_image_display(self) -> QSplitter:
        """
        Tạo image display area (split view)
        (Create image display area - split view)
        """
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Original image
        left_container = QGroupBox("Ảnh Gốc")
        left_layout = QVBoxLayout(left_container)
        self.original_image_label = ImageLabel()
        left_layout.addWidget(self.original_image_label)
        splitter.addWidget(left_container)
        
        # Right: Result image
        right_container = QGroupBox("Kết Quả Inference")
        right_layout = QVBoxLayout(right_container)
        self.result_image_label = ImageLabel()
        right_layout.addWidget(self.result_image_label)
        splitter.addWidget(right_container)
        
        # Equal size
        splitter.setSizes([500, 500])
        
        return splitter
    
    def create_results_panel(self) -> QWidget:
        """
        Tạo results panel (table + logs)
        (Create results panel - table + logs)
        """
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setSpacing(16)
        
        # === DETECTION TABLE ===
        table_container = QGroupBox("Detections")
        table_layout = QVBoxLayout(table_container)
        
        self.detections_table = QTableWidget()
        self.detections_table.setColumnCount(4)
        self.detections_table.setHorizontalHeaderLabels([
            "Class", "Confidence", "BBox (x1,y1)", "BBox (x2,y2)"
        ])
        
        # Table styling
        header = self.detections_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        self.detections_table.setAlternatingRowColors(True)
        self.detections_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        table_layout.addWidget(self.detections_table)
        layout.addWidget(table_container, stretch=2)
        
        # === LOG PANEL ===
        log_container = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_container)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_container, stretch=1)
        
        return panel
    
    # === EVENT HANDLERS ===
    
    def load_image(self):
        """Load ảnh từ file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh để test",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Validate format
        if not validate_image_format(file_path):
            QMessageBox.warning(
                self,
                "Format không hỗ trợ",
                "Định dạng ảnh không được hỗ trợ!"
            )
            return
        
        # Load image
        self.current_image = load_image_cv(file_path)
        
        if self.current_image is None:
            QMessageBox.critical(
                self,
                "Lỗi",
                "Không thể load ảnh!"
            )
            return
        
        self.current_image_path = file_path
        
        # Display image
        pixmap = cv_to_qpixmap(self.current_image)
        self.original_image_label.setPixmap(pixmap)
        
        # Update info panel
        info = get_image_info(file_path)
        if info:
            info_text = (
                f"📏 Kích thước: {info['width']}x{info['height']}\n"
                f"📊 Channels: {info['channels']}\n"
                f"💾 Dung lượng: {info['size_kb']:.1f} KB\n"
                f"📁 Format: {info['format']}"
            )
            self.image_info_panel.set_info(info_text)
        
        # Log
        self.log(f"✅ Đã load ảnh: {os.path.basename(file_path)}")
        
        # Update status
        self._update_run_button_state()
        
        # Clear previous results
        self.result_image_label.clear_image()
        self.detections_table.setRowCount(0)
    
    def load_model(self):
        """Load YOLO model từ file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn YOLO model",
            "resources/models",
            "PyTorch Model (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Load model
        self.log("Đang load model...")
        success = self.model_loader.load_model(file_path)
        
        if not success:
            QMessageBox.critical(
                self,
                "Lỗi",
                "Không thể load model!"
            )
            return
        
        # Update info panel
        info = self.model_loader.get_model_info()
        if info:
            info_text = (
                f"📦 Model: {info['name']}\n"
                f"🏷️ Classes: {info['num_classes']}\n"
                f"💾 Dung lượng: {info['size_mb']} MB\n"
                f"📂 Type: {info['type']}"
            )
            self.model_info_panel.set_info(info_text)
        
        # Log
        self.log(f"✅ Đã load model: {os.path.basename(file_path)}")
        
        # Update status
        self._update_run_button_state()
    
    def run_inference(self):
        """Chạy inference với model và ảnh hiện tại"""
        if self.current_image is None or not self.model_loader.is_model_loaded():
            return
        
        # Disable button
        self.btn_run_inference.setEnabled(False)
        self.btn_run_inference.setText("⏳ Đang xử lý...")
        
        # Update status
        self.status_label.set_status("Đang chạy inference...", "info")
        self.log("🚀 Bắt đầu inference...")
        
        # Get thresholds
        conf_threshold = self.conf_slider.value()
        iou_threshold = self.iou_slider.value()
        
        # Start inference thread
        self.inference_thread = InferenceThread(
            self.model_loader.get_model(),
            self.current_image,
            conf_threshold,
            iou_threshold,
            device='cpu'
        )
        
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.start()
    
    def on_inference_finished(self, result: dict):
        """
        Callback khi inference hoàn thành
        (Callback when inference complete)
        """
        # Display result image
        pixmap = cv_to_qpixmap(result['annotated_image'])
        self.result_image_label.setPixmap(pixmap)
        
        # Store result for saving
        self.current_result_image = result['annotated_image']
        
        # Update detections table
        self._update_detections_table(result['detections'])
        
        # Log
        self.log(
            f"✅ Inference hoàn thành!\n"
            f"   ⏱️ Thời gian: {result['inference_time']:.3f}s\n"
            f"   🎯 Detections: {result['num_detections']}"
        )
        
        # Update status
        self.status_label.set_status(
            f"Phát hiện {result['num_detections']} đối tượng "
            f"trong {result['inference_time']:.2f}s",
            "success"
        )
        
        # Re-enable button
        self.btn_run_inference.setEnabled(True)
        self.btn_run_inference.setText("▶ Chạy Inference")
        
        # Enable save button
        self.btn_save_result.setEnabled(True)
    
    def on_inference_error(self, error_msg: str):
        """
        Callback khi inference lỗi
        (Callback when inference error)
        """
        self.log(f"❌ {error_msg}")
        self.status_label.set_status(error_msg, "error")
        
        # Re-enable button
        self.btn_run_inference.setEnabled(True)
        self.btn_run_inference.setText("▶ Chạy Inference")
        
        QMessageBox.critical(self, "Lỗi Inference", error_msg)
    
    def save_result(self):
        """Lưu kết quả ảnh đã annotate"""
        if not hasattr(self, 'current_result_image'):
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu kết quả",
            "",
            "JPEG Image (*.jpg);;PNG Image (*.png);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Save image
        success = cv2.imwrite(file_path, self.current_result_image)
        
        if success:
            self.log(f"💾 Đã lưu kết quả: {os.path.basename(file_path)}")
            QMessageBox.information(
                self,
                "Thành công",
                "Đã lưu kết quả thành công!"
            )
        else:
            self.log(f"❌ Lỗi lưu file")
            QMessageBox.critical(
                self,
                "Lỗi",
                "Không thể lưu file!"
            )
    
    # === HELPER METHODS ===
    
    def _update_run_button_state(self):
        """Update trạng thái nút Run Inference"""
        can_run = (
            self.current_image is not None and
            self.model_loader.is_model_loaded()
        )
        
        self.btn_run_inference.setEnabled(can_run)
        
        if can_run:
            self.status_label.set_status("Sẵn sàng chạy inference!", "success")
    
    def _update_detections_table(self, detections: list):
        """
        Update bảng detections
        (Update detections table)
        """
        self.detections_table.setRowCount(len(detections))
        
        for i, det in enumerate(detections):
            # Class
            self.detections_table.setItem(
                i, 0, QTableWidgetItem(det['class_name'])
            )
            
            # Confidence
            conf_item = QTableWidgetItem(f"{det['confidence']:.3f}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.detections_table.setItem(i, 1, conf_item)
            
            # BBox (x1, y1)
            x1, y1, x2, y2 = det['bbox']
            bbox1_item = QTableWidgetItem(f"({x1}, {y1})")
            bbox1_item.setTextAlignment(Qt.AlignCenter)
            self.detections_table.setItem(i, 2, bbox1_item)
            
            # BBox (x2, y2)
            bbox2_item = QTableWidgetItem(f"({x2}, {y2})")
            bbox2_item.setTextAlignment(Qt.AlignCenter)
            self.detections_table.setItem(i, 3, bbox2_item)
    
    def log(self, message: str):
        """
        Thêm message vào log panel
        (Add message to log panel)
        """
        self.log_text.append(message)
        self.statusBar().showMessage(message.replace("\n", " | "))
