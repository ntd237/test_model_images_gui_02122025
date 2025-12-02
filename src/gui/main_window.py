"""
Main Window cho YOLO Model Testing Tool
UI chính với layout Master-Detail
"""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QTextEdit, QGroupBox, QFileDialog, QMessageBox,
    QHeaderView, QLabel, QMenu, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import cv2
import numpy as np

from src.gui.widgets import ImageLabel, InfoPanel, SliderWithLabel, StatusLabel, DeviceSelector
from src.core.model_loader import ModelLoader
from src.core.inference import InferenceEngine
from src.utils.image_utils import (
    load_image_cv, cv_to_qpixmap, validate_image_format, get_image_info
)
from src.utils.export_utils import (
    export_to_json, export_to_csv, generate_pdf_report
)
from src.gui.histogram_dialog import HistogramDialog
from src.gui.batch_dialog import BatchProcessDialog
from src.gui.comparison_window import ModelComparisonWindow


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
        self.setGeometry(100, 100, 1400, 950)
        
        # Initialize components
        self.model_loader = ModelLoader()
        self.current_image = None
        self.current_image_path = None
        self.inference_thread = None
        self.current_detections = []  # Store detections for export
        self.current_inference_time = 0.0  # Store inference time
        
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
        right_layout.addWidget(results_panel, stretch=2)
        
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
        self.btn_load_image.setMinimumHeight(35)
        self.btn_load_image.clicked.connect(self.load_image)
        image_layout.addWidget(self.btn_load_image)
        
        self.image_info_panel = InfoPanel("Thông tin ảnh")
        image_layout.addWidget(self.image_info_panel)
        
        layout.addWidget(image_group)
        
        # === MODEL SECTION ===
        model_group = QGroupBox("Model YOLO")
        model_layout = QVBoxLayout(model_group)
        
        self.btn_load_model = QPushButton("🧠 Load Model")
        self.btn_load_model.setMinimumHeight(35)
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
            default_value=0.5,
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
        
        # Device selector
        self.device_selector = DeviceSelector()
        settings_layout.addWidget(self.device_selector)
        
        layout.addWidget(settings_group)
        
        # === INFERENCE BUTTON ===
        self.btn_run_inference = QPushButton("▶ Chạy Inference")
        self.btn_run_inference.setObjectName("successButton")
        self.btn_run_inference.setEnabled(False)
        self.btn_run_inference.clicked.connect(self.run_inference)
        self.btn_run_inference.setMinimumHeight(35)
        font = self.btn_run_inference.font()
        font.setPointSize(14)
        font.setBold(True)
        self.btn_run_inference.setFont(font)
        layout.addWidget(self.btn_run_inference)
        
        # === BATCH PROCESSING BUTTON ===
        self.btn_batch_processing = QPushButton("📁 Batch Processing")
        self.btn_batch_processing.setMinimumHeight(35)
        self.btn_batch_processing.setEnabled(False)
        self.btn_batch_processing.clicked.connect(self.open_batch_processing)
        layout.addWidget(self.btn_batch_processing)
        
        # === MODEL COMPARISON BUTTON ===
        self.btn_model_comparison = QPushButton("🔬 Model Comparison")
        self.btn_model_comparison.setMinimumHeight(35)
        self.btn_model_comparison.clicked.connect(self.open_model_comparison)
        layout.addWidget(self.btn_model_comparison)
        
        # === STATUS ===
        self.status_label = StatusLabel()
        self.status_label.set_status("Load ảnh và model để bắt đầu", "info")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # === EXPORT MENU BUTTON ===
        self.btn_export = QPushButton("💾 Export Kết Quả")
        self.btn_export.setMinimumHeight(35)
        self.btn_export.setEnabled(False)
        
        # Create export menu
        export_menu = QMenu(self)
        export_menu.addAction("💾 Save Image", self.save_result_image)
        export_menu.addAction("📄 Export JSON", self.export_json)
        export_menu.addAction("📊 Export CSV", self.export_csv)
        export_menu.addAction("📑 Generate PDF Report", self.export_pdf)
        
        self.btn_export.setMenu(export_menu)
        layout.addWidget(self.btn_export)
        
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
        
        # Connect row click event
        self.detections_table.itemClicked.connect(self.on_detection_clicked)
        
        table_layout.addWidget(self.detections_table)
        
        # Class filter and histogram button
        filter_layout = QHBoxLayout()
        
        # Class filter
        filter_layout.addWidget(QLabel("Filter:"))
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItem("All Classes")
        self.class_filter_combo.currentIndexChanged.connect(self.on_class_filter_changed)
        filter_layout.addWidget(self.class_filter_combo, stretch=1)
        
        # Histogram button
        self.btn_histogram = QPushButton("📊 Histogram")
        self.btn_histogram.setEnabled(False)
        self.btn_histogram.clicked.connect(self.show_histogram)
        filter_layout.addWidget(self.btn_histogram)
        
        table_layout.addLayout(filter_layout)
        
        layout.addWidget(table_container, stretch=2)
        
        # === LOG PANEL ===
        log_container = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_container)
        log_layout.setSpacing(4)  # Reduce spacing between elements
        log_layout.setContentsMargins(8, 8, 8, 8)  # Balanced padding
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setContentsMargins(0, 0, 0, 0)  # Remove internal margins
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
        
        # Enable batch processing button
        self.btn_batch_processing.setEnabled(True)
    
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
        
        # Get thresholds and device
        conf_threshold = self.conf_slider.value()
        iou_threshold = self.iou_slider.value()
        selected_device = self.device_selector.get_selected_device()
        
        # Log device info
        self.log(f"💻 Device: {selected_device}")
        
        # Start inference thread
        self.inference_thread = InferenceThread(
            self.model_loader.get_model(),
            self.current_image,
            conf_threshold,
            iou_threshold,
            device=selected_device
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
        
        # Store result for export
        self.current_result_image = result['annotated_image']
        self.current_detections = result['detections']
        self.current_inference_time = result['inference_time']
        
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
        
        # Enable export button
        self.btn_export.setEnabled(True)
        
        # Enable histogram button
        self.btn_histogram.setEnabled(True)
        
        # Update class filter with detected classes
        self._update_class_filter(result['detections'])
    
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
    
    
    def save_result_image(self):
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
    
    def export_json(self):
        """
        Export detection results sang JSON
        (Export detection results to JSON)
        """
        if not self.current_detections:
            QMessageBox.warning(self, "No Data", "Chưa có kết quả inference!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export JSON",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Export
        success = export_to_json(
            self.current_detections,
            self.current_image_path,
            self.model_loader.get_model_info(),
            file_path,
            self.current_inference_time
        )
        
        if success:
            self.log(f"📄 Exported to JSON: {os.path.basename(file_path)}")
            QMessageBox.information(self, "Success", "Export JSON thành công!")
        else:
            QMessageBox.critical(self, "Error", "Không thể export JSON!")
    
    def export_csv(self):
        """
        Export detections table sang CSV
        (Export detections table to CSV)
        """
        if not self.current_detections:
            QMessageBox.warning(self, "No Data", "Chưa có kết quả inference!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Export
        success = export_to_csv(self.current_detections, file_path)
        
        if success:
            self.log(f"📊 Exported to CSV: {os.path.basename(file_path)}")
            QMessageBox.information(self, "Success", "Export CSV thành công!")
        else:
            QMessageBox.critical(self, "Error", "Không thể export CSV!")
    
    def export_pdf(self):
        """
        Generate comprehensive PDF report
        (Generate comprehensive PDF report)
        """
        if not self.current_detections or not hasattr(self, 'current_result_image'):
            QMessageBox.warning(self, "No Data", "Chưa có kết quả inference!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Generate PDF Report",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Show progress message
        self.log("📑 Generating PDF report...")
        
        # Get selected device
        device = self.device_selector.get_selected_device()
        
        # Generate PDF
        success = generate_pdf_report(
            self.current_detections,
            self.current_image_path,
            self.current_result_image,
            self.model_loader.get_model_info(),
            file_path,
            self.current_inference_time,
            device
        )
        
        if success:
            self.log(f"📑 Generated PDF: {os.path.basename(file_path)}")
            QMessageBox.information(
                self,
                "Success",
                f"PDF report đã được tạo!\n{os.path.basename(file_path)}"
            )
        else:
            QMessageBox.critical(self, "Error", "Không thể tạo PDF report!")
    
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
    
    def _update_class_filter(self, detections: list):
        """
        Update class filter combo với các class đã detect
        (Update class filter combo with detected classes)
        """
        # Clear current items (except "All Classes")
        self.class_filter_combo.clear()
        self.class_filter_combo.addItem("All Classes")
        
        # Get unique classes
        unique_classes = set()
        for det in detections:
            unique_classes.add(det['class_name'])
        
        # Add to combo
        for class_name in sorted(unique_classes):
            self.class_filter_combo.addItem(class_name)
    
    def on_detection_clicked(self, item):
        """
        Handle detection table row click - highlight bbox
        (Handle detection table row click - highlight bbox)
        """
        row = item.row()
        
        if row < 0 or row >= len(self.current_detections):
            return
        
        # Get detection
        detection = self.current_detections[row]
        
        # Log highlight
        self.log(f"🎯 Highlighted: {detection['class_name']} (conf: {detection['confidence']:.3f})")
        
        # TODO: Implement bbox highlighting on image
        # For now, just select the row
        self.detections_table.selectRow(row)
    
    def on_class_filter_changed(self, index):
        """
        Handle class filter change - filter table
        (Handle class filter change - filter table)
        """
        if index == 0:  # "All Classes"
            # Show all rows
            for row in range(self.detections_table.rowCount()):
                self.detections_table.setRowHidden(row, False)
            self.log("📋 Showing all classes")
        else:
            # Get selected class
            selected_class = self.class_filter_combo.currentText()
            
            # Filter rows
            hidden_count = 0
            for row in range(self.detections_table.rowCount()):
                class_item = self.detections_table.item(row, 0)
                if class_item:
                    class_name = class_item.text()
                    should_hide = (class_name != selected_class)
                    self.detections_table.setRowHidden(row, should_hide)
                    if should_hide:
                        hidden_count += 1
            
            visible_count = self.detections_table.rowCount() - hidden_count
            self.log(f"🔍 Filtered to '{selected_class}' ({visible_count} detections)")
    
    def show_histogram(self):
        """
        Show confidence histogram popup dialog
        (Show confidence histogram popup dialog)
        """
        if not self.current_detections:
            QMessageBox.warning(self, "No Data", "Chưa có kết quả inference!")
            return
        
        # Create and show histogram dialog
        dialog = HistogramDialog(self.current_detections, self)
        dialog.exec_()
    
    def open_batch_processing(self):
        """
        Open batch processing dialog
        (Open batch processing dialog)
        """
        # Create and show batch dialog (model can be None)
        dialog = BatchProcessDialog(
            self.model_loader.get_model() if self.model_loader.is_model_loaded() else None,
            self.model_loader.get_model_info() if self.model_loader.is_model_loaded() else None,
            self.conf_slider.value(),
            self.iou_slider.value(),
            self.device_selector.get_selected_device(),
            self
        )
        dialog.exec_()
    
    def open_model_comparison(self):
        """
        Open model comparison window
        (Open model comparison window)
        """
        # Create and show comparison window
        dialog = ModelComparisonWindow(
            self.conf_slider.value(),
            self.iou_slider.value(),
            self.device_selector.get_selected_device(),
            self
        )
        dialog.exec_()
    
    def log(self, message: str):
        """
        Thêm message vào log panel
        (Add message to log panel)
        """
        self.log_text.append(message)
        self.statusBar().showMessage(message.replace("\n", " | "))
