"""
Batch Processing Tool for YOLO Model Testing
Full-featured batch inference tool similar to main window but optimized for folder operations
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QFileDialog, QMessageBox, QGroupBox, QSplitter, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QWidget, QMenu
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import os
import cv2
import numpy as np
from typing import Dict, List, Any

from src.core.model_loader import ModelLoader
from src.core.inference import InferenceEngine
from src.gui.widgets import InfoPanel, SliderWithLabel, DeviceSelector, ImageLabel, StatusLabel
from src.utils.image_utils import load_image_cv, cv_to_qpixmap
from src.utils.export_utils import export_to_json, export_to_csv, generate_pdf_report


class BatchInferenceThread(QThread):
    """Thread for running batch inference without blocking UI"""
    
    progress_updated = pyqtSignal(int, int, str)  # current, total, filename
    image_processed = pyqtSignal(str, dict)  # filepath, result
    all_finished = pyqtSignal()
    error = pyqtSignal(str, str)  # filepath, error_msg
    
    def __init__(self, model, image_paths, conf_threshold, iou_threshold, device):
        super().__init__()
        self.model = model
        self.image_paths = image_paths
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.engine = InferenceEngine()
    
    def run(self):
        """Process all images"""
        total = len(self.image_paths)
        
        for idx, image_path in enumerate(self.image_paths, 1):
            try:
                filename = os.path.basename(image_path)
                self.progress_updated.emit(idx, total, filename)
                
                # Load image
                image = load_image_cv(image_path)
                if image is None:
                    self.error.emit(image_path, "Cannot load image")
                    continue
                
                # Run inference
                result = self.engine.run_inference(
                    self.model,
                    image,
                    self.conf_threshold,
                    self.iou_threshold,
                    self.device
                )
                
                if result:
                    result['filepath'] = image_path
                    result['filename'] = filename
                    result['original_image'] = image
                    self.image_processed.emit(image_path, result)
                else:
                    self.error.emit(image_path, "Inference failed")
                    
            except Exception as e:
                self.error.emit(image_path, str(e))
        
        self.all_finished.emit()


class BatchProcessDialog(QDialog):
    """
    Batch Processing Tool - Full-featured batch inference
    UI similar to main window but optimized for folder-based batch operations
    """
    
    def __init__(self, model, model_info, conf_threshold, iou_threshold, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.model_info = model_info
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # State management
        self.folder_path = None
        self.image_paths = []
        self.image_states = {}  # {filepath: {'processed': bool, 'detections': [], 'result_image': ndarray, 'original_image': ndarray}}
        self.current_selected_image = None
        self.current_image = None
        self.current_result_image = None
        self.current_detections = []
        self.current_inference_time = 0.0
        self.inference_thread = None
        
        self.setWindowTitle("Batch Processing Tool")
        self.setGeometry(100, 100, 1400, 950)
        self.setup_ui()
        
        # Initialize model info if provided
        if self.model_info:
            self._display_model_info()
    
    def setup_ui(self):
        """Setup UI matching main window structure"""
        # Main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # === LEFT PANEL: Controls ===
        left_panel = self.create_control_panel()
        left_panel.setMaximumWidth(320)
        main_layout.addWidget(left_panel)
        
        # === RIGHT PANEL: Display & Results ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(16)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image display area (top)
        image_splitter = self.create_image_display()
        right_layout.addWidget(image_splitter, stretch=3)
        
        # Results panel (bottom)
        results_panel = self.create_results_panel()
        right_layout.addWidget(results_panel, stretch=2)
        
        main_layout.addWidget(right_panel, stretch=1)
    
    def create_control_panel(self) -> QWidget:
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("YOLO Batch Tool")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # === FOLDER INPUT ===
        folder_group = QGroupBox("Folder Input")
        folder_layout = QVBoxLayout(folder_group)
        
        self.btn_select_folder = QPushButton("📁 Select Folder")
        self.btn_select_folder.setMinimumHeight(35)
        self.btn_select_folder.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.btn_select_folder)
        
        # Image count
        self.label_image_count = QLabel("Images: 0")
        self.label_image_count.setStyleSheet("font-weight: bold; color: #1E40AF; padding: 4px;")
        folder_layout.addWidget(self.label_image_count)
        
        # Image list (stretch to fill remaining vertical space)
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        folder_layout.addWidget(self.image_list, stretch=1)  # Add stretch factor
        
        layout.addWidget(folder_group, stretch=1)  # Add stretch factor to group
        
        # === MODEL ===
        model_group = QGroupBox("Model YOLO")
        model_layout = QVBoxLayout(model_group)
        
        self.btn_load_model = QPushButton("🧠 Load Model")
        self.btn_load_model.setMinimumHeight(35)
        self.btn_load_model.clicked.connect(self.load_model)
        model_layout.addWidget(self.btn_load_model)
        
        self.model_info_panel = InfoPanel("Model info")
        model_layout.addWidget(self.model_info_panel)
        
        layout.addWidget(model_group)
        
        # === INFERENCE SETTINGS ===
        settings_group = QGroupBox("Cài Đặt Inference")
        settings_layout = QVBoxLayout(settings_group)
        
        self.conf_slider = SliderWithLabel(
            "Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            default_value=self.conf_threshold,
            decimals=2
        )
        settings_layout.addWidget(self.conf_slider)
        
        self.iou_slider = SliderWithLabel(
            "IOU Threshold:",
            min_value=0.0,
            max_value=1.0,
            default_value=self.iou_threshold,
            decimals=2
        )
        settings_layout.addWidget(self.iou_slider)
        
        self.device_selector = DeviceSelector()
        settings_layout.addWidget(self.device_selector)
        
        layout.addWidget(settings_group)
        
        # === ACTION BUTTONS ===
        self.btn_run_inference = QPushButton("▶ Process All Images")
        self.btn_run_inference.setObjectName("successButton")
        self.btn_run_inference.setEnabled(False)
        self.btn_run_inference.clicked.connect(self.run_batch_inference)
        self.btn_run_inference.setMinimumHeight(35)
        font = self.btn_run_inference.font()
        font.setPointSize(12)
        font.setBold(True)
        self.btn_run_inference.setFont(font)
        layout.addWidget(self.btn_run_inference)
        
        # === EXPORT BUTTON ===
        self.btn_export = QPushButton("💾 Export Kết Quả")
        self.btn_export.setMinimumHeight(35)
        self.btn_export.setEnabled(False)
        
        # Export menu
        export_menu = QMenu(self)
        export_menu.addAction("💾 Save All Images", self.export_save_images)
        export_menu.addAction("📄 Export JSON", self.export_json)
        export_menu.addAction("📊 Export CSV", self.export_csv)
        export_menu.addAction("📑 Export PDF", self.export_pdf)
        self.btn_export.setMenu(export_menu)
        
        layout.addWidget(self.btn_export)
        
        # === STATUS ===
        self.status_label = StatusLabel()
        self.status_label.set_status("Select folder và model để bắt đầu", "info")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        return panel
    
    def create_image_display(self) -> QSplitter:
        """Create image display area"""
        splitter = QSplitter(Qt.Horizontal)
        
        # Original image
        original_container = QGroupBox("Ảnh Gốc")
        original_layout = QVBoxLayout(original_container)
        self.label_original_image = ImageLabel()
        original_layout.addWidget(self.label_original_image)
        splitter.addWidget(original_container)
        
        # Result image
        result_container = QGroupBox("Kết Quả Inference")
        result_layout = QVBoxLayout(result_container)
        self.label_result_image = ImageLabel()
        result_layout.addWidget(self.label_result_image)
        splitter.addWidget(result_container)
        
        return splitter
    
    def create_results_panel(self) -> QWidget:
        """Create results panel with detections table and logs"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # === DETECTIONS TABLE ===
        table_container = QGroupBox("Detections (Selected Image)")
        table_layout = QVBoxLayout(table_container)
        
        # Filter and histogram row
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Filter:"))
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItem("All Classes")
        self.class_filter_combo.currentIndexChanged.connect(self.on_class_filter_changed)
        filter_layout.addWidget(self.class_filter_combo, stretch=1)
        
        self.btn_histogram = QPushButton("📊 Histogram (All)")
        self.btn_histogram.setEnabled(False)
        self.btn_histogram.clicked.connect(self.show_histogram)
        filter_layout.addWidget(self.btn_histogram)
        
        table_layout.addLayout(filter_layout)
        
        # Table - Match main_window exactly (4 columns)
        self.table_detections = QTableWidget()
        self.table_detections.setColumnCount(4)
        self.table_detections.setHorizontalHeaderLabels([
            "Class", "Confidence", "BBox (x1, y1)", "BBox (x2, y2)"
        ])
        header = self.table_detections.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_detections.itemClicked.connect(self.on_detection_clicked)
        table_layout.addWidget(self.table_detections)
        
        layout.addWidget(table_container, stretch=2)
        
        # === LOGS ===
        log_container = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_container)
        log_layout.setSpacing(4)
        log_layout.setContentsMargins(8, 8, 8, 8)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_container, stretch=1)
        
        return panel
    
    # === IMPLEMENTATION METHODS ===
    
    def select_folder(self):
        """Select folder containing images"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Images",
            ""
        )
        
        if not folder_path:
            return
        
        self.folder_path = folder_path
        
        # Find all images in folder
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        self.image_paths = []
        
        try:
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(extensions):
                    filepath = os.path.join(folder_path, filename)
                    self.image_paths.append(filepath)
                    
                    # Initialize state for this image
                    self.image_states[filepath] = {
                        'processed': False,
                        'detections': [],
                        'result_image': None,
                        'original_image': None,
                        'inference_time': 0.0
                    }
            
            # Sort by filename
            self.image_paths.sort()
            
            # Update UI
            self.image_list.clear()
            for filepath in self.image_paths:
                filename = os.path.basename(filepath)
                self.image_list.addItem(filename)
            
            self.label_image_count.setText(f"Images: {len(self.image_paths)}")
            self.log(f"✅ Loaded {len(self.image_paths)} images from folder")
            
            self._update_ui_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot read folder: {str(e)}")
            self.log(f"❌ Error loading folder: {str(e)}")
    
    def load_model(self):
        """Load YOLO model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "resources/models",
            "PyTorch Model (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        model_loader = ModelLoader()
        success = model_loader.load_model(file_path)
        
        if not success:
            QMessageBox.critical(self, "Error", "Cannot load model!")
            return
        
        self.model = model_loader.get_model()
        self.model_info = model_loader.get_model_info()
        
        self._display_model_info()
        self.log(f"✅ Loaded model: {self.model_info['name']}")
        self._update_ui_state()
    
    def _display_model_info(self):
        """Display model information in panel"""
        if not self.model_info:
            return
        
        info_text = (
            f"📦 Model: {self.model_info['name']}\n"
            f"🏷️ Classes: {self.model_info['num_classes']}\n"
            f"💾 Size: {self.model_info['size_mb']} MB\n"
            f"📂 Type: {self.model_info['type']}"
        )
        self.model_info_panel.set_info(info_text)
    
    def on_image_selected(self, item):
        """Handle image selection from list"""
        if not item:
            return
        
        # Get filepath from filename
        filename = item.text()
        filepath = None
        for path in self.image_paths:
            if os.path.basename(path) == filename:
                filepath = path
                break
        
        if not filepath:
            return
        
        self.current_selected_image = filepath
        state = self.image_states.get(filepath, {})
        
        # Display original image
        if state.get('original_image') is not None:
            # Use cached image
            original_image = state['original_image']
        else:
            # Load and cache
            original_image = load_image_cv(filepath)
            if original_image is not None:
                self.image_states[filepath]['original_image'] = original_image
        
        if original_image is not None:
            pixmap = cv_to_qpixmap(original_image)
            self.label_original_image.setPixmap(pixmap)
            self.current_image = original_image
        
        # Display result if processed
        if state.get('processed'):
            result_image = state.get('result_image')
            if result_image is not None:
                pixmap = cv_to_qpixmap(result_image)
                self.label_result_image.setPixmap(pixmap)
                self.current_result_image = result_image
            
            # Update detections table
            self.current_detections = state.get('detections', [])
            self.current_inference_time = state.get('inference_time', 0.0)
            self._update_detections_table()
            self._update_class_filter()
        else:
            # Clear result side if not processed
            self.label_result_image.clear()
            self.label_result_image.setText("Not processed yet")
            self.current_result_image = None
            self.current_detections = []
            self.table_detections.setRowCount(0)
        
        self.log(f"📷 Selected: {filename}")
    
    def run_batch_inference(self):
        """Run inference on all images in folder"""
        if not self.model:
            QMessageBox.warning(self, "No Model", "Please load a model first!")
            return
        
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please select a folder with images!")
            return
        
        # Disable controls
        self.btn_run_inference.setEnabled(False)
        self.btn_select_folder.setEnabled(False)
        self.btn_load_model.setEnabled(False)
        
        self.log("🚀 Starting batch processing...")
        self.status_label.set_status("Processing images...", "info")
        
        # Start batch inference thread
        self.inference_thread = BatchInferenceThread(
            self.model,
            self.image_paths,
            self.conf_slider.value(),
            self.iou_slider.value(),
            self.device_selector.get_selected_device()
        )
        
        self.inference_thread.progress_updated.connect(self.on_progress_updated)
        self.inference_thread.image_processed.connect(self.on_image_processed)
        self.inference_thread.all_finished.connect(self.on_all_finished)
        self.inference_thread.error.connect(self.on_processing_error)
        self.inference_thread.start()
    
    def on_progress_updated(self, current: int, total: int, filename: str):
        """Handle progress update"""
        self.log(f"[{current}/{total}] Processing: {filename}")
        self.status_label.set_status(f"Processing {current}/{total}: {filename}", "info")
    
    def on_image_processed(self, filepath: str, result: dict):
        """Handle individual image processing completion"""
        # Store result in state
        self.image_states[filepath] = {
            'processed': True,
            'detections': result.get('detections', []),
            'result_image': result.get('annotated_image'),
            'original_image': result.get('original_image'),
            'inference_time': result.get('inference_time', 0.0)
        }
        
        # If this is the currently selected image, update display
        if filepath == self.current_selected_image:
            self.current_result_image = result.get('annotated_image')
            self.current_detections = result.get('detections', [])
            self.current_inference_time = result.get('inference_time', 0.0)
            
            # Update display
            pixmap = cv_to_qpixmap(self.current_result_image)
            self.label_result_image.setPixmap(pixmap)
            self._update_detections_table()
            self._update_class_filter()
        
        self.log(f"✅ Processed: {os.path.basename(filepath)} ({result.get('num_detections', 0)} detections)")
    
    def on_all_finished(self):
        """Handle batch processing completion"""
        processed_count = sum(1 for state in self.image_states.values() if state['processed'])
        total_detections = sum(len(state['detections']) for state in self.image_states.values())
        
        self.log(f"\n{'='*50}")
        self.log(f"✅ BATCH PROCESSING COMPLETE!")
        self.log(f"Processed: {processed_count}/{len(self.image_paths)} images")
        self.log(f"Total Detections: {total_detections}")
        self.log(f"{'='*50}\n")
        
        self.status_label.set_status(f"Complete! {processed_count} images processed", "success")
        
        # Re-enable controls
        self.btn_run_inference.setEnabled(True)
        self.btn_select_folder.setEnabled(True)
        self.btn_load_model.setEnabled(True)
        
        # Enable export
        self.btn_export.setEnabled(True)
        self.btn_histogram.setEnabled(True)
        
        # No popup - logs are sufficient
    
    def on_processing_error(self, filepath: str, error_msg: str):
        """Handle processing error"""
        filename = os.path.basename(filepath)
        self.log(f"❌ Error processing {filename}: {error_msg}")
    
    def _update_ui_state(self):
        """Update UI button states"""
        has_images = len(self.image_paths) > 0
        has_model = self.model is not None
        
        self.btn_run_inference.setEnabled(has_images and has_model)
    
    def _update_detections_table(self):
        """Update detections table for current image"""
        self.table_detections.setRowCount(0)
        
        if not self.current_detections:
            return
        
        # Apply class filter
        filter_class = self.class_filter_combo.currentText()
        
        for det in self.current_detections:
            if filter_class != "All Classes" and det['class_name'] != filter_class:
                continue
            
            row = self.table_detections.rowCount()
            self.table_detections.insertRow(row)
            
            # Match main_window format exactly - 4 columns
            # Class
            self.table_detections.setItem(row, 0, QTableWidgetItem(det['class_name']))
            
            # Confidence
            conf_item = QTableWidgetItem(f"{det['confidence']:.3f}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.table_detections.setItem(row, 1, conf_item)
            
            # BBox (x1, y1)
            x1, y1 = int(det['bbox'][0]), int(det['bbox'][1])
            bbox1_item = QTableWidgetItem(f"({x1}, {y1})")
            bbox1_item.setTextAlignment(Qt.AlignCenter)
            self.table_detections.setItem(row, 2, bbox1_item)
            
            # BBox (x2, y2)
            x2, y2 = int(det['bbox'][2]), int(det['bbox'][3])
            bbox2_item = QTableWidgetItem(f"({x2}, {y2})")
            bbox2_item.setTextAlignment(Qt.AlignCenter)
            self.table_detections.setItem(row, 3, bbox2_item)
    
    def _update_class_filter(self):
        """Update class filter dropdown"""
        # Get unique classes from current detections
        if not self.current_detections:
            return
        
        classes = set(det['class_name'] for det in self.current_detections)
        
        # Update combo box
        current_filter = self.class_filter_combo.currentText()
        self.class_filter_combo.clear()
        self.class_filter_combo.addItem("All Classes")
        
        for cls in sorted(classes):
            self.class_filter_combo.addItem(cls)
        
        # Restore previous selection if possible
        index = self.class_filter_combo.findText(current_filter)
        if index >= 0:
            self.class_filter_combo.setCurrentIndex(index)
    
    def run_batch_inference(self):
        """Run inference on all images in folder"""
        if not self.model:
            QMessageBox.warning(self, "No Model", "Please load a model first!")
            return
        
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please select a folder with images!")
            return
        
        # Disable controls
        self.btn_run_inference.setEnabled(False)
        self.btn_select_folder.setEnabled(False)
        self.btn_load_model.setEnabled(False)
        
        self.log("🚀 Starting batch processing...")
        self.status_label.set_status("Processing images...", "info")
        
        # Start batch inference thread
        self.inference_thread = BatchInferenceThread(
            self.model,
            self.image_paths,
            self.conf_slider.value(),
            self.iou_slider.value(),
            self.device_selector.get_selected_device()
        )
        
        self.inference_thread.progress_updated.connect(self.on_progress_updated)
        self.inference_thread.image_processed.connect(self.on_image_processed)
        self.inference_thread.all_finished.connect(self.on_all_finished)
        self.inference_thread.error.connect(self.on_processing_error)
        self.inference_thread.start()
    
    def on_progress_updated(self, current: int, total: int, filename: str):
        """Handle progress update"""
        self.log(f"[{current}/{total}] Processing: {filename}")
        self.status_label.set_status(f"Processing {current}/{total}: {filename}", "info")
    
    def on_image_processed(self, filepath: str, result: dict):
        """Handle individual image processing completion"""
        # Store result in state
        self.image_states[filepath] = {
            'processed': True,
            'detections': result.get('detections', []),
            'result_image': result.get('annotated_image'),
            'original_image': result.get('original_image'),
            'inference_time': result.get('inference_time', 0.0)
        }
        
        # If this is the currently selected image, update display
        if filepath == self.current_selected_image:
            self.current_result_image = result.get('annotated_image')
            self.current_detections = result.get('detections', [])
            self.current_inference_time = result.get('inference_time', 0.0)
            
            # Update display
            pixmap = cv_to_qpixmap(self.current_result_image)
            self.label_result_image.setPixmap(pixmap)
            self._update_detections_table()
            self._update_class_filter()
        
        self.log(f"✅ Processed: {os.path.basename(filepath)} ({result.get('num_detections', 0)} detections)")

    # === EXPORT METHODS ===
    
    def export_save_images(self):
        """Save all processed images"""
        # Check if any images processed
        processed = [fp for fp, state in self.image_states.items() if state.get('processed')]
        
        if not processed:
            QMessageBox.warning(self, "No Results", "No processed images to save!")
            return
        
        # Select output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Images",
            ""
        )
        
        if not output_dir:
            return
        
        try:
            saved_count = 0
            for filepath in processed:
                state = self.image_states[filepath]
                result_image = state.get('result_image')
                
                if result_image is not None:
                    # Generate output filename: {original_name}_output.{ext}
                    original_name = os.path.basename(filepath)
                    name_without_ext, ext = os.path.splitext(original_name)
                    output_filename = f"{name_without_ext}_output{ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save image
                    cv2.imwrite(output_path, result_image)
                    saved_count += 1
            
            self.log(f"✅ Saved {saved_count} images to {output_dir}")
            QMessageBox.information(
                self,
                "Success",
                f"Saved {saved_count} processed images to:\n{output_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save images: {str(e)}")
            self.log(f"❌ Error saving images: {str(e)}")
    
    def export_json(self):
        """Export batch results to JSON"""
        # Check if any images processed
        processed = {fp: state for fp, state in self.image_states.items() if state.get('processed')}
        
        if not processed:
            QMessageBox.warning(self, "No Results", "No processed images to export!")
            return
        
        # Select output file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON Export",
            "batch_results.json",
            "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            import json
            from datetime import datetime
            
            # Build JSON structure
            export_data = {
                "metadata": {
                    "export_time": datetime.now().isoformat(),
                    "tool": "YOLO Batch Processing Tool",
                    "total_images": len(self.image_paths),
                    "processed_images": len(processed)
                },
                "folder": {
                    "path": self.folder_path,
                    "total_images": len(self.image_paths)
                },
                "model": self.model_info if self.model_info else {},
                "inference": {
                    "confidence_threshold": self.conf_slider.value(),
                    "iou_threshold": self.iou_slider.value(),
                    "device": self.device_selector.get_selected_device(),
                    "total_detections": sum(len(state['detections']) for state in processed.values()),
                    "avg_detections_per_image": sum(len(state['detections']) for state in processed.values()) / len(processed) if processed else 0
                },
                "images": []
            }
            
            # Add per-image data
            for filepath, state in processed.items():
                image_data = {
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "inference_time": state.get('inference_time', 0.0),
                    "num_detections": len(state['detections']),
                    "detections": state['detections']
                }
                export_data["images"].append(image_data)
            
            # Save
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.log(f"✅ Exported JSON to {file_path}")
            QMessageBox.information(self, "Success", f"Exported JSON to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export JSON: {str(e)}")
            self.log(f"❌ Error exporting JSON: {str(e)}")
    
    def export_csv(self):
        """Export batch results to CSV"""
        # Check if any images processed
        processed = {fp: state for fp, state in self.image_states.items() if state.get('processed')}
        
        if not processed:
            QMessageBox.warning(self, "No Results", "No processed images to export!")
            return
        
        # Select output file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV Export",
            "batch_results.csv",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            import csv
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header row with Filename column first
                writer.writerow(['Filename', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
                
                # Write detections for each image
                for filepath, state in processed.items():
                    filename = os.path.basename(filepath)
                    detections = state.get('detections', [])
                    
                    for det in detections:
                        writer.writerow([
                            filename,
                            det['class_name'],
                            f"{det['confidence']:.4f}",
                            int(det['bbox'][0]),
                            int(det['bbox'][1]),
                            int(det['bbox'][2]),
                            int(det['bbox'][3])
                        ])
            
            self.log(f"✅ Exported CSV to {file_path}")
            QMessageBox.information(self, "Success", f"Exported CSV to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV: {str(e)}")
            self.log(f"❌ Error exporting CSV: {str(e)}")
    
    def export_pdf(self):
        """Export batch results to PDF"""
        # Check if any images processed
        processed = {fp: state for fp, state in self.image_states.items() if state.get('processed')}
        
        if not processed:
            QMessageBox.warning(self, "No Results", "No processed images to export!")
            return
        
        # Select output file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF Report",
            "batch_report.pdf",
            "PDF Files (*.pdf)"
        )
        
        if not file_path:
            return
        
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            import tempfile
            from datetime import datetime
            
            # Create PDF
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2563EB'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            story.append(Paragraph("YOLO Batch Processing Report", title_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Summary table
            summary_data = [
                ['Total Images', str(len(self.image_paths))],
                ['Processed Images', str(len(processed))],
                ['Total Detections', str(sum(len(state['detections']) for state in processed.values()))],
                ['Model', self.model_info.get('name', 'Unknown') if self.model_info else 'Unknown'],
                ['Confidence Threshold', f"{self.conf_slider.value():.2f}"],
                ['IOU Threshold', f"{self.iou_slider.value():.2f}"],
                ['Export Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            summary_table = Table(summary_data, colWidths=[2.5*inch, 4*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E0E7FF')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(summary_table)
            story.append(PageBreak())
            
            # Per-image results (first 10 images to keep PDF reasonable size)
            image_count = 0
            max_images = 10
            
            for filepath, state in list(processed.items())[:max_images]:
                filename = os.path.basename(filepath)
                
                # Image header
                story.append(Paragraph(f"Image: {filename}", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                
                # Save temp annotated image
                result_image = state.get('result_image')
                if result_image is not None:
                    # Fix: Use delete=False and manual cleanup to avoid file lock
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    # Write image after closing file handle
                    cv2.imwrite(tmp_path, result_image)
                    
                    # Add to PDF
                    img = Image(tmp_path, width=5*inch, height=3*inch)
                    story.append(img)
                    
                    # Clean up after adding to story (file is not locked anymore)
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass  # Ignore cleanup errors
                
                story.append(Spacer(1, 0.1*inch))
                
                # Detections table
                if state['detections']:
                    det_data = [['Class', 'Confidence', 'BBox']]
                    for det in state['detections']:
                        bbox_str = f"({int(det['bbox'][0])}, {int(det['bbox'][1])}, {int(det['bbox'][2])}, {int(det['bbox'][3])})"
                        det_data.append([
                            det['class_name'],
                            f"{det['confidence']:.2f}",
                            bbox_str
                        ])
                    
                    det_table = Table(det_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
                    det_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DBEAFE')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                    ]))
                    story.append(det_table)
                
                story.append(PageBreak())
                image_count += 1
                
        except Exception as e:
            self.log(f"❌ Error in previous export: {str(e)}")
            
    def export_pdf(self):
        """Export batch results to PDF"""
        # Check if any images processed
        processed = {fp: state for fp, state in self.image_states.items() if state.get('processed')}
        
        if not processed:
            QMessageBox.warning(self, "No Results", "No processed images to export!")
            return
        
        # Select output file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF Report",
            "batch_report.pdf",
            "PDF Files (*.pdf)"
        )
        
        if not file_path:
            return
        
        # Track temp files for cleanup
        temp_files = []
        
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            import tempfile
            from datetime import datetime
            import matplotlib.pyplot as plt
            from collections import Counter
            import numpy as np
            
            # Create PDF
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                alignment=TA_CENTER,
                fontSize=24,
                spaceAfter=20
            )
            story.append(Paragraph("YOLO Batch Processing Report", title_style))
            story.append(Spacer(1, 20))
            
            # Calculate Metrics
            processed_count = len(processed)
            total_detections = sum(len(state['detections']) for state in processed.values())
            
            # Calculate inference times
            inference_times = []
            for state in processed.values():
                if 'inference_time' in state:
                    inference_times.append(state['inference_time'])
            
            total_inference_time = sum(inference_times)
            avg_inference_time = total_inference_time / len(inference_times) if inference_times else 0
            
            # Prepare summary data
            device_id = self.device_selector.combo.currentData()
            device_display = "CUDA" if "cuda" in str(device_id).lower() else "CPU"
            folder_name = os.path.basename(self.folder_path)

            # Summary Table Data
            summary_data = [
                ["Metric", "Value"],
                ["Generated Report", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ["Model", self.model_info.get('name', 'Unknown') if self.model_info else 'Unknown'],
                ["Device", device_display],
                ["Folder", folder_name],
                ["Total Images", str(len(self.image_paths))],
                ["Processed Images", str(processed_count)],
                ["Total Detections", str(total_detections)],
                ["Total Inference Time", f"{total_inference_time:.3f} s"],
                ["Avg Inference Time", f"{avg_inference_time:.3f} s"],
                ["Confidence Threshold", f"{self.conf_slider.value():.2f}"],
                ["IOU Threshold", f"{self.iou_slider.value():.2f}"]
            ]
            
            t = Table(summary_data, colWidths=[3*inch, 4*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            
            # === STATISTICAL ANALYSIS ===
            story.append(PageBreak())
            story.append(Paragraph("Statistical Analysis", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # Aggregate data
            all_confidences = []
            all_classes = []
            for state in processed.values():
                for det in state['detections']:
                    all_confidences.append(det['confidence'])
                    all_classes.append(det['class_name'])
            
            if all_confidences:
                # 1. Confidence Distribution Chart
                plt.figure(figsize=(6, 4))
                plt.hist(all_confidences, bins=20, color='#3B82F6', edgecolor='black', alpha=0.7)
                plt.title('Confidence Distribution')
                plt.xlabel('Confidence Score')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.3)
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_conf:
                    plt.savefig(tmp_conf.name, bbox_inches='tight', dpi=100)
                    temp_files.append(tmp_conf.name)
                    story.append(Paragraph("Confidence Distribution", styles['Heading3']))
                    story.append(Image(tmp_conf.name, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 12))
                plt.close()
                
                # 2. Class Distribution Chart (Pie Chart)
                plt.figure(figsize=(6, 4))
                class_counts = Counter(all_classes)
                classes = list(class_counts.keys())
                counts = list(class_counts.values())
                
                # Sort by count desc
                sorted_indices = np.argsort(counts)[::-1]
                classes = [classes[i] for i in sorted_indices]
                counts = [counts[i] for i in sorted_indices]
                
                # Pie chart
                plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90, 
                        colors=plt.cm.Pastel1(np.linspace(0, 1, len(classes))))
                plt.title('Class Distribution')
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_class:
                    plt.savefig(tmp_class.name, bbox_inches='tight', dpi=100)
                    temp_files.append(tmp_class.name)
                    story.append(Paragraph("Class Distribution", styles['Heading3']))
                    story.append(Image(tmp_class.name, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 12))
                plt.close()
            else:
                story.append(Paragraph("No detections found to analyze.", styles['Normal']))
            
            story.append(PageBreak())
            
            # === DETAILED RESULTS ===
            story.append(Paragraph("Detailed Results", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # Limit images to avoid huge PDF
            max_images = 50
            image_count = 0
            
            for filepath, state in processed.items():
                if image_count >= max_images:
                    break
                
                filename = os.path.basename(filepath)
                story.append(Paragraph(f"Image: {filename}", styles['Heading3']))
                story.append(Spacer(1, 6))
                
                # Add Original Image
                original_image = state.get('original_image')
                if original_image is not None:
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_orig:
                        tmp_path = tmp_orig.name
                    
                    cv2.imwrite(tmp_path, original_image)
                    temp_files.append(tmp_path)
                    
                    story.append(Paragraph("Original Image:", styles['Heading4']))
                    story.append(Image(tmp_path, width=6*inch, height=4*inch, kind='proportional'))
                    story.append(Spacer(1, 12))
                
                # Add Processed Image
                result_image = state.get('result_image')
                if result_image is not None:
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_res:
                        tmp_path = tmp_res.name
                    
                    cv2.imwrite(tmp_path, result_image)
                    temp_files.append(tmp_path)
                    
                    story.append(Paragraph("Processed Image:", styles['Heading4']))
                    story.append(Image(tmp_path, width=6*inch, height=4*inch, kind='proportional'))
                    story.append(Spacer(1, 12))
                
                # Add Detections Table
                detections = state.get('detections', [])
                if detections:
                    det_data = [["Class", "Conf", "BBox"]]
                    for det in detections:
                        bbox = f"({int(det['bbox'][0])},{int(det['bbox'][1])}) - ({int(det['bbox'][2])},{int(det['bbox'][3])})"
                        det_data.append([
                            det['class_name'],
                            f"{det['confidence']:.2f}",
                            bbox
                        ])
                    
                    det_table = Table(det_data, colWidths=[2*inch, 1*inch, 3*inch])
                    det_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DBEAFE')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                    ]))
                    story.append(det_table)
                
                story.append(PageBreak())
                image_count += 1
            
            if len(processed) > max_images:
                story.append(Paragraph(
                    f"Note: Showing first {max_images} of {len(processed)} processed images",
                    styles['Normal']
                ))
            
            # Build PDF
            doc.build(story)
            
            self.log(f"✅ Exported PDF to {file_path}")
            QMessageBox.information(self, "Success", f"Exported PDF report to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export PDF: {str(e)}")
            self.log(f"❌ Error exporting PDF: {str(e)}")
            
        finally:
            # Cleanup temp files
            for path in temp_files:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except:
                    pass

    
    def show_histogram(self):
        """Show histogram for all processed images"""
        # Aggregate all detections from all processed images
        all_detections = []
        for state in self.image_states.values():
            if state.get('processed'):
                all_detections.extend(state.get('detections', []))
        
        if not all_detections:
            QMessageBox.warning(self, "No Data", "No processed detections to display!")
            return
        
        # Open histogram dialog (corrected arguments)
        from src.gui.histogram_dialog import HistogramDialog
        dialog = HistogramDialog(all_detections, self)
        dialog.exec_()
    
    def on_class_filter_changed(self): 
        """Handle class filter change"""
        self._update_detections_table()
    
    def on_detection_clicked(self, item):
        """Handle detection click"""
        if item:
            row = item.row()
            class_name = self.table_detections.item(row, 0).text()
            confidence = self.table_detections.item(row, 1).text()
            self.log(f"🎯 Selected detection: {class_name} (confidence: {confidence})")
    
    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
