"""
Model Comparison Window cho YOLO Model Testing Tool
Side-by-side comparison của multiple models với layout mới
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QFileDialog, QMessageBox,
    QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QTextEdit, QWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import cv2
import numpy as np
import time
import os

from src.core.model_manager import ModelManager
from src.core.inference import InferenceEngine
from src.gui.widgets import ImageLabel
from src.utils.image_utils import load_image_cv, cv_to_qpixmap


class ComparisonThread(QThread):
    """
    Thread để run inference trên multiple models
    (Thread to run inference on multiple models)
    """
    
    progress_updated = pyqtSignal(int, int, str)  # current, total, model_name
    model_finished = pyqtSignal(str, dict)  # model_id, result
    all_finished = pyqtSignal(dict)  # all_results
    error = pyqtSignal(str, str)  # model_id, error_msg
    
    def __init__(self, models_dict, image, conf_threshold, iou_threshold, device):
        super().__init__()
        self.models_dict = models_dict
        self.image = image
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.engine = InferenceEngine()
    
    def run(self):
        """Run inference on all models"""
        all_results = {}
        total = len(self.models_dict)
        current = 0
        
        for model_id, model_data in self.models_dict.items():
            try:
                current += 1
                self.progress_updated.emit(current, total, model_data['name'])
                
                # Run inference
                result = self.engine.run_inference(
                    model_data['model'],
                    self.image,
                    self.conf_threshold,
                    self.iou_threshold,
                    self.device
                )
                
                if result:
                    result['model_id'] = model_id
                    result['model_name'] = model_data['name']
                    all_results[model_id] = result
                    self.model_finished.emit(model_id, result)
                else:
                    self.error.emit(model_id, "Inference failed")
                    
            except Exception as e:
                self.error.emit(model_id, str(e))
        
        self.all_finished.emit(all_results)


class ModelComparisonWindow(QDialog):
    """
    Window để compare multiple YOLO models với layout mới
    (Window to compare multiple YOLO models with new layout)
    """
    
    def __init__(self, conf_threshold, iou_threshold, device, parent=None):
        super().__init__(parent)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self.model_manager = ModelManager(max_models=4)
        self.test_image = None
        self.test_image_path = None
        self.model_results = {}
        self.comparison_thread = None
        
        self.setWindowTitle("Model Comparison Tool")
        self.setGeometry(100, 100, 1400, 950)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        
        # Main content splitter (no title)
        splitter = QSplitter(Qt.Horizontal)
        
        # === LEFT PANEL: Model Management ===
        left_panel = self.create_model_panel()
        left_panel.setMaximumWidth(350)
        splitter.addWidget(left_panel)
        
        # === RIGHT PANEL: Comparison Results ===
        right_panel = self.create_comparison_panel()
        splitter.addWidget(right_panel)
        
        layout.addWidget(splitter)
    
    def create_model_panel(self) -> QGroupBox:
        """Create model management panel với layout mới"""
        panel = QGroupBox("Model Management")
        layout = QVBoxLayout(panel)
        layout.setSpacing(0)  # Set to 0 to control spacing manually
        layout.setContentsMargins(8, 4, 8, 8)  # Tight margins
        
        # === 1. MODEL LIST (Max 4 models, reduced height) ===
        list_label = QLabel("Loaded Models (Max: 4)")
        list_label.setStyleSheet("font-weight: bold; font-size: 11px; line-height: 12px; padding: 0px;")
        list_label.setContentsMargins(0, 0, 0, 0)
        list_label.setSizePolicy(list_label.sizePolicy().horizontalPolicy(), 0)  # Fixed height
        layout.addWidget(list_label)
        layout.addSpacing(2)  # 2px spacing
        
        self.model_list = QListWidget()
        # Height for exactly 4 rows
        self.model_list.setMaximumHeight(100)
        self.model_list.setMinimumHeight(100)
        layout.addWidget(self.model_list)
        layout.addSpacing(4)  # 4px spacing
        
        # === 2. ADD/REMOVE BUTTONS (35px height each) ===
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.btn_add_model = QPushButton("➕ Add Model")
        self.btn_add_model.setMinimumHeight(35)
        self.btn_add_model.setMaximumHeight(35)
        self.btn_add_model.clicked.connect(self.add_model)
        btn_layout.addWidget(self.btn_add_model)
        
        self.btn_remove_model = QPushButton("🗑️ Remove")
        self.btn_remove_model.setMinimumHeight(35)
        self.btn_remove_model.setMaximumHeight(35)
        self.btn_remove_model.clicked.connect(self.remove_model)
        self.btn_remove_model.setEnabled(False)
        btn_layout.addWidget(self.btn_remove_model)
        
        layout.addLayout(btn_layout)
        layout.addSpacing(4)  # 4px spacing
        
        # === 3. LOAD IMAGE BUTTON (35px height) ===
        self.btn_load_image = QPushButton("📁 Load Image")
        self.btn_load_image.setMinimumHeight(35)
        self.btn_load_image.setMaximumHeight(35)
        self.btn_load_image.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load_image)
        layout.addSpacing(4)  # 4px spacing
        
        # === 4. IMAGE PATH LABEL ===
        self.image_path_label = QLabel("No image selected")
        self.image_path_label.setStyleSheet("""
            padding: 6px; 
            background: #F1F5F9; 
            border-radius: 4px;
            font-size: 10px;
            color: #64748B;
            line-height: 12px;
        """)
        self.image_path_label.setWordWrap(True)
        self.image_path_label.setMaximumHeight(40)  # Limit to ~2 lines (12px line-height * 2 + 12px padding)
        layout.addWidget(self.image_path_label)
        layout.addSpacing(2)  # 2px spacing
        
        # === 5. IMAGE PREVIEW (Small) ===
        preview_label = QLabel("Preview:")
        preview_label.setStyleSheet("font-weight: bold; font-size: 10px; line-height: 11px; padding: 0px;")
        preview_label.setContentsMargins(0, 0, 0, 0)
        preview_label.setSizePolicy(preview_label.sizePolicy().horizontalPolicy(), 0)  # Fixed height
        layout.addWidget(preview_label)
        layout.addSpacing(2)  # 2px spacing
        
        # Image preview - use QLabel for proper center alignment
        self.image_preview = QLabel()
        self.image_preview.setMinimumHeight(200)
        self.image_preview.setMaximumHeight(200)
        self.image_preview.setAlignment(Qt.AlignCenter)  # This will work with QLabel
        self.image_preview.setScaledContents(False)
        self.image_preview.setStyleSheet("""
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 4px;
        """)
        self.image_preview.setText("No image loaded")
        layout.addWidget(self.image_preview)
        layout.addSpacing(4)  # 4px spacing
        
        # === 6. RUN COMPARISON BUTTON (35px height) ===
        self.btn_compare = QPushButton("▶ Run Comparison")
        self.btn_compare.setObjectName("successButton")
        self.btn_compare.setMinimumHeight(35)
        self.btn_compare.setMaximumHeight(35)
        self.btn_compare.setEnabled(False)
        self.btn_compare.clicked.connect(self.run_comparison)
        font = self.btn_compare.font()
        font.setBold(True)
        self.btn_compare.setFont(font)
        layout.addWidget(self.btn_compare)
        layout.addSpacing(2)  # 2px spacing
        
        # === 7. LOGS SECTION ===
        logs_label = QLabel("Logs:")
        logs_label.setStyleSheet("font-weight: bold; font-size: 10px; line-height: 11px; padding: 0px;")
        logs_label.setContentsMargins(0, 0, 0, 0)
        logs_label.setSizePolicy(logs_label.sizePolicy().horizontalPolicy(), 0)  # Fixed height
        layout.addWidget(logs_label)
        layout.addSpacing(2)  # 2px spacing
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)  # Ensure minimum space
        # No max height - will take remaining space
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: #0F172A;
                color: #E2E8F0;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px;
                font-size: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
        """)
        layout.addWidget(self.log_text)
        
        return panel
    
    def create_comparison_panel(self) -> QWidget:
        """Create comparison results panel với 4 fixed slots và 3:1 height ratio"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # === VISUAL COMPARISON (3 parts of height) ===
        grid_group = QGroupBox("Visual Comparison")
        grid_layout = QVBoxLayout(grid_group)
        
        # Create 2x2 grid for 4 fixed slots
        self.comparison_grid_widget = QWidget()
        self.comparison_grid = QGridLayout(self.comparison_grid_widget)
        self.comparison_grid.setSpacing(10)
        
        # Initialize 4 empty slots
        self.comparison_slots = []
        for row in range(2):
            for col in range(2):
                slot_container = QWidget()
                slot_layout = QVBoxLayout(slot_container)
                slot_layout.setContentsMargins(0, 0, 0, 0)
                slot_layout.setSpacing(4)
                
                # Model name label
                model_label = QLabel(f"Slot {row*2 + col + 1}")
                model_label.setAlignment(Qt.AlignCenter)
                model_label.setStyleSheet("""
                    font-size: 10px;
                    font-weight: bold;
                    padding: 4px;
                    background: #1E293B;
                    color: #94A3B8;
                    border-radius: 4px;
                """)
                slot_layout.addWidget(model_label)
                
                # Image display
                image_label = ImageLabel()
                image_label.setMinimumHeight(200)
                image_label.setStyleSheet("""
                    background: #0F172A;
                    border: 2px dashed #334155;
                    border-radius: 4px;
                """)
                slot_layout.addWidget(image_label, stretch=1)
                
                self.comparison_grid.addWidget(slot_container, row, col)
                self.comparison_slots.append({
                    'container': slot_container,
                    'label': model_label,
                    'image': image_label,
                    'model_id': None
                })
        
        # Set equal stretch for all rows and columns to ensure uniform sizing
        self.comparison_grid.setRowStretch(0, 1)
        self.comparison_grid.setRowStretch(1, 1)
        self.comparison_grid.setColumnStretch(0, 1)
        self.comparison_grid.setColumnStretch(1, 1)
        
        grid_layout.addWidget(self.comparison_grid_widget)
        
        # Set stretch factor for visual comparison (3 parts)
        layout.addWidget(grid_group, stretch=3)
        
        # === PERFORMANCE METRICS (1 part of height) ===
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels([
            "Model", "Detections", "Inference Time (s)", "Avg Confidence", "FPS"
        ])
        
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        
        metrics_layout.addWidget(self.metrics_table)
        
        # Set stretch factor for metrics table (1 part)
        layout.addWidget(metrics_group, stretch=1)
        
        return panel
    
    def add_model(self):
        """Add a model to comparison"""
        if self.model_manager.get_model_count() >= 4:
            QMessageBox.warning(self, "Max Models", "Maximum 4 models allowed!")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "resources/models",
            "PyTorch Model (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Load model
        model_id = self.model_manager.load_model(file_path)
        
        if model_id:
            self.log(f"✅ Added model: {os.path.basename(file_path)}")
            self._update_model_list()
            self._update_ui_state()
    
    def remove_model(self):
        """Remove selected model"""
        current_item = self.model_list.currentItem()
        if not current_item:
            return
        
        # Extract model_id from item text
        model_name = current_item.text()
        
        # Find and remove
        for model_id, data in self.model_manager.get_all_models().items():
            if data['name'] in model_name:
                self.model_manager.remove_model(model_id)
                self.log(f"🗑️ Removed model: {data['name']}")
                break
        
        self._update_model_list()
        self._update_ui_state()
    
    def load_image(self):
        """Load test image và hiển thị preview"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Test Image",
            "resources/sample_images",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Load image
        self.test_image = load_image_cv(file_path)
        
        if self.test_image is None:
            QMessageBox.critical(self, "Error", "Failed to load image!")
            return
        
        self.test_image_path = file_path
        
        # Update image path label
        self.image_path_label.setText(file_path)
        
        # Display preview - scale to fit while keeping aspect ratio
        pixmap = cv_to_qpixmap(self.test_image)
        scaled_pixmap = pixmap.scaled(
            self.image_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_preview.setPixmap(scaled_pixmap)
        
        self.log(f"✅ Loaded image: {os.path.basename(file_path)}")
        self._update_ui_state()
    
    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
    
    def run_comparison(self):
        """Run comparison on all models"""
        if self.test_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first!")
            return
        
        # Clear previous results
        self.model_results = {}
        self._clear_comparison_display()
        self.metrics_table.setRowCount(0)
        
        # Disable controls
        self.btn_compare.setEnabled(False)
        self.btn_add_model.setEnabled(False)
        self.btn_remove_model.setEnabled(False)
        self.btn_load_image.setEnabled(False)
        
        self.log("🔄 Starting comparison...")
        
        # Start comparison thread
        self.comparison_thread = ComparisonThread(
            self.model_manager.get_all_models(),
            self.test_image,
            self.conf_threshold,
            self.iou_threshold,
            self.device
        )
        
        self.comparison_thread.progress_updated.connect(self.on_progress_updated)
        self.comparison_thread.model_finished.connect(self.on_model_finished)
        self.comparison_thread.all_finished.connect(self.on_all_finished)
        self.comparison_thread.error.connect(self.on_comparison_error)
        self.comparison_thread.start()
    
    def on_progress_updated(self, current: int, total: int, model_name: str):
        """Handle progress update"""
        self.log(f"🔄 Processing [{current}/{total}]: {model_name}")
    
    def on_model_finished(self, model_id: str, result: dict):
        """Handle individual model completion"""
        self.model_results[model_id] = result
        self._update_comparison_display()
        self._update_metrics_table()
    
    def on_all_finished(self, all_results: dict):
        """Handle all comparisons complete"""
        self.log(f"✅ Comparison complete! ({len(all_results)} models)")
        
        # Re-enable controls
        self.btn_compare.setEnabled(True)
        self.btn_add_model.setEnabled(True)
        self.btn_load_image.setEnabled(True)
        if self.model_list.count() > 0:
            self.btn_remove_model.setEnabled(True)
    
    def on_comparison_error(self, model_id: str, error_msg: str):
        """Handle comparison error"""
        self.log(f"❌ Error with {model_id}: {error_msg}")
    
    def _update_comparison_display(self):
        """Update comparison grid with results on fixed slots"""
        # Fill slots with results
        for idx, (model_id, result) in enumerate(self.model_results.items()):
            if idx >= 4:  # Max 4 slots
                break
            
            slot = self.comparison_slots[idx]
            
            # Update slot label
            slot['label'].setText(result['model_name'])
            slot['label'].setStyleSheet("""
                font-size: 10px;
                font-weight: bold;
                padding: 4px;
                background: #2563EB;
                color: #FFFFFF;
                border-radius: 4px;
            """)
            
            # Update slot image
            pixmap = cv_to_qpixmap(result['annotated_image'])
            slot['image'].setPixmap(pixmap)
            slot['image'].setStyleSheet("""
                background: #0F172A;
                border: 2px solid #2563EB;
                border-radius: 4px;
            """)
            
            slot['model_id'] = model_id
    
    def _update_metrics_table(self):
        """Update metrics comparison table"""
        self.metrics_table.setRowCount(len(self.model_results))
        
        for idx, (model_id, result) in enumerate(self.model_results.items()):
            # Model name
            self.metrics_table.setItem(idx, 0, QTableWidgetItem(result['model_name']))
            
            # Detections
            det_item = QTableWidgetItem(str(result['num_detections']))
            det_item.setTextAlignment(Qt.AlignCenter)
            self.metrics_table.setItem(idx, 1, det_item)
            
            # Inference time
            time_item = QTableWidgetItem(f"{result['inference_time']:.4f}")
            time_item.setTextAlignment(Qt.AlignCenter)
            self.metrics_table.setItem(idx, 2, time_item)
            
            # Avg confidence
            if result['detections']:
                avg_conf = sum(d['confidence'] for d in result['detections']) / len(result['detections'])
            else:
                avg_conf = 0.0
            
            conf_item = QTableWidgetItem(f"{avg_conf:.3f}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.metrics_table.setItem(idx, 3, conf_item)
            
            # FPS
            fps = 1.0 / result['inference_time'] if result['inference_time'] > 0 else 0
            fps_item = QTableWidgetItem(f"{fps:.2f}")
            fps_item.setTextAlignment(Qt.AlignCenter)
            self.metrics_table.setItem(idx, 4, fps_item)
    
    def _clear_comparison_display(self):
        """Clear comparison display - reset all slots"""
        for idx, slot in enumerate(self.comparison_slots):
            slot['label'].setText(f"Slot {idx + 1}")
            slot['label'].setStyleSheet("""
                font-size: 10px;
                font-weight: bold;
                padding: 4px;
                background: #1E293B;
                color: #94A3B8;
                border-radius: 4px;
            """)
            slot['image'].clear_image()
            slot['image'].setStyleSheet("""
                background: #0F172A;
                border: 2px dashed #334155;
                border-radius: 4px;
            """)
            slot['model_id'] = None
    
    def _update_model_list(self):
        """Update model list widget"""
        self.model_list.clear()
        
        for model_data in self.model_manager.get_model_list():
            self.model_list.addItem(
                f"{model_data['name']} ({model_data['num_classes']} classes)"
            )
    
    def _update_ui_state(self):
        """Update UI button states"""
        has_models = self.model_manager.get_model_count() > 0
        has_image = self.test_image is not None
        
        self.btn_remove_model.setEnabled(has_models and self.model_list.currentItem() is not None)
        self.btn_compare.setEnabled(has_models and has_image)
