"""
Model Comparison Window cho YOLO Model Testing Tool
Side-by-side comparison của multiple models
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
    Window để compare multiple YOLO models
    (Window to compare multiple YOLO models)
    """
    
    def __init__(self, conf_threshold, iou_threshold, device, parent=None):
        super().__init__(parent)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self.model_manager = ModelManager(max_models=4)
        self.current_image = None
        self.current_image_path = None
        self.comparison_results = {}
        self.comparison_thread = None
        
        self.setWindowTitle("Model Comparison Tool")
        self.setGeometry(100, 100, 1400, 900)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🔬 Multi-Model Comparison")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2563EB; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # === LEFT PANEL: Model Management ===
        left_panel = self.create_model_panel()
        left_panel.setMaximumWidth(350)
        splitter.addWidget(left_panel)
        
        # === RIGHT PANEL: Comparison Results ===
        right_panel = self.create_comparison_panel()
        splitter.addWidget(right_panel)
        
        layout.addWidget(splitter)
        
        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_close.setMinimumWidth(100)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
    
    def create_model_panel(self) -> QGroupBox:
        """Create model management panel"""
        panel = QGroupBox("Model Management")
        layout = QVBoxLayout(panel)
        
        # Model list
        list_label = QLabel("Loaded Models (Max: 4)")
        list_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(list_label)
        
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(200)
        layout.addWidget(self.model_list)
        
        # Model buttons
        btn_layout = QHBoxLayout()
        
        self.btn_add_model = QPushButton("➕ Add Model")
        self.btn_add_model.clicked.connect(self.add_model)
        btn_layout.addWidget(self.btn_add_model)
        
        self.btn_remove_model = QPushButton("🗑️ Remove")
        self.btn_remove_model.clicked.connect(self.remove_model)
        self.btn_remove_model.setEnabled(False)
        btn_layout.addWidget(self.btn_remove_model)
        
        layout.addLayout(btn_layout)
        
        # Image selection
        layout.addWidget(QLabel("Test Image"))
        
        self.image_label = QLabel("No image selected")
        self.image_label.setStyleSheet("padding: 10px; background: #F1F5F9; border-radius: 4px;")
        self.image_label.setWordWrap(True)
        layout.addWidget(self.image_label)
        
        self.btn_load_image = QPushButton("📁 Load Image")
        self.btn_load_image.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load_image)
        
        # Run comparison
        self.btn_compare = QPushButton("▶ Run Comparison")
        self.btn_compare.setObjectName("successButton")
        self.btn_compare.setEnabled(False)
        self.btn_compare.clicked.connect(self.run_comparison)
        self.btn_compare.setMinimumHeight(50)
        font = self.btn_compare.font()
        font.setPointSize(12)
        font.setBold(True)
        self.btn_compare.setFont(font)
        layout.addWidget(self.btn_compare)
        
        # Progress
        self.progress_label = QLabel("Ready")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("padding: 8px; background: #E0F2FE; border-radius: 4px;")
        layout.addWidget(self.progress_label)
        
        layout.addStretch()
        
        return panel
    
    def create_comparison_panel(self) -> QWidget:
        """Create comparison results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Comparison grid for images
        grid_group = QGroupBox("Visual Comparison")
        self.comparison_grid = QGridLayout(grid_group)
        self.comparison_grid.setSpacing(10)
        
        layout.addWidget(grid_group)
        
        # Metrics table
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
        
        self.metrics_table.setMaximumHeight(200)
        metrics_layout.addWidget(self.metrics_table)
        
        layout.addWidget(metrics_group)
        
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
            # Update list
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
                break
        
        self._update_model_list()
        self._update_ui_state()
    
    def load_image(self):
        """Load test image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Test Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self.current_image = load_image_cv(file_path)
        
        if self.current_image is None:
            QMessageBox.critical(self, "Error", "Cannot load image!")
            return
        
        self.current_image_path = file_path
        self.image_label.setText(f"📷 {file_path}")
        self._update_ui_state()
    
    def run_comparison(self):
        """Run comparison on all models"""
        if not self.current_image is not None:
            QMessageBox.warning(self, "No Image", "Please load an image first!")
            return
        
        # Clear previous results
        self.comparison_results = {}
        self._clear_comparison_grid()
        self.metrics_table.setRowCount(0)
        
        # Disable controls
        self.btn_compare.setEnabled(False)
        self.btn_add_model.setEnabled(False)
        self.btn_remove_model.setEnabled(False)
        self.btn_load_image.setEnabled(False)
        
        self.progress_label.setText("🔄 Running comparisons...")
        
        # Start comparison thread
        self.comparison_thread = ComparisonThread(
            self.model_manager.get_all_models(),
            self.current_image,
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
        self.progress_label.setText(f"🔄 Processing [{current}/{total}]: {model_name}")
    
    def on_model_finished(self, model_id: str, result: dict):
        """Handle individual model completion"""
        self.comparison_results[model_id] = result
        self._update_comparison_display()
    
    def on_all_finished(self, all_results: dict):
        """Handle all comparisons complete"""
        self.progress_label.setText(f"✅ Comparison complete! ({len(all_results)} models)")
        
        # Re-enable controls
        self.btn_compare.setEnabled(True)
        self.btn_add_model.setEnabled(True)
        self.btn_load_image.setEnabled(True)
        if self.model_list.count() > 0:
            self.btn_remove_model.setEnabled(True)
    
    def on_comparison_error(self, model_id: str, error_msg: str):
        """Handle comparison error"""
        print(f"❌ Error with {model_id}: {error_msg}")
    
    def _update_comparison_display(self):
        """Update comparison grid and metrics table"""
        # Clear grid
        self._clear_comparison_grid()
        
        # Display results in grid (2x2 layout)
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for idx, (model_id, result) in enumerate(self.comparison_results.items()):
            if idx >= 4:
                break
            
            row, col = positions[idx]
            
            # Create result widget
            result_widget = QGroupBox(result['model_name'])
            result_layout = QVBoxLayout(result_widget)
            
            # Image
            image_label = ImageLabel()
            pixmap = cv_to_qpixmap(result['annotated_image'])
            image_label.setPixmap(pixmap)
            result_layout.addWidget(image_label)
            
            # Stats
            stats_label = QLabel(
                f"Detections: {result['num_detections']} | "
                f"Time: {result['inference_time']:.3f}s"
            )
            stats_label.setAlignment(Qt.AlignCenter)
            stats_label.setStyleSheet("font-weight: bold; padding: 5px;")
            result_layout.addWidget(stats_label)
            
            self.comparison_grid.addWidget(result_widget, row, col)
        
        # Update metrics table
        self._update_metrics_table()
    
    def _update_metrics_table(self):
        """Update metrics comparison table"""
        self.metrics_table.setRowCount(len(self.comparison_results))
        
        for idx, (model_id, result) in enumerate(self.comparison_results.items()):
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
    
    def _clear_comparison_grid(self):
        """Clear comparison grid"""
        while self.comparison_grid.count():
            item = self.comparison_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
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
        has_image = self.current_image is not None
        
        self.btn_remove_model.setEnabled(has_models)
        self.btn_compare.setEnabled(has_models and has_image)
