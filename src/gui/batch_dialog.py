"""
Batch Processing Dialog cho YOLO Model Testing Tool
UI để select và process multiple images
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QFileDialog, QProgressBar,
    QTextEdit, QGroupBox, QCheckBox, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os

from src.core.batch_processor import BatchProcessor
from src.core.model_loader import ModelLoader


class BatchProcessingThread(QThread):
    """
    Thread để chạy batch processing không block UI
    (Thread to run batch processing without blocking UI)
    """
    
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(dict)  # summary
    error = pyqtSignal(str)
    
    def __init__(
        self,
        model,
        image_paths,
        conf_threshold,
        iou_threshold,
        device,
        output_dir,
        save_intermediate,
        model_info
    ):
        super().__init__()
        self.model = model
        self.image_paths = image_paths
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.output_dir = output_dir
        self.save_intermediate = save_intermediate
        self.model_info = model_info
    
    def run(self):
        """Run batch processing"""
        try:
            processor = BatchProcessor(self.model, self.device)
            
            summary = processor.process_batch(
                self.image_paths,
                self.conf_threshold,
                self.iou_threshold,
                self.output_dir,
                self.save_intermediate,
                progress_callback=self.emit_progress,
                model_info=self.model_info
            )
            
            self.finished.emit(summary)
            
        except Exception as e:
            self.error.emit(f"Batch processing error: {str(e)}")
    
    def emit_progress(self, current: int, total: int, message: str):
        """Emit progress signal"""
        self.progress_updated.emit(current, total, message)


class BatchProcessDialog(QDialog):
    """
    Dialog để batch process multiple images
    (Dialog for batch processing multiple images)
    """
    
    def __init__(self, model, model_info, conf_threshold, iou_threshold, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.model_info = model_info
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self.image_paths = []
        self.processing_thread = None
        
        self.setWindowTitle("Batch Processing")
        self.setGeometry(150, 150, 800, 700)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📁 Batch Image Processing")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2563EB; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # === MODEL SECTION ===
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout(model_group)
        
        self.btn_load_model = QPushButton("🧠 Load Model")
        self.btn_load_model.setMinimumHeight(35)
        self.btn_load_model.clicked.connect(self.load_model)
        model_layout.addWidget(self.btn_load_model)
        
        self.model_info_label = QLabel("No model loaded")
        self.model_info_label.setStyleSheet("padding: 8px; background: #F1F5F9; border-radius: 4px; color: #64748B;")
        model_layout.addWidget(self.model_info_label)
        
        layout.addWidget(model_group)
        
        # === IMAGE SELECTION ===
        selection_group = QGroupBox("Image Selection")
        selection_layout = QVBoxLayout(selection_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_add_images = QPushButton("➕ Add Images")
        self.btn_add_images.setMinimumHeight(35)
        self.btn_add_images.clicked.connect(self.add_images)
        btn_layout.addWidget(self.btn_add_images)
        
        self.btn_add_folder = QPushButton("📁 Add Folder")
        self.btn_add_folder.setMinimumHeight(35)
        self.btn_add_folder.clicked.connect(self.add_folder)
        btn_layout.addWidget(self.btn_add_folder)
        
        self.btn_clear = QPushButton("🗑️ Clear All")
        self.btn_clear.setMinimumHeight(35)
        self.btn_clear.clicked.connect(self.clear_images)
        btn_layout.addWidget(self.btn_clear)
        
        selection_layout.addLayout(btn_layout)
        
        # Image list
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(200)
        selection_layout.addWidget(self.image_list)
        
        self.label_count = QLabel("Images: 0")
        self.label_count.setStyleSheet("font-weight: bold; color: #1E40AF;")
        selection_layout.addWidget(self.label_count)
        
        layout.addWidget(selection_group)
        
        # === OUTPUT SETTINGS ===
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout(output_group)
        
        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output Directory:"))
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        dir_layout.addWidget(self.output_dir_edit, stretch=1)
        
        btn_browse = QPushButton("Browse")
        btn_browse.setMinimumHeight(35)
        btn_browse.clicked.connect(self.browse_output_dir)
        dir_layout.addWidget(btn_browse)
        
        output_layout.addLayout(dir_layout)
        
        # Options
        self.check_save_intermediate = QCheckBox("Save intermediate results (images + JSON for each)")
        self.check_save_intermediate.setChecked(True)
        output_layout.addWidget(self.check_save_intermediate)
        
        layout.addWidget(output_group)
        
        # === PROGRESS ===
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready to process")
        self.progress_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        progress_layout.addWidget(self.log_text)
        
        layout.addWidget(progress_group)
        
        # === ACTION BUTTONS ===
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        self.btn_start = QPushButton("▶ Start Batch Processing")
        self.btn_start.setObjectName("successButton")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setMinimumWidth(200)
        self.btn_start.setMinimumHeight(35)
        action_layout.addWidget(self.btn_start)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.setMinimumHeight(35)
        self.btn_close.clicked.connect(self.accept)
        self.btn_close.setMinimumWidth(100)
        action_layout.addWidget(self.btn_close)
        
        layout.addLayout(action_layout)
    
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
        
        # Load model
        model_loader = ModelLoader()
        success = model_loader.load_model(file_path)
        
        if not success:
            QMessageBox.critical(self, "Error", "Cannot load model!")
            return
        
        # Update model and info
        self.model = model_loader.get_model()
        self.model_info = model_loader.get_model_info()
        
        # Update UI
        self.model_info_label.setText(
            f"✅ {self.model_info['name']} | "
            f"{self.model_info['num_classes']} classes | "
            f"{self.model_info['size_mb']} MB"
        )
        self.model_info_label.setStyleSheet(
            "padding: 8px; background: #DBEAFE; border-radius: 4px; "
            "color: #1E40AF; font-weight: bold;"
        )
        self.log(f"✅ Loaded model: {self.model_info['name']}")
        self._update_ui()
    
    def add_images(self):
        """Add individual images"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All Files (*)"
        )
        
        if file_paths:
            for path in file_paths:
                if path not in self.image_paths:
                    self.image_paths.append(path)
                    self.image_list.addItem(os.path.basename(path))
            
            self._update_ui()
    
    def add_folder(self):
        """Add all images from folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder"
        )
        
        if folder_path:
            # Supported extensions
            extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
            
            # Find all images
            count = 0
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(extensions):
                    file_path = os.path.join(folder_path, file_name)
                    if file_path not in self.image_paths:
                        self.image_paths.append(file_path)
                        self.image_list.addItem(file_name)
                        count += 1
            
            self.log(f"✅ Added {count} images from folder")
            self._update_ui()
    
    def clear_images(self):
        """Clear all images"""
        self.image_paths = []
        self.image_list.clear()
        self._update_ui()
    
    def browse_output_dir(self):
        """Browse output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )
        
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def start_processing(self):
        """Start batch processing"""
        if not self.model:
            QMessageBox.warning(self, "No Model", "Please load a model first!")
            return
        
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please add images first!")
            return
        
        output_dir = self.output_dir_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "No Output Dir", "Please select output directory!")
            return
        
        # Disable controls
        self.btn_start.setEnabled(False)
        self.btn_add_images.setEnabled(False)
        self.btn_add_folder.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_close.setEnabled(False)
        
        # Reset progress
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # Start processing thread
        self.log("🚀 Starting batch processing...")
        
        self.processing_thread = BatchProcessingThread(
            self.model,
            self.image_paths,
            self.conf_threshold,
            self.iou_threshold,
            self.device,
            output_dir,
            self.check_save_intermediate.isChecked(),
            self.model_info
        )
        
        self.processing_thread.progress_updated.connect(self.on_progress_updated)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()
    
    def on_progress_updated(self, current: int, total: int, message: str):
        """Handle progress update"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Processing {current}/{total}: {message}")
        self.log(f"[{current}/{total}] {message}")
    
    def on_processing_finished(self, summary: dict):
        """Handle processing completion"""
        self.progress_bar.setValue(100)
        self.progress_label.setText("✅ Batch processing complete!")
        
        # Log summary
        self.log("\n" + "="*50)
        self.log("📊 BATCH PROCESSING SUMMARY")
        self.log("="*50)
        self.log(f"Total Images: {summary['total_images']}")
        self.log(f"Processed: {summary['processed_images']}")
        self.log(f"Failed: {summary['failed_images']}")
        self.log(f"Success Rate: {summary['success_rate']:.1f}%")
        self.log(f"Total Detections: {summary['total_detections']}")
        self.log(f"Avg Detections/Image: {summary['avg_detections_per_image']:.2f}")
        self.log(f"Total Time: {summary['total_elapsed_time']:.2f}s")
        self.log(f"Avg Time/Image: {summary['avg_time_per_image']:.3f}s")
        self.log("="*50)
        
        # Re-enable controls
        self.btn_close.setEnabled(True)
        
        # Show completion message
        QMessageBox.information(
            self,
            "Complete",
            f"Batch processing complete!\n\n"
            f"Processed: {summary['processed_images']}/{summary['total_images']}\n"
            f"Total Detections: {summary['total_detections']}\n"
            f"Results saved to: {self.output_dir_edit.text()}"
        )
    
    def on_processing_error(self, error_msg: str):
        """Handle processing error"""
        self.log(f"❌ Error: {error_msg}")
        self.progress_label.setText("❌ Processing failed")
        
        # Re-enable controls
        self.btn_start.setEnabled(True)
        self.btn_add_images.setEnabled(True)
        self.btn_add_folder.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_close.setEnabled(True)
        
        QMessageBox.critical(self, "Error", error_msg)
    
    def _update_ui(self):
        """Update UI state"""
        self.label_count.setText(f"Images: {len(self.image_paths)}")
        # Enable start only if model loaded and images added
        self.btn_start.setEnabled(self.model is not None and len(self.image_paths) > 0)
    
    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
