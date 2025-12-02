"""
Custom Widgets cho YOLO Model Testing Tool
Các widget tùy chỉnh với enhanced functionality
"""

from PyQt5.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QColor


class ImageLabel(QLabel):
    """
    Custom QLabel với zoom và pan support
    (Custom QLabel with zoom and pan support)
    """
    
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("imageLabel")
        self.setMinimumSize(400, 300)
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.setText("Chưa load ảnh\n(No image loaded)")
    
    def setPixmap(self, pixmap: QPixmap):
        """
        Set pixmap và lưu original
        (Set pixmap and save original)
        """
        self.original_pixmap = pixmap
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self._update_display()
    
    def _update_display(self):
        """
        Update display với zoom và pan
        (Update display with zoom and pan)
        """
        if self.original_pixmap is None:
            return
        
        # Scale pixmap to fit label while keeping aspect ratio
        scaled = self.original_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        super().setPixmap(scaled)
    
    def resizeEvent(self, event):
        """Handle resize event"""
        super().resizeEvent(event)
        if self.original_pixmap is not None:
            self._update_display()
    
    def mousePressEvent(self, event):
        """Emit clicked signal"""
        self.clicked.emit()
        super().mousePressEvent(event)
    
    def clear_image(self):
        """
        Clear ảnh hiện tại
        (Clear current image)
        """
        self.original_pixmap = None
        self.clear()
        self.setText("Chưa load ảnh\n(No image loaded)")


class InfoPanel(QFrame):
    """
    Panel hiển thị thông tin (model hoặc image)
    (Panel for displaying information - model or image)
    """
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setup_ui(title)
    
    def setup_ui(self, title: str):
        """Setup UI cho info panel"""
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        # Title
        title_label = QLabel(title)
        title_label.setObjectName("subtitleLabel")
        layout.addWidget(title_label)
        
        # Info text
        self.info_label = QLabel("Chưa có thông tin")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 4px;")
        layout.addWidget(self.info_label)
    
    def set_info(self, info_text: str):
        """
        Set thông tin hiển thị
        (Set information to display)
        """
        self.info_label.setText(info_text)
    
    def clear_info(self):
        """
        Clear thông tin
        (Clear information)
        """
        self.info_label.setText("Chưa có thông tin")


class SliderWithLabel(QWidget):
    """
    QSlider với label hiển thị giá trị
    (QSlider with value label display)
    """
    
    valueChanged = pyqtSignal(float)
    
    def __init__(
        self,
        label_text: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
        default_value: float = 0.5,
        decimals: int = 2,
        parent=None
    ):
        super().__init__(parent)
        self.min_value = min_value
        self.max_value = max_value
        self.decimals = decimals
        self.setup_ui(label_text, default_value)
    
    def setup_ui(self, label_text: str, default_value: float):
        """Setup UI cho slider with label"""
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Top row: Label and value
        top_layout = QHBoxLayout()
        
        self.label = QLabel(label_text)
        top_layout.addWidget(self.label)
        
        top_layout.addStretch()
        
        self.value_label = QLabel()
        self.value_label.setStyleSheet("color: #3B82F6; font-weight: 600;")
        top_layout.addWidget(self.value_label)
        
        layout.addLayout(top_layout)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        
        # Set default value
        default_pos = int((default_value - self.min_value) / 
                         (self.max_value - self.min_value) * 100)
        self.slider.setValue(default_pos)
        
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)
        
        # Update initial value label
        self._on_slider_changed(default_pos)
    
    def _on_slider_changed(self, slider_value: int):
        """
        Handle slider value change
        (Handle slider value change)
        """
        # Convert slider position (0-100) to actual value
        actual_value = (slider_value / 100.0) * (self.max_value - self.min_value) + self.min_value
        
        # Update label
        self.value_label.setText(f"{actual_value:.{self.decimals}f}")
        
        # Emit signal
        self.valueChanged.emit(actual_value)
    
    def value(self) -> float:
        """
        Lấy giá trị hiện tại
        (Get current value)
        """
        slider_value = self.slider.value()
        return (slider_value / 100.0) * (self.max_value - self.min_value) + self.min_value
    
    def setValue(self, value: float):
        """
        Set giá trị
        (Set value)
        """
        # Convert actual value to slider position
        slider_pos = int((value - self.min_value) / 
                        (self.max_value - self.min_value) * 100)
        self.slider.setValue(slider_pos)


class StatusLabel(QLabel):
    """
    Label hiển thị status với icon và color
    (Label for displaying status with icon and color)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setStyleSheet("padding: 8px; border-radius: 4px;")
    
    def set_status(self, message: str, status_type: str = "info"):
        """
        Set status message với type (info, success, error, warning)
        (Set status message with type - info, success, error, warning)
        """
        # Color mapping
        colors = {
            "info": "#3B82F6",
            "success": "#10B981",
            "error": "#EF4444",
            "warning": "#F59E0B"
        }
        
        # Icons
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "error": "❌",
            "warning": "⚠️"
        }
        
        color = colors.get(status_type, colors["info"])
        icon = icons.get(status_type, icons["info"])
        
        self.setText(f"{icon} {message}")
        self.setStyleSheet(f"""
            padding: 8px;
            border-radius: 4px;
            background-color: {color}33;
            border-left: 3px solid {color};
        """)
