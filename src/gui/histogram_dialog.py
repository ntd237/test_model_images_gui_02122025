"""
Histogram Dialog cho YOLO Model Testing Tool
Popup window hiển thị confidence distribution histogram
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from typing import List, Dict, Any


class HistogramDialog(QDialog):
    """
    Dialog hiển thị confidence histogram
    (Dialog displaying confidence histogram)
    """
    
    def __init__(self, detections: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.detections = detections
        self.setWindowTitle("Confidence Distribution Histogram")
        self.setGeometry(200, 200, 900, 750)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI cho histogram dialog"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📊 Confidence Distribution Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2563EB; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(11, 7))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Statistics label
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet(
            "padding: 10px; "
            "background-color: #F1F5F9; "
            "color: #1E293B; "
            "border-radius: 6px; "
            "font-size: 13px;"
        )
        layout.addWidget(self.stats_label)
        
        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setMinimumWidth(100)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        # Plot histogram
        self.plot_histogram()
    
    def plot_histogram(self):
        """
        Vẽ confidence histogram
        (Plot confidence histogram)
        """
        # Extract confidences
        confidences = [det['confidence'] for det in self.detections]
        
        if not confidences:
            return
        
        # Calculate statistics
        mean_conf = np.mean(confidences)
        median_conf = np.median(confidences)
        std_conf = np.std(confidences)
        min_conf = np.min(confidences)
        max_conf = np.max(confidences)
        
        # Create plot
        ax = self.figure.add_subplot(111)
        
        # Plot histogram
        n, bins, patches = ax.hist(
            confidences,
            bins=20,
            color='#2563EB',
            alpha=0.7,
            edgecolor='black',
            linewidth=1.2
        )
        
        # Add mean and median lines
        ax.axvline(
            mean_conf,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {mean_conf:.3f}'
        )
        ax.axvline(
            median_conf,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Median: {median_conf:.3f}'
        )
        
        # Labels and title
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Detection Confidence Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        
        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Update statistics label
        stats_text = (
            f"📈 <b>Statistics Summary:</b><br>"
            f"• Total Detections: {len(confidences)}<br>"
            f"• Mean Confidence: {mean_conf:.4f}<br>"
            f"• Median Confidence: {median_conf:.4f}<br>"
            f"• Std Deviation: {std_conf:.4f}<br>"
            f"• Min Confidence: {min_conf:.4f}<br>"
            f"• Max Confidence: {max_conf:.4f}"
        )
        self.stats_label.setText(stats_text)
