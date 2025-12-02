"""
YOLO Model Testing Tool - Main Entry Point
Tool GUI để test YOLO models với image inference

Author: ntd237
Email: ntd237.work@gmail.com
"""

import sys
from PyQt5.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.gui.styles import get_stylesheet


def main():
    """
    Entry point chính của application
    (Main entry point of application)
    """
    # Create application
    app = QApplication(sys.argv)
    
    # Apply stylesheet
    app.setStyleSheet(get_stylesheet())
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
