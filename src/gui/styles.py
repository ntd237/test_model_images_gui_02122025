"""
Qt Stylesheet cho YOLO Model Testing Tool
Modern Dark Theme với Blue accent
"""

# === COLOR PALETTE ===
PRIMARY = "#2563EB"
PRIMARY_LIGHT = "#3B82F6"
PRIMARY_DARK = "#1E40AF"

SUCCESS = "#10B981"
ERROR = "#EF4444"
WARNING = "#F59E0B"

BACKGROUND = "#1E293B"
SURFACE = "#334155"
SURFACE_VARIANT = "#475569"
TEXT_PRIMARY = "#F1F5F9"
TEXT_SECONDARY = "#CBD5E1"
BORDER = "#64748B"


def get_stylesheet():
    """
    Trả về Qt stylesheet cho toàn bộ application
    (Returns Qt stylesheet for entire application)
    
    Returns:
        str: Qt stylesheet string
    """
    return f"""
    /* === GLOBAL STYLES === */
    QWidget {{
        background-color: {BACKGROUND};
        color: {TEXT_PRIMARY};
        font-family: "Segoe UI", sans-serif;
        font-size: 12px;
    }}
    
    /* === MAIN WINDOW === */
    QMainWindow {{
        background-color: {BACKGROUND};
    }}
    
    /* === BUTTONS === */
    QPushButton {{
        background-color: {PRIMARY};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 24px;
        font-weight: 600;
        font-size: 12px;
    }}
    
    QPushButton:hover {{
        background-color: {PRIMARY_LIGHT};
    }}
    
    QPushButton:pressed {{
        background-color: {PRIMARY_DARK};
    }}
    
    QPushButton:disabled {{
        background-color: {SURFACE_VARIANT};
        color: {BORDER};
    }}
    
    QPushButton#secondaryButton {{
        background-color: transparent;
        color: {PRIMARY};
        border: 1px solid {PRIMARY};
    }}
    
    QPushButton#secondaryButton:hover {{
        background-color: {PRIMARY};
        color: white;
    }}
    
    QPushButton#dangerButton {{
        background-color: {ERROR};
    }}
    
    QPushButton#dangerButton:hover {{
        background-color: #DC2626;
    }}
    
    QPushButton#successButton {{
        background-color: {SUCCESS};
    }}
    
    QPushButton#successButton:hover {{
        background-color: #059669;
    }}
    
    /* === FRAMES AND GROUP BOXES === */
    QFrame {{
        background-color: {SURFACE};
        border-radius: 6px;
        border: 1px solid {BORDER};
    }}
    
    QGroupBox {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 12px;
        font-weight: 600;
        font-size: 13px;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px;
        color: {TEXT_PRIMARY};
    }}
    
    /* === LABELS === */
    QLabel {{
        background-color: transparent;
        color: {TEXT_PRIMARY};
        border: none;
    }}
    
    QLabel#titleLabel {{
        font-size: 16px;
        font-weight: 700;
        color: {PRIMARY_LIGHT};
    }}
    
    QLabel#subtitleLabel {{
        font-size: 11px;
        color: {TEXT_SECONDARY};
    }}
    
    QLabel#imageLabel {{
        background-color: {BACKGROUND};
        border: 2px dashed {BORDER};
        border-radius: 6px;
    }}
    
    /* === TABLE WIDGET === */
    QTableWidget {{
        background-color: {SURFACE};
        alternate-background-color: {SURFACE_VARIANT};
        gridline-color: {BORDER};
        border: 1px solid {BORDER};
        border-radius: 6px;
    }}
    
    QTableWidget::item {{
        padding: 8px;
        color: {TEXT_PRIMARY};
    }}
    
    QTableWidget::item:selected {{
        background-color: {PRIMARY};
        color: white;
    }}
    
    QHeaderView::section {{
        background-color: {SURFACE_VARIANT};
        color: {TEXT_PRIMARY};
        padding: 8px;
        border: none;
        border-right: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER};
        font-weight: 600;
    }}
    
    /* === TEXT EDIT === */
    QTextEdit {{
        background-color: {SURFACE};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 8px;
    }}
    
    /* === SLIDERS === */
    QSlider::groove:horizontal {{
        background: {SURFACE_VARIANT};
        height: 6px;
        border-radius: 3px;
    }}
    
    QSlider::handle:horizontal {{
        background: {PRIMARY};
        border: 2px solid {PRIMARY_LIGHT};
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background: {PRIMARY_LIGHT};
    }}
    
    QSlider::sub-page:horizontal {{
        background: {PRIMARY};
        border-radius: 3px;
    }}
    
    /* === SCROLL BAR === */
    QScrollBar:vertical {{
        background: {SURFACE};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background: {BORDER};
        border-radius: 6px;
        min-height: 20px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {TEXT_SECONDARY};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    QScrollBar:horizontal {{
        background: {SURFACE};
        height: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:horizontal {{
        background: {BORDER};
        border-radius: 6px;
        min-width: 20px;
    }}
    
    QScrollBar::handle:horizontal:hover {{
        background: {TEXT_SECONDARY};
    }}
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    
    /* === SPLITTER === */
    QSplitter::handle {{
        background-color: {BORDER};
    }}
    
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    
    QSplitter::handle:vertical {{
        height: 2px;
    }}
    
    /* === STATUS BAR === */
    QStatusBar {{
        background-color: {SURFACE};
        color: {TEXT_SECONDARY};
        border-top: 1px solid {BORDER};
    }}
    
    /* === MENU BAR === */
    QMenuBar {{
        background-color: {SURFACE};
        color: {TEXT_PRIMARY};
        border-bottom: 1px solid {BORDER};
    }}
    
    QMenuBar::item:selected {{
        background-color: {PRIMARY};
    }}
    
    QMenu {{
        background-color: {SURFACE};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
    }}
    
    QMenu::item:selected {{
        background-color: {PRIMARY};
    }}
    """
