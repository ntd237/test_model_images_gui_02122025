"""
Utility functions cho image processing
Xử lý conversion giữa OpenCV, PIL, và Qt formats
"""

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from typing import Optional, Tuple


def load_image_cv(image_path: str) -> Optional[np.ndarray]:
    """
    Load ảnh từ file path sử dụng OpenCV
    (Load image from file path using OpenCV)
    
    Args:
        image_path: Đường dẫn tới file ảnh (Path to image file)
        
    Returns:
        np.ndarray nếu thành công, None nếu thất bại
        (np.ndarray if successful, None if failed)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể load ảnh: {image_path}")
        return img
    except Exception as e:
        print(f"Lỗi load ảnh: {e}")
        return None


def cv_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """
    Convert OpenCV image (BGR) sang QPixmap (RGB)
    (Convert OpenCV image (BGR) to QPixmap (RGB))
    
    Args:
        cv_img: OpenCV image (numpy array, BGR format)
        
    Returns:
        QPixmap object
    """
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    height, width, channel = rgb_img.shape
    bytes_per_line = 3 * width
    
    q_img = QImage(
        rgb_img.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888
    )
    
    return QPixmap.fromImage(q_img)


def resize_image_keep_aspect(
    image: np.ndarray,
    target_width: int,
    target_height: int
) -> np.ndarray:
    """
    Resize ảnh giữ nguyên aspect ratio
    (Resize image while keeping aspect ratio)
    
    Args:
        image: OpenCV image
        target_width: Chiều rộng mục tiêu (Target width)
        target_height: Chiều cao mục tiêu (Target height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Tính aspect ratio
    aspect = w / h
    target_aspect = target_width / target_height
    
    if aspect > target_aspect:
        # Image rộng hơn, fit theo width
        new_w = target_width
        new_h = int(target_width / aspect)
    else:
        # Image cao hơn, fit theo height
        new_h = target_height
        new_w = int(target_height * aspect)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def qpixmap_to_cv(pixmap: QPixmap) -> np.ndarray:
    """
    Convert QPixmap sang OpenCV image (BGR)
    (Convert QPixmap to OpenCV image (BGR))
    
    Args:
        pixmap: QPixmap object
        
    Returns:
        OpenCV image (numpy array, BGR format)
    """
    # Convert QPixmap to QImage
    qimg = pixmap.toImage()
    
    # Convert QImage to numpy array
    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  # RGBA
    
    # Convert RGBA to BGR
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return bgr


def validate_image_format(image_path: str) -> bool:
    """
    Validate image format có supported không
    (Validate if image format is supported)
    
    Args:
        image_path: Đường dẫn tới file ảnh (Path to image file)
        
    Returns:
        True nếu format hợp lệ, False nếu không
        (True if format is valid, False otherwise)
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    import os
    ext = os.path.splitext(image_path)[1].lower()
    return ext in supported_formats


def get_image_info(image_path: str) -> Optional[dict]:
    """
    Lấy thông tin cơ bản về ảnh
    (Get basic information about image)
    
    Args:
        image_path: Đường dẫn tới file ảnh (Path to image file)
        
    Returns:
        Dict chứa thông tin ảnh hoặc None nếu lỗi
        (Dict containing image info or None if error)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        import os
        file_size = os.path.getsize(image_path)
        
        return {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': img.shape[2] if len(img.shape) == 3 else 1,
            'size_kb': file_size / 1024,
            'format': os.path.splitext(image_path)[1].upper()[1:]
        }
    except Exception as e:
        print(f"Lỗi lấy image info: {e}")
        return None
