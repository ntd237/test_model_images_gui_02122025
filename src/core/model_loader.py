"""
Model Loader cho YOLO models
Hỗ trợ load và validate YOLO models từ Ultralytics
"""

import os
from typing import Optional, Dict, Any
from ultralytics import YOLO


class ModelLoader:
    """Class quản lý việc load và cache YOLO models"""
    
    def __init__(self):
        """Khởi tạo ModelLoader"""
        self.current_model: Optional[YOLO] = None
        self.current_model_path: Optional[str] = None
        self.model_info: Optional[Dict[str, Any]] = None
    
    def load_model(self, model_path: str) -> bool:
        """
        Load YOLO model từ file .pt
        (Load YOLO model from .pt file)
        
        Args:
            model_path: Đường dẫn tới file model .pt
                       (Path to model .pt file)
        
        Returns:
            True nếu load thành công, False nếu thất bại
            (True if load successful, False if failed)
        """
        try:
            # Validate file tồn tại
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"File model không tồn tại: {model_path}")
            
            # Validate extension
            if not model_path.endswith('.pt'):
                raise ValueError("Model phải có định dạng .pt")
            
            # Load model
            print(f"Đang load model: {model_path}")
            model = YOLO(model_path)
            
            # Extract model info
            self.current_model = model
            self.current_model_path = model_path
            self.model_info = self._extract_model_info(model, model_path)
            
            print(f"✅ Load model thành công: {os.path.basename(model_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            self.current_model = None
            self.current_model_path = None
            self.model_info = None
            return False
    
    def _extract_model_info(self, model: YOLO, model_path: str) -> Dict[str, Any]:
        """
        Extract thông tin từ model
        (Extract information from model)
        
        Args:
            model: YOLO model object
            model_path: Đường dẫn tới model file
        
        Returns:
            Dict chứa thông tin model
            (Dict containing model information)
        """
        try:
            # Get model names (classes)
            names = model.names if hasattr(model, 'names') else {}
            
            # Get number of classes
            num_classes = len(names)
            
            # Get model type from filename
            model_name = os.path.basename(model_path)
            
            # Get file size
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            info = {
                'name': model_name,
                'path': model_path,
                'num_classes': num_classes,
                'class_names': names,
                'size_mb': round(file_size_mb, 2),
                'type': 'YOLO'
            }
            
            return info
            
        except Exception as e:
            print(f"Lỗi extract model info: {e}")
            return {
                'name': os.path.basename(model_path),
                'path': model_path,
                'num_classes': 0,
                'class_names': {},
                'size_mb': 0,
                'type': 'YOLO'
            }
    
    def get_model(self) -> Optional[YOLO]:
        """
        Lấy model hiện tại
        (Get current model)
        
        Returns:
            YOLO model object hoặc None
            (YOLO model object or None)
        """
        return self.current_model
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin model hiện tại
        (Get current model information)
        
        Returns:
            Dict chứa thông tin model hoặc None
            (Dict containing model info or None)
        """
        return self.model_info
    
    def is_model_loaded(self) -> bool:
        """
        Kiểm tra model đã được load chưa
        (Check if model is loaded)
        
        Returns:
            True nếu đã load, False nếu chưa
            (True if loaded, False otherwise)
        """
        return self.current_model is not None
    
    def unload_model(self):
        """
        Unload model hiện tại
        (Unload current model)
        """
        self.current_model = None
        self.current_model_path = None
        self.model_info = None
        print("Model đã được unload")
