"""
Model Manager cho YOLO Model Testing Tool
Quản lý multiple models để comparison
"""

from typing import Dict, List, Optional, Any
import os
from src.core.model_loader import ModelLoader


class ModelManager:
    """
    Manager để quản lý multiple YOLO models
    (Manager to handle multiple YOLO models)
    """
    
    def __init__(self, max_models: int = 4):
        """
        Initialize model manager
        
        Args:
            max_models: Maximum number of models to load simultaneously
        """
        self.max_models = max_models
        self.models: Dict[str, Dict[str, Any]] = {}  # model_id -> {loader, info, path}
        self.next_id = 1
    
    def load_model(self, model_path: str) -> Optional[str]:
        """
        Load a new model
        
        Args:
            model_path: Path to model file
        
        Returns:
            Model ID string if successful, None if failed
        """
        # Check max models
        if len(self.models) >= self.max_models:
            print(f"❌ Maximum {self.max_models} models reached!")
            return None
        
        # Check if already loaded
        for model_id, model_data in self.models.items():
            if model_data['path'] == model_path:
                print(f"⚠️ Model already loaded: {model_id}")
                return model_id
        
        # Load model
        loader = ModelLoader()
        success = loader.load_model(model_path)
        
        if not success:
            print(f"❌ Failed to load model: {model_path}")
            return None
        
        # Create model ID
        model_id = f"model_{self.next_id}"
        self.next_id += 1
        
        # Store model
        self.models[model_id] = {
            'loader': loader,
            'model': loader.get_model(),
            'info': loader.get_model_info(),
            'path': model_path,
            'name': os.path.basename(model_path)
        }
        
        print(f"✅ Loaded model {model_id}: {os.path.basename(model_path)}")
        return model_id
    
    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from manager
        
        Args:
            model_id: Model ID to remove
        
        Returns:
            True if successful, False if model not found
        """
        if model_id in self.models:
            del self.models[model_id]
            print(f"✅ Removed model: {model_id}")
            return True
        
        print(f"❌ Model not found: {model_id}")
        return False
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """
        Get model by ID
        
        Args:
            model_id: Model ID
        
        Returns:
            Model object or None if not found
        """
        if model_id in self.models:
            return self.models[model_id]['model']
        return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model info by ID
        
        Args:
            model_id: Model ID
        
        Returns:
            Model info dict or None if not found
        """
        if model_id in self.models:
            return self.models[model_id]['info']
        return None
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all loaded models
        
        Returns:
            Dict of model_id -> model data
        """
        return self.models.copy()
    
    def get_model_count(self) -> int:
        """
        Get number of loaded models
        
        Returns:
            Count of loaded models
        """
        return len(self.models)
    
    def clear_all(self):
        """Clear all models"""
        self.models.clear()
        self.next_id = 1
        print("✅ Cleared all models")
    
    def get_model_list(self) -> List[Dict[str, str]]:
        """
        Get list of models for UI display
        
        Returns:
            List of dicts with id, name, path
        """
        return [
            {
                'id': model_id,
                'name': data['name'],
                'path': data['path'],
                'num_classes': data['info'].get('num_classes', 'Unknown')
            }
            for model_id, data in self.models.items()
        ]
