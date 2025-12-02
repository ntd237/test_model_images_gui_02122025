"""
Inference Engine cho YOLO models
Chạy inference và trả về kết quả với annotated image
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, List, Any
from ultralytics import YOLO


class InferenceEngine:
    """Class thực hiện inference với YOLO models"""
    
    def __init__(self):
        """Khởi tạo InferenceEngine"""
        self.last_inference_time = 0.0
    
    def run_inference(
        self,
        model: YOLO,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cpu'
    ) -> Optional[Dict[str, Any]]:
        """
        Chạy inference với YOLO model
        (Run inference with YOLO model)
        
        Args:
            model: YOLO model object
            image: OpenCV image (BGR format)
            conf_threshold: Confidence threshold (0.0-1.0)
            iou_threshold: IOU threshold cho NMS (0.0-1.0)
            device: Device để chạy inference ('cpu' hoặc 'cuda')
        
        Returns:
            Dict chứa kết quả inference hoặc None nếu lỗi:
            {
                'detections': List of detections,
                'annotated_image': np.ndarray,
                'inference_time': float (seconds),
                'num_detections': int
            }
        """
        try:
            # Start timer
            start_time = time.time()
            
            # Run inference
            results = model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                verbose=False
            )
            
            # Get first result (single image)
            result = results[0]
            
            # Extract detections
            detections = self._extract_detections(result)
            
            # Get annotated image
            annotated_image = result.plot()
            
            # Calculate inference time
            inference_time = time.time() - start_time
            self.last_inference_time = inference_time
            
            return {
                'detections': detections,
                'annotated_image': annotated_image,
                'inference_time': inference_time,
                'num_detections': len(detections)
            }
            
        except Exception as e:
            print(f"❌ Lỗi inference: {e}")
            return None
    
    def _extract_detections(self, result) -> List[Dict[str, Any]]:
        """
        Extract detections từ YOLO result object
        (Extract detections from YOLO result object)
        
        Args:
            result: YOLO result object
        
        Returns:
            List of detection dicts
        """
        detections = []
        
        try:
            # Get boxes
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                return detections
            
            # Extract thông tin từ mỗi box
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Get bounding box coordinates (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Get confidence
                conf = float(box.conf[0].cpu().numpy())
                
                # Get class
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names[cls_id] if cls_id in result.names else f"Class_{cls_id}"
                
                detection = {
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': round(conf, 3),
                    'bbox': [
                        int(x1), int(y1),
                        int(x2), int(y2)
                    ]
                }
                
                detections.append(detection)
            
        except Exception as e:
            print(f"Lỗi extract detections: {e}")
        
        return detections
    
    def get_last_inference_time(self) -> float:
        """
        Lấy thời gian inference lần cuối
        (Get last inference time)
        
        Returns:
            Inference time in seconds
        """
        return self.last_inference_time
    
    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        colors: Optional[Dict[int, tuple]] = None
    ) -> np.ndarray:
        """
        Vẽ bounding boxes và labels lên ảnh (custom drawing)
        (Draw bounding boxes and labels on image - custom drawing)
        
        Args:
            image: OpenCV image
            detections: List of detection dicts
            colors: Dict mapping class_id to (B,G,R) color
        
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Default colors nếu không cung cấp
        if colors is None:
            colors = {}
        
        for det in detections:
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            # Get color cho class này
            if class_id in colors:
                color = colors[class_id]
            else:
                # Random color based on class_id
                np.random.seed(class_id)
                color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        return annotated
