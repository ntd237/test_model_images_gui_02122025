"""
Batch Processor cho YOLO Model Testing Tool
Process multiple images và generate batch reports
"""

import os
from typing import List, Dict, Any, Callable, Optional
import cv2
import time
from datetime import datetime

from src.core.inference import InferenceEngine
from src.utils.export_utils import export_to_json, export_to_csv, generate_pdf_report


class BatchProcessor:
    """
    Batch processor cho multiple images
    (Batch processor for multiple images)
    """
    
    def __init__(self, model, device: str = 'cpu'):
        """
        Initialize batch processor
        
        Args:
            model: YOLO model instance
            device: Device to use ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.engine = InferenceEngine()
        
        # Results storage
        self.results = []
        self.failed_images = []
        
        # Statistics
        self.total_images = 0
        self.processed_images = 0
        self.total_detections = 0
        self.total_time = 0.0
    
    def process_batch(
        self,
        image_paths: List[str],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        output_dir: Optional[str] = None,
        save_intermediate: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        model_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process batch of images
        
        Args:
            image_paths: List of image file paths
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold
            output_dir: Output directory for results (optional)
            save_intermediate: Save intermediate results per image
            progress_callback: Callback(current, total, status_msg)
            model_info: Model information dict
        
        Returns:
            Batch summary dict
        """
        self.total_images = len(image_paths)
        self.processed_images = 0
        self.results = []
        self.failed_images = []
        self.total_detections = 0
        self.total_time = 0.0
        
        start_time = time.time()
        
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process each image
        for idx, image_path in enumerate(image_paths):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(
                        idx + 1,
                        self.total_images,
                        f"Processing: {os.path.basename(image_path)}"
                    )
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Cannot load image: {image_path}")
                
                # Run inference
                result = self.engine.run_inference(
                    self.model,
                    image,
                    conf_threshold,
                    iou_threshold,
                    self.device
                )
                
                if result is None:
                    raise ValueError("Inference failed")
                
                # Store result
                result['image_path'] = image_path
                result['image_name'] = os.path.basename(image_path)
                self.results.append(result)
                
                # Update statistics
                self.processed_images += 1
                self.total_detections += result['num_detections']
                self.total_time += result['inference_time']
                
                # Save intermediate results
                if save_intermediate and output_dir:
                    self._save_intermediate_result(
                        result,
                        output_dir,
                        model_info
                    )
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                self.failed_images.append({
                    'path': image_path,
                    'error': str(e)
                })
        
        # Calculate total processing time
        total_elapsed = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(total_elapsed)
        
        # Save summary report if output dir specified
        if output_dir:
            self._save_summary_report(summary, output_dir, model_info)
        
        return summary
    
    def _save_intermediate_result(
        self,
        result: Dict[str, Any],
        output_dir: str,
        model_info: Optional[Dict[str, Any]]
    ):
        """
        Save intermediate result for single image
        
        Args:
            result: Inference result dict
            output_dir: Output directory
            model_info: Model information
        """
        try:
            base_name = os.path.splitext(result['image_name'])[0]
            
            # Save annotated image
            img_output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
            cv2.imwrite(img_output_path, result['annotated_image'])
            
            # Save JSON
            json_output_path = os.path.join(output_dir, f"{base_name}_result.json")
            export_to_json(
                result['detections'],
                result['image_path'],
                model_info or {},
                json_output_path,
                result['inference_time']
            )
            
        except Exception as e:
            print(f"Warning: Failed to save intermediate result: {e}")
    
    def _generate_summary(self, total_elapsed: float) -> Dict[str, Any]:
        """
        Generate batch processing summary
        
        Args:
            total_elapsed: Total elapsed time in seconds
        
        Returns:
            Summary dict
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': self.total_images,
            'processed_images': self.processed_images,
            'failed_images': len(self.failed_images),
            'success_rate': (self.processed_images / self.total_images * 100) if self.total_images > 0 else 0,
            'total_detections': self.total_detections,
            'avg_detections_per_image': self.total_detections / self.processed_images if self.processed_images > 0 else 0,
            'total_inference_time': self.total_time,
            'total_elapsed_time': total_elapsed,
            'avg_time_per_image': self.total_time / self.processed_images if self.processed_images > 0 else 0,
            'results': self.results,
            'failed': self.failed_images
        }
        
        return summary
    
    def _save_summary_report(
        self,
        summary: Dict[str, Any],
        output_dir: str,
        model_info: Optional[Dict[str, Any]]
    ):
        """
        Save batch summary report
        
        Args:
            summary: Summary dict
            output_dir: Output directory
            model_info: Model information
        """
        try:
            # Save summary JSON
            summary_json_path = os.path.join(output_dir, "batch_summary.json")
            
            import json
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                # Create clean summary without large image arrays
                clean_summary = {
                    'timestamp': summary['timestamp'],
                    'statistics': {
                        'total_images': summary['total_images'],
                        'processed_images': summary['processed_images'],
                        'failed_images': summary['failed_images'],
                        'success_rate': summary['success_rate'],
                        'total_detections': summary['total_detections'],
                        'avg_detections_per_image': summary['avg_detections_per_image'],
                        'total_inference_time': summary['total_inference_time'],
                        'total_elapsed_time': summary['total_elapsed_time'],
                        'avg_time_per_image': summary['avg_time_per_image']
                    },
                    'results': [
                        {
                            'image_name': r['image_name'],
                            'num_detections': r['num_detections'],
                            'inference_time': r['inference_time'],
                            'detections': r['detections']
                        }
                        for r in summary['results']
                    ],
                    'failed': summary['failed']
                }
                
                json.dump(clean_summary, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved batch summary: {summary_json_path}")
            
            # Save CSV summary
            csv_path = os.path.join(output_dir, "batch_summary.csv")
            
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Image', 'Detections', 'Inference Time (s)', 'Status'])
                
                for r in summary['results']:
                    writer.writerow([
                        r['image_name'],
                        r['num_detections'],
                        f"{r['inference_time']:.3f}",
                        'Success'
                    ])
                
                for f_img in summary['failed']:
                    writer.writerow([
                        os.path.basename(f_img['path']),
                        0,
                        0,
                        f"Failed: {f_img['error']}"
                    ])
            
            print(f"✅ Saved batch CSV: {csv_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save summary report: {e}")
