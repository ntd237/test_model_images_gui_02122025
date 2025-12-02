"""
Export Utilities cho YOLO Model Testing Tool
Export detection results sang JSON, CSV, và PDF formats
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Charts generation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def export_to_json(
    detections: List[Dict[str, Any]],
    image_path: str,
    model_info: Dict[str, Any],
    output_path: str,
    inference_time: float = 0.0
) -> bool:
    """
    Export detection results sang JSON format
    (Export detection results to JSON format)
    
    Args:
        detections: List of detection dicts
        image_path: Path to source image
        model_info: Model information dict
        output_path: Output JSON file path
        inference_time: Inference time in seconds
    
    Returns:
        True nếu export thành công, False nếu thất bại
    """
    try:
        # Prepare export data
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'tool': 'YOLO Model Testing Tool',
                'version': '1.0.0'
            },
            'image': {
                'path': image_path,
                'filename': os.path.basename(image_path)
            },
            'model': model_info,
            'inference': {
                'time_seconds': round(inference_time, 4),
                'num_detections': len(detections)
            },
            'detections': detections
        }
        
        # Write to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported to JSON: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting to JSON: {e}")
        return False


def export_to_csv(
    detections: List[Dict[str, Any]],
    output_path: str
) -> bool:
    """
    Export detections table sang CSV format
    (Export detections table to CSV format)
    
    Args:
        detections: List of detection dicts
        output_path: Output CSV file path
    
    Returns:
        True nếu export thành công, False nếu thất bại
    """
    try:
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
            
            # Data rows
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                writer.writerow([
                    det['class_name'],
                    det['confidence'],
                    x1, y1, x2, y2
                ])
        
        print(f"✅ Exported to CSV: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting to CSV: {e}")
        return False


def generate_pdf_report(
    detections: List[Dict[str, Any]],
    original_image_path: str,
    annotated_image: np.ndarray,
    model_info: Dict[str, Any],
    output_path: str,
    inference_time: float = 0.0,
    device: str = 'cpu'
) -> bool:
    """
    Generate comprehensive PDF report với images và charts
    (Generate comprehensive PDF report with images and charts)
    
    Args:
        detections: List of detection dicts
        original_image_path: Path to original image
        annotated_image: Annotated image as numpy array (BGR)
        model_info: Model information dict
        output_path: Output PDF file path
        inference_time: Inference time in seconds
        device: Device used for inference
    
    Returns:
        True nếu export thành công, False nếu thất bại
    """
    try:
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2563EB'),
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1E40AF'),
            spaceAfter=12
        )
        
        # === COVER PAGE ===
        story.append(Paragraph("YOLO Detection Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Metadata table
        metadata = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Input Image:', os.path.basename(original_image_path)],
            ['Model:', model_info.get('name', 'Unknown')],
            ['Device:', device.upper()],
            ['Inference Time:', f"{inference_time:.3f}s"],
            ['Total Detections:', str(len(detections))]
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F1F5F9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(metadata_table)
        story.append(PageBreak())
        
        # === DETECTION RESULTS ===
        story.append(Paragraph("Detection Results", heading_style))
        story.append(Spacer(1, 0.2*inch))
        
        if detections:
            # Detections table
            table_data = [['#', 'Class', 'Confidence', 'Bounding Box']]
            for i, det in enumerate(detections, 1):
                x1, y1, x2, y2 = det['bbox']
                bbox_str = f"({x1},{y1}) - ({x2},{y2})"
                table_data.append([
                    str(i),
                    det['class_name'],
                    f"{det['confidence']:.3f}",
                    bbox_str
                ])
            
            det_table = Table(table_data, colWidths=[0.5*inch, 2*inch, 1.2*inch, 2.3*inch])
            det_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            story.append(det_table)
        else:
            story.append(Paragraph("No detections found.", styles['Normal']))
        
        story.append(PageBreak())
        
        # === IMAGES ===
        story.append(Paragraph("Visual Results", heading_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Save annotated image temporarily
        temp_annotated_path = output_path.replace('.pdf', '_temp_annotated.jpg')
        cv2.imwrite(temp_annotated_path, annotated_image)
        
        # Add images
        try:
            # Original image
            story.append(Paragraph("Original Image:", styles['Heading3']))
            orig_img = RLImage(original_image_path, width=5*inch, height=3.5*inch)
            story.append(orig_img)
            story.append(Spacer(1, 0.2*inch))
            
            # Annotated image
            story.append(Paragraph("Annotated Result:", styles['Heading3']))
            annot_img = RLImage(temp_annotated_path, width=5*inch, height=3.5*inch)
            story.append(annot_img)
        except Exception as e:
            print(f"Warning: Could not add images to PDF: {e}")
        
        story.append(PageBreak())
        
        # === CHARTS ===
        if detections:
            story.append(Paragraph("Statistical Analysis", heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Generate confidence histogram
            conf_chart_path = _generate_confidence_histogram(detections, output_path.replace('.pdf', '_conf_hist.png'))
            if conf_chart_path:
                story.append(Paragraph("Confidence Distribution:", styles['Heading3']))
                conf_img = RLImage(conf_chart_path, width=5*inch, height=3*inch)
                story.append(conf_img)
                story.append(Spacer(1, 0.2*inch))
            
            # Generate class distribution pie chart
            class_chart_path = _generate_class_distribution(detections, output_path.replace('.pdf', '_class_pie.png'))
            if class_chart_path:
                story.append(Paragraph("Class Distribution:", styles['Heading3']))
                class_img = RLImage(class_chart_path, width=5*inch, height=3*inch)
                story.append(class_img)
        
        # Build PDF
        doc.build(story)
        
        # Cleanup temp files
        try:
            if os.path.exists(temp_annotated_path):
                os.remove(temp_annotated_path)
        except:
            pass
        
        print(f"✅ Generated PDF report: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
        import traceback
        traceback.print_exc()
        return False


def _generate_confidence_histogram(detections: List[Dict], output_path: str) -> Optional[str]:
    """
    Generate confidence distribution histogram
    (Generate confidence distribution histogram)
    """
    try:
        confidences = [det['confidence'] for det in detections]
        
        plt.figure(figsize=(8, 5))
        plt.hist(confidences, bins=20, color='#2563EB', alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_conf = np.mean(confidences)
        median_conf = np.median(confidences)
        plt.axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.3f}')
        plt.axvline(median_conf, color='green', linestyle='--', linewidth=2, label=f'Median: {median_conf:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    except Exception as e:
        print(f"Error generating confidence histogram: {e}")
        return None


def _generate_class_distribution(detections: List[Dict], output_path: str) -> Optional[str]:
    """
    Generate class distribution pie chart
    (Generate class distribution pie chart)
    """
    try:
        # Count classes
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Plot
        plt.figure(figsize=(8, 6))
        colors_pie = plt.cm.Set3(range(len(class_counts)))
        plt.pie(
            class_counts.values(),
            labels=class_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_pie
        )
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    except Exception as e:
        print(f"Error generating class distribution chart: {e}")
        return None
