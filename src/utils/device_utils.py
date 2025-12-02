"""
Device Utilities cho YOLO Model Testing Tool
Auto-detect và manage CUDA/CPU devices
"""

import torch
from typing import List, Dict, Any, Optional


def detect_available_devices() -> List[Dict[str, Any]]:
    """
    Auto-detect các devices available (CPU và CUDA)
    (Auto-detect available devices - CPU and CUDA)
    
    Returns:
        List of device dicts với thông tin:
        - name: Device name
        - id: Device ID (string: 'cpu', 'cuda:0', 'cuda:1', etc.)
        - type: 'CPU' hoặc 'CUDA'
        - memory_gb: Total memory (GB) - chỉ cho CUDA
        - compute_capability: Compute capability - chỉ cho CUDA
    """
    devices = []
    
    # Always add CPU
    devices.append({
        'name': 'CPU',
        'id': 'cpu',
        'type': 'CPU',
        'memory_gb': None,
        'compute_capability': None
    })
    
    # Check CUDA availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            
            device_info = {
                'name': f'{props.name} (GPU {i})',
                'id': f'cuda:{i}' if i > 0 else 'cuda',
                'type': 'CUDA',
                'memory_gb': round(props.total_memory / (1024**3), 2),
                'compute_capability': f'{props.major}.{props.minor}'
            }
            devices.append(device_info)
    
    return devices


def is_cuda_available() -> bool:
    """
    Check CUDA có available không
    (Check if CUDA is available)
    
    Returns:
        True nếu CUDA available, False nếu không
    """
    return torch.cuda.is_available()


def get_device_info(device_id: str) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin chi tiết về device
    (Get detailed information about device)
    
    Args:
        device_id: Device ID ('cpu', 'cuda', 'cuda:0', etc.)
    
    Returns:
        Dict chứa device info hoặc None nếu invalid device
    """
    devices = detect_available_devices()
    
    for device in devices:
        if device['id'] == device_id:
            return device
    
    return None


def get_default_device() -> str:
    """
    Lấy default device (CUDA nếu available, nếu không thì CPU)
    (Get default device - CUDA if available, otherwise CPU)
    
    Returns:
        Device ID string
    """
    if is_cuda_available():
        return 'cuda'
    return 'cpu'


def format_device_display_name(device_info: Dict[str, Any]) -> str:
    """
    Format device name để hiển thị trong UI
    (Format device name for UI display)
    
    Args:
        device_info: Device info dict
    
    Returns:
        Formatted display string
    """
    if device_info['type'] == 'CPU':
        return "💻 CPU"
    else:
        # CUDA device
        name = device_info['name']
        memory = device_info['memory_gb']
        return f"🚀 {name} ({memory} GB)"


def validate_device(device_id: str) -> bool:
    """
    Validate device ID có hợp lệ không
    (Validate if device ID is valid)
    
    Args:
        device_id: Device ID to validate
    
    Returns:
        True nếu valid, False nếu không
    """
    devices = detect_available_devices()
    device_ids = [d['id'] for d in devices]
    return device_id in device_ids
