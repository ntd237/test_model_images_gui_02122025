# YOLO Model Testing Tool

> Tool GUI chuyên nghiệp để test các YOLO models với image inference

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## 📝 Mô Tả

YOLO Model Testing Tool là công cụ GUI được xây dựng với PyQt5, cho phép người dùng dễ dàng test các YOLO models (YOLOv8, YOLOv11, etc.) đã được finetune. Tool cung cấp giao diện trực quan để:

- ✅ Load và hiển thị ảnh test
- ✅ Load YOLO models (định dạng .pt)
- ✅ Chạy inference với confidence/IOU thresholds tùy chỉnh
- ✅ Hiển thị kết quả side-by-side (ảnh gốc vs ảnh detected)
- ✅ Xem chi tiết detections trong bảng
- ✅ Lưu kết quả ảnh đã annotate

## 🎨 Giao Diện

Tool sử dụng **Modern Dark Theme** với layout Master-Detail:

- **Left Panel**: Control panel với buttons và settings
- **Center-Right Panel**: Split view hiển thị ảnh gốc và kết quả
- **Bottom Panel**: Bảng detections và log panel

## 🚀 Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- Windows / Linux / macOS
- (Optional) CUDA nếu muốn chạy inference trên GPU

### Các Bước Cài Đặt

1. **Clone hoặc download project**

2. **Cài đặt dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Chuẩn bị models và images**:
   - Đặt YOLO models (.pt files) vào folder `resources/models/`
   - Đặt ảnh test vào folder `resources/sample_images/` (optional)

## 📖 Hướng Dẫn Sử Dụng

### Khởi Chạy Tool

```bash
python main.py
```

### Workflow Cơ Bản

1. **Load Ảnh**:
   - Click nút "📁 Load Ảnh"
   - Chọn ảnh từ file system
   - Ảnh sẽ hiển thị trong panel "Ảnh Gốc"

2. **Load Model**:
   - Click nút "🧠 Load Model"
   - Chọn file model .pt (mặc định tìm trong `resources/models/`)
   - Thông tin model sẽ hiển thị

3. **Cấu Hình Settings**:
   - Điều chỉnh **Confidence Threshold** (default: 0.25)
   - Điều chỉnh **IOU Threshold** (default: 0.45)

4. **Chạy Inference**:
   - Click nút "▶ Chạy Inference"
   - Kết quả sẽ hiển thị trong panel "Kết Quả Inference"
   - Xem chi tiết detections trong bảng bên dưới

5. **Lưu Kết Quả** (Optional):
   - Click nút "💾 Lưu Kết Quả"
   - Chọn nơi lưu và format (JPG/PNG)

## 📁 Cấu Trúc Dự Án

```
test_model_images_web_02122025/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── README.md                  # Documentation (file này)
│
├── resources/                 # Resources folder
│   ├── models/               # YOLO models (.pt files)
│   ├── sample_images/        # Sample test images
│   └── icons/                # UI icons
│
└── src/                      # Source code
    ├── __init__.py
    │
    ├── gui/                  # GUI components
    │   ├── __init__.py
    │   ├── main_window.py    # Main window implementation
    │   ├── widgets.py        # Custom widgets
    │   └── styles.py         # Qt stylesheets (dark theme)
    │
    ├── core/                 # Core logic
    │   ├── __init__.py
    │   ├── model_loader.py   # YOLO model loading
    │   └── inference.py      # Inference engine
    │
    └── utils/                # Utilities
        ├── __init__.py
        └── image_utils.py    # Image processing utilities
```

## 🔧 Dependencies

- **PyQt5**: GUI framework
- **ultralytics**: YOLO models support
- **opencv-python**: Image processing
- **numpy**: Numerical operations
- **Pillow**: Image I/O

Xem chi tiết trong `requirements.txt`.

## 🎯 Tính Năng Nổi Bật

### 1. Giao Diện Hiện Đại
- Dark theme chuyên nghiệp
- Layout trực quan, dễ sử dụng
- Responsive design

### 2. Inference Threading
- Inference chạy trên thread riêng
- UI không bị block trong khi xử lý
- Real-time progress feedback

### 3. Flexible Configuration
- Điều chỉnh confidence threshold (0.0 - 1.0)
- Điều chỉnh IOU threshold (0.0 - 1.0)
- Sliders với real-time value display

### 4. Detailed Results
- Bảng detections với thông tin đầy đủ
- Class name, confidence score, bounding box coordinates
- Sortable table

### 5. Model Support
- Hỗ trợ tất cả YOLO models từ Ultralytics
- YOLOv8n/s/m/l/x
- YOLOv11n/s/m/l/x
- Custom finetuned models

## 🐛 Troubleshooting

### Lỗi "No module named 'PyQt5'"
```bash
pip install PyQt5==5.15.10
```

### Lỗi "No module named 'ultralytics'"
```bash
pip install ultralytics>=8.0.0
```

### Model không load được
- Kiểm tra file model có định dạng .pt
- Đảm bảo model được train với Ultralytics YOLO
- Kiểm tra model file không bị corrupt

### Ảnh không hiển thị
- Kiểm tra format ảnh (hỗ trợ: jpg, jpeg, png, bmp, tiff, webp)
- Kiểm tra file ảnh không bị corrupt
- Kiểm tra đường dẫn file

## 📝 Notes

- Tool mặc định chạy inference trên **CPU**
- Nếu có CUDA, có thể modify `device='cuda'` trong `main_window.py`
- Inference time phụ thuộc vào:
  - Kích thước ảnh
  - Model size (n/s/m/l/x)
  - Hardware (CPU vs GPU)

## 👤 Author

**ntd237**
- Email: ntd237.work@gmail.com
- GitHub: [@ntd237](https://github.com/ntd237)

## 📄 License

MIT License - Free to use and modify

---

**Enjoy testing your YOLO models! 🚀**
