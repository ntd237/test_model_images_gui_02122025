# YOLO Model Testing Tool

> **Công cụ GUI chuyên nghiệp để kiểm thử và đánh giá model YOLO với tính năng xử lý hàng loạt và báo cáo chi tiết.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-green)](https://github.com/ultralytics/ultralytics)

---

## 📚 Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Tính Năng](#tính-năng)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Liên Hệ](#liên-hệ)

---

## 🎯 Giới Thiệu

### Vấn Đề
Việc kiểm thử model YOLO sau khi training thường gặp nhiều khó khăn:
- ❌ **Thủ công**: Phải chạy từng ảnh hoặc script dòng lệnh phức tạp.
- ❌ **Khó so sánh**: Không có giao diện trực quan để so sánh ảnh gốc và kết quả.
- ❌ **Thiếu báo cáo**: Khó tổng hợp kết quả thống kê cho hàng trăm ảnh.

### Giải Pháp
**YOLO Model Testing Tool** cung cấp giải pháp toàn diện:
- ✅ **GUI Trực quan**: Giao diện hiện đại, dễ sử dụng với Dark Theme.
- ✅ **Batch Processing**: Xử lý hàng loạt thư mục ảnh với tốc độ cao.
- ✅ **Báo cáo Tự động**: Xuất báo cáo PDF chuyên nghiệp với biểu đồ thống kê.

### Công Nghệ
Dự án được xây dựng với:
- **Python 3.10+**: Ngôn ngữ chính.
- **PyQt5**: Framework GUI mạnh mẽ.
- **Ultralytics YOLO**: Engine nhận diện đối tượng state-of-the-art.
- **ReportLab & Matplotlib**: Tạo báo cáo và biểu đồ.

---

## ✨ Tính Năng

### Core Features
- 🎯 **Single Image Inference**: Test nhanh từng ảnh, điều chỉnh threshold realtime.
- 📁 **Batch Processing**: 
  - Xử lý toàn bộ thư mục ảnh.
  - Thanh tiến trình (Progress bar) và Log chi tiết.
  - Hỗ trợ tạm dừng/tiếp tục.
- ⚡ **GPU Support**: Tự động phát hiện và cho phép chọn thiết bị (CPU/CUDA).

### Advanced Features
- 📊 **Advanced Visualization**:
  - Biểu đồ phân phối độ tin cậy (Confidence Distribution).
  - Biểu đồ phân phối lớp (Class Distribution - Pie Chart).
  - Click vào bảng kết quả để highlight bounding box trên ảnh.
- 💾 **Export Options**:
  - **PDF Report**: Báo cáo đầy đủ với biểu đồ và hình ảnh minh họa.
  - **CSV/JSON**: Xuất dữ liệu thô để phân tích thêm.
  - **Save Images**: Lưu ảnh kết quả hàng loạt.
- 🔍 **Class Filtering**: Lọc kết quả hiển thị theo lớp đối tượng.

---

## 💻 Yêu Cầu Hệ Thống

### Phần Cứng
- **CPU**: Intel Core i5 hoặc tương đương.
- **RAM**: Tối thiểu 8GB.
- **GPU** (Khuyến nghị): NVIDIA GPU với CUDA support để tăng tốc độ xử lý.

### Phần Mềm
- **OS**: Windows 10/11, macOS, Linux.
- **Python**: 3.10 trở lên.

### Dependencies Chính
```
PyQt5>=5.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
reportlab>=4.0.0
numpy>=1.24.0
```

---

## 🚀 Cài Đặt

### Bước 1: Clone Repository

```bash
git clone https://github.com/ntd237/test_model_images_gui_02122025.git
cd test_model_images_gui_02122025
```

### Bước 2: Tạo Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Bước 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Chuẩn bị Model
Đặt các file model `.pt` của bạn vào thư mục `resources/models/` (tùy chọn).

---

## 📖 Sử Dụng

### Khởi Chạy Tool

```bash
python main.py
```

### Workflow Xử Lý Hàng Loạt (Batch Processing)

1. **Mở Batch Dialog**: Click nút "Batch Processing" trên giao diện chính.
2. **Chọn Folder**: Chọn thư mục chứa ảnh cần test.
3. **Load Model**: Chọn model YOLO (.pt).
4. **Cấu Hình**:
   - Chọn thiết bị (CPU/CUDA).
   - Điều chỉnh Confidence và IOU Threshold.
5. **Chạy**: Nhấn "Process All Images".
6. **Xuất Báo Cáo**: Sau khi chạy xong, chọn "Export Kết Quả" -> "Export PDF".

---

## 📁 Cấu Trúc Dự Án

```
test_model_images_gui_02122025/
├── src/                          # Source code chính
│   ├── core/                     # Core logic
│   │   ├── __init__.py
│   │   ├── batch_processor.py    # Xử lý hàng loạt (Batch Processing Logic)
│   │   ├── inference.py          # Engine chạy model YOLO
│   │   ├── model_loader.py       # Quản lý load model
│   │   └── model_manager.py      # Quản lý so sánh nhiều model
│   │
│   ├── gui/                      # Giao diện người dùng
│   │   ├── __init__.py
│   │   ├── batch_dialog.py       # Hộp thoại xử lý hàng loạt
│   │   ├── comparison_window.py  # Cửa sổ so sánh model
│   │   ├── histogram_dialog.py   # Biểu đồ phân phối
│   │   ├── main_window.py        # Cửa sổ chính
│   │   ├── styles.py             # Stylesheet (Dark Theme)
│   │   └── widgets.py            # Custom widgets (ImageLabel, InfoPanel...)
│   │
│   └── utils/                    # Các tiện ích
│       ├── __init__.py
│       ├── device_utils.py       # Tiện ích quản lý thiết bị (CPU/GPU)
│       ├── export_utils.py       # Tiện ích xuất báo cáo (PDF, CSV, JSON)
│       └── image_utils.py        # Xử lý ảnh (Resize, Draw BBox)
│
├── resources/                    # Tài nguyên
│   ├── models/                   # Chứa file model .pt
│   ├── sample_images/            # Ảnh mẫu để test
│   ├── output_images/            # Thư mục lưu kết quả mặc định
│   └── icons/                    # Icons cho giao diện
│
├── main.py                       # File khởi chạy ứng dụng
├── requirements.txt              # Danh sách thư viện phụ thuộc
└── README.md                     # Tài liệu hướng dẫn
```

---

## 🐛 Troubleshooting

### Lỗi "No module named 'PyQt5'"
```bash
pip install PyQt5
```

### Lỗi khi xuất PDF
Đảm bảo bạn đã cài đặt `reportlab`:
```bash
pip install reportlab
```

### Không nhận diện được GPU
Kiểm tra cài đặt PyTorch với CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Nếu trả về `False`, hãy cài lại PyTorch phiên bản hỗ trợ CUDA từ trang chủ pytorch.org.

---

## 👤 Author

**ntd237**
- Email: ntd237.work@gmail.com
- GitHub: [@ntd237](https://github.com/ntd237)

---

## 📄 License

Dự án này được phân phối dưới giấy phép [MIT License](LICENSE).

---

**Enjoy testing your YOLO models! 🚀**
