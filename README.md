# DPED: Deep Photo Enhancer Dataset

## Tóm tắt Model

DPED là một mô hình học sâu dùng để nâng cao chất lượng ảnh chụp từ điện thoại thông minh, giúp ảnh đạt chất lượng tương đương máy ảnh chuyên nghiệp. Mô hình sử dụng mạng nơ-ron tích chập (CNN) để học chuyển đổi giữa ảnh điện thoại và ảnh máy ảnh DSLR.

## Mô tả các File

- `train.py`: Huấn luyện mô hình nâng cao ảnh trên tập dữ liệu DPED.
- `test.py`: Chạy mô hình đã huấn luyện để nâng cao ảnh mới.
- `dataset.py`: Xử lý và chuẩn bị dữ liệu đầu vào cho mô hình.
- `utils.py`: Chứa các hàm hỗ trợ như tiền xử lý ảnh, lưu kết quả, v.v.
- `download_dataset.sh`: Script tải tập dữ liệu DPED về máy.

>script: python run_on_image.py --input "đường_dẫn_ảnh_hoặc_thư_mục" .