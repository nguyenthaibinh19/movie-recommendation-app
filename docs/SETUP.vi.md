# Chạy Lantern trên máy

## Cài đặt

Dùng Python 3.11–3.12 và Node.js 22 có npm. Có sẵn trọng số mô hình, không cần GPU hay huấn luyện lại.

Tại thư mục gốc, mở PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r src-backend/requirements.txt
.\.venv\Scripts\python.exe scripts/download_dataset.py
cd src-frontend
npm ci
cd ..
.\start.ps1
```

Có thể dùng uv: `uv venv --python 3.12 .venv`, rồi `uv pip install --python .venv/Scripts/python.exe -r src-backend/requirements.txt`.

Mở http://localhost:3000. API health check: http://localhost:5000/health. Ctrl+C dừng launcher; log backend ở `.logs/`. Nếu PowerShell chặn script, có thể chạy bằng `powershell -ExecutionPolicy Bypass -File .\start.ps1` cho lần chạy đó.

Chạy riêng hai terminal khi cần xem log trực tiếp:

```powershell
# Terminal 1, tại thư mục gốc
.\.venv\Scripts\python.exe src-backend/run.py
# Terminal 2
cd src-frontend
npm start
```

## Poster và API

Sao chép `src-frontend/.env.example` thành `.env.local`, tùy chọn điền `REACT_APP_TMDB_API_KEY`, rồi khởi động lại frontend. Giao diện nền tối dùng poster và ảnh nền thật từ TMDB. Không có key hoặc ảnh lỗi, ứng dụng hiển thị trạng thái thiếu ảnh, không dùng bìa chữ hay poster tự tạo. Poster tải riêng, không chặn danh sách gợi ý. Biến `REACT_APP_API_URL` đổi địa chỉ backend. Key trong React được gửi xuống trình duyệt.

## Dùng thử để phỏng vấn

1. Mở trang: phim popular có sẵn.
2. Mở **Personalize your picks**, điền đủ thông tin, chọn **Find my films**: API chấm điểm bằng DeepFM thật.
3. Nhấn bìa phim: mở danh sách phim tương tự bằng FAISS.
4. Like vài phim rồi mở **For your taste**: Lantern thích nghi theo tương tác. Một lượt thích có thể chưa vượt ngưỡng để bỏ popular fallback.
5. Chọn **Reset** để trở lại trạng thái ban đầu.

Lantern dùng mã phiên ngẫu nhiên trong `sessionStorage`, thay cho `guest` dùng chung trước đây. Đây không phải cơ chế đăng nhập. Gu vẫn lưu trong RAM backend và mất khi server khởi động lại.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe scripts/check_artifacts.py
.\.venv\Scripts\python.exe -m unittest discover -s src-backend -p "test_*.py" -v
cd src-frontend
npm test -- --watchAll=false --runInBand
npm run build
```

`requirements_lock.txt` và `npm_lock.txt` là bản ghi môi trường cũ; không dùng để cài. Nếu cổng 3000/5000 đang được dùng, dừng phiên server cũ trước khi chạy launcher.

`run.py` giữ workaround OpenMP cũ (`KMP_DUPLICATE_LIB_OK`) để nạp PyTorch và FAISS trên Windows; các test dùng cùng cấu hình. Cần môi trường OpenMP tương thích trước khi triển khai production. Dataset và các lưu ý phân phối: [DATA.md](DATA.md).
