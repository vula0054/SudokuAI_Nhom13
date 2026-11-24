# Sudoku CSP Solver (đa biến thể + web + nhận dạng ảnh + đánh giá)
Kỹ thuật AI trọng tâm: **CSP = backtracking + lan truyền ràng buộc (AC-3)**; ML (RF/MLP) chỉ dùng để đối chứng và đọc ảnh đầu vào.
Chạy web: `pip install numpy scikit-learn opencv-python-headless joblib` rồi `python server.py` → http://localhost:8000
- csp.py: **`choose_method()` tự chọn thuật toán theo đề; `solve(..., method="auto")`**; mô hình CSP tổng quát: Classic, X, Anti-knight, Jigsaw (4/9/16), Killer (4/9); AC-3, MAC, MRV, lọc tổng lồng Killer, sinh đề, kiểm tra đề
- vision.py + digits.joblib (train_digits.py): nhận dạng đề 4×4/9×9 từ ảnh (OpenCV tìm lưới + MLP đọc chữ số); test_vision.py kiểm thử
- server.py + index.html: giao diện web (mặc định thuật toán = "Tự động"; sinh đề / nạp chuỗi / nạp ảnh đều hiện thuật toán được chọn và lý do; thêm API `/api/suggest`)
- evaluate.py: Precision/Recall/F1 (đề lỗi, độ khó), neural vs CSP, giải theo biến thể, nhận dạng ảnh → results/eval.json
- legacy_9x9/: bản 9×9 gốc cho mục 2.5 (chạy riêng)
Giới hạn: ảnh chỉ đọc lưới số (Killer/Jigsaw không đọc được lồng/vùng từ ảnh); ảnh kiểm thử là ảnh tổng hợp, chưa thử ảnh chụp thật.
