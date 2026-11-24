"""Nhận dạng đề Sudoku từ ảnh (lưới số, n = 4 hoặc 9): OpenCV tìm lưới + nắn phối cảnh, MLP nhỏ (digits.joblib) đọc từng chữ số.
recognize(bytes, n) -> dict(grid, n, uncertain=[chỉ số ô đọc không chắc], found_grid=bool)
Giới hạn: chỉ đọc các chữ số đã in (đề có sẵn gợi ý); Killer/Jigsaw cần nhập thêm lồng/vùng bằng tay."""
import os, cv2, numpy as np, joblib

def _order(pts):
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])

def find_grid(gray):
    """Trả về ảnh xám đã nắn vuông của lưới; nếu không thấy khung thì dùng cả ảnh."""
    h, w = gray.shape; sc = 1000 / max(h, w)
    if sc < 1: gray = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    th = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(cs, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < 0.15 * gray.shape[0] * gray.shape[1]: break
        ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(ap) == 4: best = ap.reshape(4, 2).astype(np.float32); break
    S = 900
    if best is None: return cv2.resize(gray, (S, S)), False
    M = cv2.getPerspectiveTransform(_order(best), np.float32([[0, 0], [S, 0], [S, S], [0, S]]))
    return cv2.warpPerspective(gray, M, (S, S)), True

def _digit(cell):
    """Tách chữ số trong 1 ô đã nhị phân hóa (bỏ nét kẻ dính viền). None nếu ô trống."""
    h, w = cell.shape; m = int(h * 0.10); c = cell[m:h-m, m:w-m]
    bw = c
    k, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    H, W = bw.shape; keep = np.zeros_like(bw)
    for i in range(1, k):
        x, y, ww, hh, a = st[i]
        touches = x <= 1 or y <= 1 or x + ww >= W - 1 or y + hh >= H - 1
        if a >= 0.012 * H * W and hh >= 0.25 * H and not (touches and (ww > 0.8 * W or hh > 0.9 * H)):
            keep[lab == i] = 255
    ys, xs = np.nonzero(keep)
    if len(xs) < 0.015 * H * W: return None
    return norm(keep[ys.min():ys.max()+1, xs.min():xs.max()+1])

def norm(d):
    """Chữ trắng trên nền đen → ảnh 28×28 (cạnh dài 20px, căn giữa), dùng chung cho huấn luyện và nhận dạng."""
    s = 20 / max(d.shape); d = cv2.resize(d, (max(1, round(d.shape[1] * s)), max(1, round(d.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.uint8); y0, x0 = (28 - d.shape[0]) // 2, (28 - d.shape[1]) // 2
    out[y0:y0+d.shape[0], x0:x0+d.shape[1]] = d; return out

_model = None
def _ocr(img, n):
    global _model
    if _model is None: _model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "digits.joblib"))
    p = _model.predict_proba(img.reshape(1, -1) / 255.0)[0]; ok = [k for k, c in enumerate(_model.classes_) if c <= n]
    k = max(ok, key=lambda k: p[k]); return int(_model.classes_[k]), 100 * float(p[k])

def recognize(data, n=9):
    if n not in (4, 9): raise ValueError("Nhận dạng ảnh hỗ trợ lưới 4×4 và 9×9")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError("Không đọc được ảnh")
    g, found = find_grid(img); S = g.shape[0] // n
    g = cv2.adaptiveThreshold(cv2.medianBlur(g, 5), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 12)
    grid, unsure = [0] * (n * n), []
    for r in range(n):
        for c in range(n):
            d = _digit(g[r*S:(r+1)*S, c*S:(c+1)*S])
            if d is None: continue
            v, cf = _ocr(d, n)
            if v is None: unsure.append(r*n + c); continue
            grid[r*n + c] = v
            if cf < 70: unsure.append(r*n + c)
    return dict(grid=grid, n=n, uncertain=unsure, found_grid=found, clues=sum(1 for x in grid if x))
