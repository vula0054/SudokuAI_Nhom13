"""Kiểm thử nhận dạng ảnh trên ảnh tổng hợp: nhiều phông chữ, xoay/phối cảnh, nhiễu, nền. python test_vision.py"""
import random, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from csp import *
from vision import recognize
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
def render(g, n, font, rng, mode):
    S = 70 * n + 40; im = Image.new("L", (S, S), 255); d = ImageDraw.Draw(im); f = ImageFont.truetype(font, 40 if n == 9 else 44)
    b = int(n ** .5)
    for k in range(n + 1): d.line([(20, 20 + 70*k), (20 + 70*n, 20 + 70*k)], fill=0, width=4 if k % b == 0 else 1); d.line([(20 + 70*k, 20), (20 + 70*k, 20 + 70*n)], fill=0, width=4 if k % b == 0 else 1)
    for i, v in enumerate(g):
        if v: r, c = divmod(i, n); d.text((20 + 70*c + 35, 20 + 70*r + 35), str(v), fill=0, font=f, anchor="mm")
    a = np.array(im)
    if mode == "clean": return a
    h, w = a.shape; big = np.full((int(h*1.5), int(w*1.5)), 215, np.uint8); big[h//4:h//4+h, w//4:w//4+w] = a
    src = np.float32([[0, 0], [big.shape[1], 0], [big.shape[1], big.shape[0]], [0, big.shape[0]]]); j = lambda: rng.uniform(-70, 70)
    dst = src + np.float32([[j(), j()] for _ in range(4)]) * (1 if mode == "persp" else .3)
    M = cv2.getPerspectiveTransform(src, dst); a = cv2.warpPerspective(big, M, (big.shape[1], big.shape[0]), borderValue=200)
    a = cv2.GaussianBlur(a, (3, 3), 0); a = np.clip(a.astype(float) + np.random.RandomState(rng.randint(0, 9999)).normal(0, 9, a.shape), 0, 255).astype(np.uint8)
    return a
if __name__ == "__main__":
    rng = random.Random(1); res = {}
    for n, K in ((9, 5), (4, 3)):
        for mode in ("clean", "persp"):
            ok = tot = exact = 0; cnt = 0
            for k in range(K):
                sp, g, _, _, _ = generate(n, "classic", 1, rng); font = FONTS[(k + cnt) % len(FONTS)]; cnt += 1
                png = cv2.imencode(".png", render(g, n, font, rng, mode))[1].tobytes()
                r = recognize(png, n)["grid"]; ok += sum(a == b for a, b in zip(r, g)); tot += len(g); exact += r == g
            res[(n, mode)] = (ok / tot, exact, K); print(f"{n}x{n} {mode:6s}: đúng từng ô {ok/tot:.4f}, nguyên lưới {exact}/{K}", flush=True)
