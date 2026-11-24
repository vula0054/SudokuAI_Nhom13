"""Huấn luyện bộ đọc chữ số 1-9 từ ảnh: render chữ số bằng các phông có sẵn + biến dạng nhẹ → MLP. python train_digits.py"""
import glob, random, numpy as np, cv2, joblib
from PIL import Image, ImageDraw, ImageFont
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from vision import norm
def fonts():
    fs = sorted(set(glob.glob("/usr/share/fonts/**/*.ttf", recursive=True) + glob.glob("/usr/share/fonts/**/*.otf", recursive=True) + glob.glob("/usr/share/texmf/fonts/opentype/**/*.otf", recursive=True)))
    bad = ("Italic", "Oblique", "Thin", "Light", "Symbol", "Dingbat", "Emoji", "Math", "Mono-", "Icons", "Awesome", "Wingdings", "Webdings", "cmex", "Hebrew", "Arabic", "Thai", "Devanagari")
    return [f for f in fs if not any(b in f for b in bad)]
def sample(font, v, rng):
    S = 96; im = Image.new("L", (S, S), 0); d = ImageDraw.Draw(im)
    try: f = ImageFont.truetype(font, rng.randint(40, 64)); d.text((S/2, S/2), str(v), fill=255, font=f, anchor="mm")
    except Exception: return None
    a = np.array(im)
    if a.max() == 0: return None
    M = cv2.getRotationMatrix2D((S/2, S/2), rng.uniform(-4, 4), 1); a = cv2.warpAffine(a, M, (S, S))
    if rng.random() < .5: a = cv2.GaussianBlur(a, (3, 3), 0)
    a = cv2.threshold(a, rng.randint(70, 150), 255, cv2.THRESH_BINARY)[1]
    ys, xs = np.nonzero(a)
    if len(xs) == 0: return None
    return norm(a[ys.min():ys.max()+1, xs.min():xs.max()+1]).reshape(-1) / 255.0
def build(fl, reps, rng):
    X, y = [], []
    for f in fl:
        for v in range(1, 10):
            for _ in range(reps):
                x = sample(f, v, rng)
                if x is not None: X.append(x); y.append(v)
    return np.array(X), np.array(y)
if __name__ == "__main__":
    rng = random.Random(0); fl = fonts(); rng.shuffle(fl); print(len(fl), "phông")
    cut = int(len(fl) * .8); Xtr, ytr = build(fl[:cut], 4, rng); Xte, yte = build(fl[cut:], 2, rng)
    m = make_pipeline(StandardScaler(), MLPClassifier((256, 128), alpha=1e-3, max_iter=300, random_state=0)).fit(Xtr, ytr)
    print("Độ chính xác trên PHÔNG CHƯA THẤY:", round(float((m.predict(Xte) == yte).mean()), 4), "| n =", len(yte))
    X, y = build(fl, 4, rng); m.fit(X, y); joblib.dump(m, "digits.joblib", compress=3)
