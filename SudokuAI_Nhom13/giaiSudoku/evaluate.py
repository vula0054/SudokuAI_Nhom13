"""Đánh giá: (1) phát hiện đề không hợp lệ, (2) phân loại độ khó bằng ML, (3) neural vs CSP, (4) giải theo biến thể.
Chạy: python evaluate.py  → results/eval.json (web đọc ở tab 'Đánh giá')."""
import random, json, time
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import precision_recall_fscore_support as PRF, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from csp import *
SP = get_spec(9, "classic")

def gen9(seed):
    rng = random.Random(seed); g, sol = make_puzzle(SP, rng, rng.randint(24, 38), 4); return g, sol

# ---------- 1. Phát hiện đề không hợp lệ ----------
# Bộ phát hiện = check(): AC-3 (miền rỗng → "conflict"), rồi đếm nghiệm bằng MAC+MRV tới 2 nghiệm (0 → vô nghiệm, ≥2 → nhiều nghiệm).
# Nhãn thật = count_solutions bằng backtracking thuần A (giới hạn 3 s, cài đặt độc lập với bộ phát hiện); ca quá giờ hoặc nhãn không như mong đợi bị loại.
# Lớp dương tính = "đề KHÔNG hợp lệ". cm = [TP, FP, FN, TN].
def invalid_case(args):
    g, seed, kind = args; rng = random.Random(seed); g = g[:]
    emp = [i for i in range(81) if g[i] == 0]
    if kind == "valid": h = g
    elif kind == "conflict":
        i = rng.choice(emp); j = rng.choice([p for p in SP.peers[i] if g[p]]); g[i] = g[j]; h = g
    elif kind == "multiple":
        h = g[:]; cl = [i for i in range(81) if h[i]]; rng.shuffle(cl)
        for i in cl[:12]:
            h[i] = 0
            if count_solutions(SP, h, 3, "A") == 2: break
        else: return None
    else:  # no_solution: thêm gợi ý sai không xung đột trực tiếp
        for _ in range(30):
            h = g[:]; i = rng.choice(emp); used = {g[p] for p in SP.peers[i]}
            ok = [v for v in range(1, 10) if v not in used]
            if not ok: continue
            h[i] = rng.choice(ok)
            if count_solutions(SP, h, 3, "A") == 0: break
        else: return None
    truth = count_solutions(SP, h, 3, "A")                 # trọng tài độc lập
    if truth == -1 or (kind == "valid") != (truth == 1): return None
    # conflict: nhãn đúng theo cấu trúc (trùng giá trị trong cùng nhóm)
    pred = check(SP, h, 5)["status"]
    return kind, int(kind != "valid"), int(pred != "ok")

def eval_invalid():
    with Pool() as p: base = p.map(gen9, range(1000, 1130))
    jobs = [(g, 7*k + t, kind) for k, (g, _) in enumerate(base) for t, kind in enumerate(["valid", "conflict", "multiple", "no_solution"])]
    with Pool() as p: rs = [r for r in p.map(invalid_case, jobs) if r]
    y = [r[1] for r in rs]; yh = [r[2] for r in rs]; P, R, F, _ = PRF(y, yh, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    bt = {k: sum(r[2] for r in rs if r[0] == k) / max(1, sum(1 for r in rs if r[0] == k)) for k in ("conflict", "multiple", "no_solution")}
    return dict(n=len(rs), precision=P, recall=R, f1=F, acc=accuracy_score(y, yh), cm=[int(tp), int(fp), int(fn), int(tn)], recall_by_type=bt)

# ---------- 2. Phân loại độ khó (nhãn = chi phí thật của backtracking; đặc trưng KHÔNG chứa chi phí) ----------
def feats(g):
    cand = []
    for i in range(81):
        if g[i] == 0:
            u = {g[p] for p in SP.peers[i] if g[p]}; cand.append(9 - len(u))
    ok, det, emp = ac3_only(SP, g)
    rowc = [sum(1 for c in range(9) if g[r*9+c]) for r in range(9)]; colc = [sum(1 for r in range(9) if g[r*9+c]) for c in range(9)]
    return [81 - emp, np.mean(cand), min(cand), sum(1 for c in cand if c == 1) / len(cand), det / emp, min(rowc), min(colc), np.std(rowc)]
def cost(g):
    _, st = solve(SP, g, "A", 1.0); return st.assign
def diff_case(seed):
    g, _ = gen9(seed); return feats(g), cost(g)

def eval_difficulty():
    with Pool() as p: rs = p.map(diff_case, range(5000, 5500))
    X = np.array([r[0] for r in rs]); c = np.log10(np.array([r[1] for r in rs]) + 1)
    q = np.quantile(c, [1/3, 2/3]); y = np.digitize(c, q); names = ["Dễ", "Trung bình", "Khó"]
    yh = cross_val_predict(RandomForestClassifier(200, random_state=0), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    P, R, F, _ = PRF(y, yh, labels=[0, 1, 2], zero_division=0)
    return dict(model="Random Forest, 5-fold CV, n=%d" % len(y), f1_macro=float(F.mean()), acc=accuracy_score(y, yh),
                per_class=[dict(cls=names[k], p=P[k], r=R[k], f1=F[k]) for k in range(3)],
                note="Nhãn = tam phân vị của log10(số lần gán + 1) khi giải bằng backtracking thuần A (giới hạn 1 s/đề, nên chi phí bị chặn ở các đề rất khó). 8 đặc trưng: số gợi ý; trung bình và min số ứng viên ngây thơ mỗi ô trống; tỉ lệ ô trống chỉ có 1 ứng viên ngây thơ; tỉ lệ ô trống được AC-3 xác định hoàn toàn; min số gợi ý theo hàng; min theo cột; độ lệch chuẩn gợi ý theo hàng. Không dùng chi phí tìm kiếm làm đặc trưng.")

# ---------- 3. Neural (MLP) đối chứng ----------
def onehot(gs):
    X = np.zeros((len(gs), 81 * 10), dtype=np.float32)
    for k, g in enumerate(gs):
        for i, v in enumerate(g): X[k, i*10 + v] = 1
    return X
def eval_neural():
    with Pool() as p: data = p.map(gen9, range(10000, 12400))
    tr, te = data[:2200], data[2200:]
    Y = np.zeros((len(tr), 729), dtype=np.float32)
    for k, (_, s) in enumerate(tr):
        for i, v in enumerate(s): Y[k, i*9 + v - 1] = 1
    m = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=40, random_state=0).fit(onehot([g for g, _ in tr]), Y)
    out = m.predict(onehot([g for g, _ in te])).reshape(-1, 81, 9).argmax(2) + 1
    ca = tot = ok = csp_ok = 0
    for (g, s), o in zip(te, out):
        pred = [g[i] or int(o[i]) for i in range(81)]
        for i in range(81):
            if g[i] == 0: tot += 1; ca += pred[i] == s[i]
        ok += verify(SP, g, pred); csp_ok += solve(SP, g, "D", 5)[1].solved
    return dict(model="MLP 810-256-256-729", train=len(tr), test=len(te), cell_acc=ca / tot, grid_ok=ok / len(te), csp_grid_ok=csp_ok / len(te))

# ---------- 4. Giải theo biến thể ----------
CONF = [(4, "classic", 15), (9, "classic", 20), (9, "x", 20), (9, "antiknight", 20), (9, "jigsaw", 20), (16, "classic", 5), (4, "killer", 15), (9, "killer", 20)]
def var_case(args):
    n, v, seed = args; sp, g, _, _, _ = generate(n, v, 2, random.Random(seed))
    sol, st = solve(sp, g, "D", 20); return sp.n, v, st.solved, st.assign, st.t * 1000, check(sp, g, 5)["status"] == "ok"
def eval_variants():
    jobs = [(n, v, 900 + k) for n, v, K in CONF for k in range(K)]
    with Pool() as p: rs = p.map(var_case, jobs)
    out = []
    for n, v, K in CONF:
        r = [x for x in rs if x[0] == n and x[1] == v]
        out.append(dict(name=f"{VARIANTS[v]} {n}×{n}", k=len(r), solved=f"{sum(x[2] for x in r)}/{len(r)}",
                        assign=float(np.mean([x[3] for x in r])), ms=float(np.mean([x[4] for x in r])),
                        unique=f"{sum(x[5] for x in r)}/{len(r)}"))
    return out

# ---------- 5. Nhận dạng ảnh (ảnh tổng hợp: 6 phông, nền xám, méo phối cảnh, nhiễu) ----------
def eval_vision():
    import cv2
    from test_vision import render, FONTS
    from vision import recognize
    rng = random.Random(1); out = []
    for n, K in ((9, 24), (4, 12)):
        for mode, nm in (("clean", "ảnh sạch"), ("persp", "méo phối cảnh + nhiễu")):
            ok = tot = ex = 0
            for k in range(K):
                sp, g, _, _, _ = generate(n, "classic", 1, rng); png = cv2.imencode(".png", render(g, n, FONTS[k % len(FONTS)], rng, mode))[1].tobytes()
                r = recognize(png, n)["grid"]; ok += sum(a == b for a, b in zip(r, g)); tot += len(g); ex += r == g
            out.append(dict(name=f"{n}×{n}, {nm}", k=K, cell_acc=ok / tot, grid_ok=f"{ex}/{K}"))
    return out

if __name__ == "__main__":
    t = time.time(); R = {}
    for k, f in [("invalid", eval_invalid), ("difficulty", eval_difficulty), ("neural", eval_neural), ("variants", eval_variants), ("vision", eval_vision)]:
        R[k] = f(); print(k, "xong", round(time.time() - t), "s", flush=True)
    json.dump(R, open("results/eval.json", "w"), ensure_ascii=False, indent=1, default=float)
