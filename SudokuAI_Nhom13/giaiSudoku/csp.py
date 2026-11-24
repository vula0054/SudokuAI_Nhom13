"""Lõi CSP tổng quát cho Sudoku + biến thể (classic, x, antiknight, jigsaw, killer), kích thước 4/9/16.
Thêm biến thể = thêm ràng buộc trong Spec; bộ giải (AC-3, MAC, MRV) không đổi.
Killer: ràng buộc 'khác nhau' trong lồng (qua peers) + ràng buộc tổng lồng (cage_prune, lọc miền theo mọi tổ hợp hợp lệ)."""
import random, time
from collections import deque
VARIANTS = {"classic": "Classic", "x": "X-Sudoku", "antiknight": "Anti-knight", "jigsaw": "Jigsaw", "killer": "Killer"}
SIZES = {"classic": (4, 9, 16), "x": (4, 9, 16), "antiknight": (4, 9, 16), "jigsaw": (4, 9, 16), "killer": (4, 9)}
KN = [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
def bit(v): return 1 << (v - 1)
def val(m): return m.bit_length()
def pc(x): return bin(x).count("1")
def nbr(i, n):
    r, c = divmod(i, n)
    return [x*n+y for x, y in ((r+1,c),(r-1,c),(r,c+1),(r,c-1)) if 0 <= x < n and 0 <= y < n]

def _make_jigsaw(n, rng):
    """Chia lưới n×n thành n vùng liên thông, mỗi vùng n ô (mọc dần từ n hạt giống)."""
    while True:
        reg = [-1]*(n*n); size = [1]*n
        for k, c in enumerate(rng.sample(range(n*n), n)): reg[c] = k
        while -1 in reg:
            for k in sorted(range(n), key=lambda k: (size[k], rng.random())):
                if size[k] >= n: continue
                fr = [j for i in range(n*n) if reg[i] == k for j in nbr(i, n) if reg[j] == -1]
                if fr: reg[rng.choice(fr)] = k; size[k] += 1; break
            else: break
        if -1 not in reg: return reg

_cache = {}
def get_spec(n=9, variant="classic", regions=None, cages=None):
    cg = tuple((tuple(c), int(t)) for c, t in cages) if cages else None
    key = (n, variant, tuple(regions) if regions else None, cg)
    if key not in _cache:
        if len(_cache) > 200: _cache.clear()
        _cache[key] = Spec(n, variant, regions, cg)
    return _cache[key]

class Spec:
    def __init__(s, n, variant, regions=None, cages=None):
        b = int(round(n ** .5)); assert b*b == n and variant in VARIANTS
        s.n, s.N, s.variant, s.ALL = n, n*n, variant, (1 << n) - 1
        s.cages, s.memo = [], {}
        if variant == "killer":
            if not cages: raise ValueError("Killer cần danh sách lồng (cages)")
            s.cages = [(tuple(c), t) for c, t in cages]
            cover = sorted(x for c, _ in s.cages for x in c)
            if cover != list(range(s.N)): raise ValueError("Các lồng phải phủ mỗi ô đúng một lần")
        rows = [[r*n+c for c in range(n)] for r in range(n)]
        cols = [[r*n+c for r in range(n)] for c in range(n)]
        if variant == "jigsaw":
            assert regions and len(regions) == s.N, "Jigsaw cần bản đồ vùng"
            s.box_of = list(regions)
        else:
            s.box_of = [(r//b)*b + c//b for r in range(n) for c in range(n)]
        boxes = [[i for i in range(s.N) if s.box_of[i] == k] for k in range(n)]
        groups = rows + cols + boxes
        if variant == "x":
            groups += [[i*n+i for i in range(n)], [i*n+n-1-i for i in range(n)]]
        if variant == "killer":
            groups += [list(c) for c, _ in s.cages]
            s.cage_of = [0]*s.N
            for k, (c, _) in enumerate(s.cages):
                for i in c: s.cage_of[i] = k
        P = [set() for _ in range(s.N)]
        for g in groups:
            for i in g: P[i].update(g)
        if variant == "antiknight":
            for i in range(s.N):
                r, c = divmod(i, n)
                for dr, dc in KN:
                    if 0 <= r+dr < n and 0 <= c+dc < n: P[i].add((r+dr)*n + c+dc)
        for i in range(s.N): P[i].discard(i)
        s.peers = [sorted(p) for p in P]
        s.arcs = [(i, j) for i in range(s.N) for j in s.peers[i]]

class Timeout(Exception): pass
class Stats:
    def __init__(s):
        s.assign = s.backtrack = s.revise = s.prune = 0; s.t = 0.0; s.solved = s.timeout = False
        s.method = s.reason = None; s.fallback = False   # thuật toán thực dùng, lý do chọn, có phải chạy lại bằng D không

class Solver:
    def __init__(s, sp, limit=None, trace=None, cap=5000):
        s.sp, s.limit, s.trace, s.cap, s.st, s.tick = sp, limit, trace, cap, Stats(), 0
        s.t0 = time.perf_counter()
    def _t(s, e):
        if s.trace is not None and len(s.trace) < s.cap: s.trace.append(e)
    def _time(s):
        s.tick += 1
        if s.limit and s.tick % 128 == 0 and time.perf_counter() - s.t0 > s.limit: raise Timeout()
    def init_domains(s, g): return [bit(v) if v else s.sp.ALL for v in g]

    def ac3(s, D, queue=None):
        """AC-3 cho ràng buộc ≠: Xi mất hỗ trợ chỉ khi D[Xj] đơn trị. False nếu có miền rỗng."""
        q = deque(s.sp.arcs if queue is None else queue); st = s.st; P = s.sp.peers
        while q:
            xi, xj = q.popleft(); st.revise += 1; dj = D[xj]
            if dj & (dj-1) == 0 and D[xi] & dj:
                D[xi] &= ~dj; st.prune += 1; s._t(("PRUNE", xi, val(dj), xj))
                if D[xi] == 0: return False
                if D[xi] & (D[xi]-1) == 0:
                    for xk in P[xi]:
                        if xk != xj: q.append((xk, xi))
        return True

    def cage_prune(s, D):
        """Killer: với mỗi lồng, giữ lại các giá trị xuất hiện trong ít nhất một tổ hợp (khác nhau, đúng tổng).
        Trả về (ok, danh sách ô bị đổi miền)."""
        sp, n, changed = s.sp, s.sp.n, []
        for ci, (cells, tot) in enumerate(sp.cages):
            doms = tuple(D[c] for c in cells); key = (ci, doms); res = sp.memo.get(key)
            if res is None:
                k = len(cells); uni = [0]*k; cur = [0]*k
                def rec(j, rem, used):
                    if j == k - 1:
                        if 1 <= rem <= n and doms[j] & bit(rem) and not used & bit(rem):
                            cur[j] = rem
                            for x in range(k): uni[x] |= bit(cur[x])
                            return True
                        return False
                    ok = False
                    for v in range(1, min(n, rem - (k - 1 - j)) + 1):
                        if doms[j] & bit(v) and not used & bit(v):
                            cur[j] = v
                            if rec(j+1, rem - v, used | bit(v)): ok = True
                    return ok
                res = tuple(uni) if rec(0, tot, 0) else None
                if len(sp.memo) > 400000: sp.memo.clear()
                sp.memo[key] = res
            if res is None: return False, changed
            for x, c in enumerate(cells):
                nd = D[c] & res[x]
                if nd != D[c]:
                    for v in range(1, n+1):
                        if D[c] & bit(v) and not nd & bit(v): s._t(("CAGE", c, v, ci))
                    D[c] = nd; changed.append(c); s.st.prune += 1
                    if nd == 0: return False, changed
        return True, changed

    def propagate(s, D, queue=None):
        """AC-3 (+ lọc tổng lồng cho Killer) lặp tới điểm bất động. False nếu có miền rỗng."""
        if not s.ac3(D, queue): return False
        if not s.sp.cages: return True
        P = s.sp.peers
        while True:
            ok, ch = s.cage_prune(D)
            if not ok: return False
            if not ch: return True
            if not s.ac3(D, [(k, i) for i in ch for k in P[i]]): return False

    def bt_plain(s, grid, count=False, domains=None):
        """A/B: duyệt ô theo thứ tự cố định, chỉ kiểm tra xung đột khi gán (Killer: thêm kiểm tra tổng lồng)."""
        g = list(grid); P = s.sp.peers; st = s.st; sols = []; sp = s.sp
        E = [i for i in range(sp.N) if g[i] == 0]
        def cage_ok(i):
            if not sp.cages: return True
            cells, tot = sp.cages[sp.cage_of[i]]
            sm = sum(g[c] for c in cells); u = sum(1 for c in cells if g[c] == 0)
            return sm + u <= tot <= sm + u * sp.n and (u > 0 or sm == tot)
        def rec(k):
            if k == len(E): sols.append(list(g)); return not count or len(sols) >= 2
            i = E[k]; used = 0
            for j in P[i]:
                if g[j]: used |= bit(g[j])
            cand = (domains[i] if domains else s.sp.ALL) & ~used
            for v in range(1, s.sp.n+1):
                if cand & bit(v):
                    s._time(); g[i] = v
                    if not cage_ok(i): g[i] = 0; continue
                    st.assign += 1; s._t(("ASSIGN", i, v, None))
                    if rec(k+1): return True
                    g[i] = 0; s._t(("BACKTRACK", i, v, None))
            st.backtrack += 1; return False
        rec(0); return sols

    def mac(s, D, mrv, count=False, rand=None):
        """C/D: AC-3 sau mỗi lần gán (MAC), tùy chọn MRV."""
        st = s.st; P = s.sp.peers; sols = []
        def pick(D):
            best, bs = -1, 99
            for i in range(s.sp.N):
                p = pc(D[i])
                if p > 1:
                    if not mrv: return i
                    if p < bs: best, bs = i, p
                    if p == 2: break
            return best
        def rec(D):
            i = pick(D)
            if i < 0: sols.append([val(d) for d in D]); return not count or len(sols) >= 2
            vals = [v for v in range(1, s.sp.n+1) if D[i] & bit(v)]
            if rand: rand.shuffle(vals)
            for v in vals:
                s._time(); D2 = D[:]; D2[i] = bit(v); st.assign += 1; s._t(("ASSIGN", i, v, None))
                if s.propagate(D2, [(k, i) for k in P[i]]) and rec(D2): return True
                s._t(("BACKTRACK", i, v, None))
            st.backtrack += 1; return False
        rec(D); return sols

def verify(sp, grid, sol):
    return (all(not grid[i] or grid[i] == sol[i] for i in range(sp.N))
            and all(1 <= x <= sp.n for x in sol)
            and all(sol[i] != sol[j] for i in range(sp.N) for j in sp.peers[i])
            and all(sum(sol[c] for c in cells) == t for cells, t in sp.cages))

METHODS = {"A": "Backtracking thuần", "B": "AC-3 tiền xử lý + Backtracking", "C": "MAC", "D": "MAC + MRV"}

def choose_method(sp, grid):
    """Tự chọn thuật toán theo đặc trưng của đề (chạy AC-3 một lần, rẻ so với giải).
    Quy tắc dựa trên đo thực tế:
      1. AC-3 phát hiện mâu thuẫn          -> D (dừng ngay, đề vô nghiệm)
      2. AC-3 xác định hết mọi ô           -> D (không cần tìm kiếm, nghiệm có ngay sau lan truyền)
      3. Lưới 4×4 (≤16 ô)                  -> A (không gian tìm kiếm rất nhỏ, khỏi tốn chi phí suy diễn)
      4. Còn lại (còn ô nhiều ứng viên)    -> D (MRV + AC-3 cắt nhánh; A/B/C có thể quá giờ ở đề khó)
    Trả về dict(method, reason, features)."""
    s = Solver(sp); D = s.init_domains(grid); ok = s.propagate(D)
    empty = sum(1 for x in grid if x == 0)
    open_ = [i for i in range(sp.N) if ok and grid[i] == 0 and pc(D[i]) > 1]
    feat = dict(n=sp.n, variant=sp.variant, empty=empty, open=len(open_),
                avg_dom=round(sum(pc(D[i]) for i in open_) / len(open_), 2) if open_ else 0)
    if not ok:
        return dict(method="D", features=feat, reason="AC-3 phát hiện mâu thuẫn ngay từ đầu → đề vô nghiệm, không cần tìm kiếm")
    if not open_:
        return dict(method="D", features=feat, reason=f"AC-3 đã xác định đủ {empty} ô trống → giải bằng lan truyền ràng buộc, không cần quay lui")
    if sp.N <= 16:
        return dict(method="A", features=feat, reason="Lưới 4×4: không gian tìm kiếm rất nhỏ, backtracking thuần là đủ và nhẹ nhất")
    return dict(method="D", features=feat,
                reason=f"Sau AC-3 còn {len(open_)}/{empty} ô chưa xác định (TB {feat['avg_dom']} ứng viên) → dùng MAC + MRV để cắt nhánh sớm")

def solve(sp, grid, method="D", limit=10.0, trace=None):
    """method: A/B/C/D, hoặc 'auto' để tự chọn (nếu thuật toán đã chọn quá giờ thì chạy lại bằng D)."""
    auto = method == "auto"; reason = None
    if auto: info = choose_method(sp, grid); method, reason = info["method"], info["reason"]
    s = Solver(sp, limit, trace); sol = None
    try:
        if method == "A": r = s.bt_plain(grid)
        else:
            D = s.init_domains(grid)
            if not s.propagate(D): r = []
            elif method == "B": r = s.bt_plain(grid, domains=D)
            else: r = s.mac(D, mrv=(method == "D"))
        sol = r[0] if r else None
    except Timeout: s.st.timeout = True
    s.st.t = time.perf_counter() - s.t0
    s.st.solved = sol is not None and verify(sp, grid, sol)
    s.st.method, s.st.reason = method, reason
    if auto and s.st.timeout and method != "D":      # phương án dự phòng
        if trace is not None: del trace[:]
        sol2, st2 = solve(sp, grid, "D", limit, trace)
        st2.t += s.st.t; st2.fallback = True
        st2.reason = f"{reason}; quá {limit:g} s nên chuyển sang D (MAC + MRV)"
        return sol2, st2
    return sol, s.st

def count_solutions(sp, grid, limit=None, method="D"):
    """Đếm nghiệm tới 2. Trả -1 nếu quá thời gian. method A dùng làm 'trọng tài' độc lập."""
    s = Solver(sp, limit)
    try:
        if method == "A": return len(s.bt_plain(grid, count=True))
        D = s.init_domains(grid)
        return len(s.mac(D, True, count=True)) if s.propagate(D) else 0
    except Timeout: return -1

def solutions(sp, grid, limit=None):
    """Tối đa 2 nghiệm (để biết đề có duy nhất không). Ném Timeout nếu quá giờ."""
    s = Solver(sp, limit); D = s.init_domains(grid)
    return s.mac(D, True, count=True) if s.propagate(D) else []

def ac3_only(sp, grid):
    s = Solver(sp); D = s.init_domains(grid); ok = s.propagate(D)
    emp = sum(1 for x in grid if x == 0)
    return ok, (sum(1 for i in range(sp.N) if grid[i] == 0 and pc(D[i]) == 1) if ok else 0), emp

def check(sp, grid, limit=10):
    """Kiểm tra đề: conflict / no_solution / multiple / ok, kèm độ khó."""
    ok, det, emp = ac3_only(sp, grid)
    if not ok: return dict(status="conflict", count=0)
    n = count_solutions(sp, grid, limit)
    if n == -1: return dict(status="timeout", count=-1)
    if n == 0: return dict(status="no_solution", count=0)
    if n > 1: return dict(status="multiple", count=n)
    _, st = solve(sp, grid, "D", limit)
    lvl = "Dễ" if det == emp else ("Trung bình" if st.backtrack == 0 else "Khó")
    return dict(status="ok", count=1, difficulty=lvl, ac3_ratio=det/max(emp, 1))

def _pattern(sp, rng):   # lưới đầy cho Sudoku chuẩn cỡ lớn: mẫu cơ sở + hoán vị trong dải
    n = sp.n; b = int(round(n ** .5))
    def sh():
        o = []
        for x in rng.sample(range(b), b): o += [x*b + y for y in rng.sample(range(b), b)]
        return o
    R, C, V = sh(), sh(), rng.sample(range(1, n+1), n)
    return [V[(b*(R[r] % b) + R[r]//b + C[c]) % n] for r in range(n) for c in range(n)]

def full_grid(sp, rng, tries=60, tl=1.5):
    if sp.n == 16 and sp.variant == "classic": return _pattern(sp, rng)
    for _ in range(tries):
        s = Solver(sp, tl)
        try: r = s.mac(s.init_domains([0]*sp.N), True, rand=rng)
        except Timeout: continue
        if r: return r[0]
    raise ValueError("Không sinh được lưới đầy cho cấu hình này")

LEVEL_FRAC = {4: (.6, .5, .4), 9: (.49, .40, .31), 16: (.62, .55, .5)}
def make_puzzle(sp, rng, target, tlimit=8.0):
    """Xóa dần ô khỏi lưới đầy, giữ nghiệm duy nhất (dừng khi đủ gợi ý hoặc hết thời gian)."""
    sol = full_grid(sp, rng); g = sol[:]; cells = list(range(sp.N)); rng.shuffle(cells)
    clues, t0 = sp.N, time.perf_counter()
    for i in cells:
        if clues <= target or time.perf_counter() - t0 > tlimit: break
        old = g[i]; g[i] = 0
        if count_solutions(sp, g, 2) != 1: g[i] = old
        else: clues -= 1
    return g, sol

def parse_grid(cells): return [int(x) if str(x).strip() else 0 for x in cells]

def make_jigsaw(n, rng):
    """Bản đồ vùng ngẫu nhiên nhưng đảm bảo có ít nhất một lưới đầy hợp lệ (loại bố cục vô nghiệm)."""
    while True:
        reg = _make_jigsaw(n, rng)
        try: full_grid(get_spec(n, "jigsaw", reg), rng, 3, 0.4); return reg
        except ValueError: continue

def make_cages(n, sol, rng, maxsize):
    """Chia lưới thành các lồng liên thông, giá trị trong lồng đôi một khác nhau; tổng lấy từ nghiệm."""
    N = n*n; cage = [-1]*N; cages = []
    for i in rng.sample(range(N), N):
        if cage[i] != -1: continue
        cells, vals, target = [i], {sol[i]}, rng.randint(1, maxsize); cage[i] = len(cages)
        while len(cells) < target:
            fr = [j for c in cells for j in nbr(c, n) if cage[j] == -1 and sol[j] not in vals]
            if not fr: break
            j = rng.choice(fr); cells.append(j); cage[j] = len(cages); vals.add(sol[j])
        cages.append((sorted(cells), sum(sol[c] for c in cells)))
    return cages

KILLER_LEVEL = {4: ((2, 2), (3, 1), (3, 0)), 9: ((3, 8), (4, 3), (5, 0))}   # (cỡ lồng tối đa, số gợi ý thêm)
def make_killer(n, rng, lvl, tl=6.0):
    """Sinh Killer: lồng ngẫu nhiên, rồi thêm gợi ý tại ô khác nhau giữa 2 nghiệm cho tới khi nghiệm duy nhất."""
    maxsize, extra = KILLER_LEVEL[n][lvl]; N = n*n
    for _ in range(30):
        sol = full_grid(get_spec(n, "classic"), rng); cages = make_cages(n, sol, rng, maxsize)
        sp = get_spec(n, "killer", cages=cages); g = [0]*N
        try:
            for _ in range(N):
                ss = solutions(sp, g, tl)
                if len(ss) <= 1: break
                i = rng.choice([k for k in range(N) if ss[0][k] != ss[1][k]]); g[i] = sol[i]
        except Timeout: continue
        rest = [i for i in range(N) if not g[i]]; rng.shuffle(rest)
        for i in rest[:extra]: g[i] = sol[i]
        return sp, g, sol, cages
    raise ValueError("Không sinh được Killer")

def generate(n, variant, lvl, rng, tlimit=6.0):
    """Điểm vào chung cho web và đánh giá. lvl: 0 Dễ, 1 Trung bình, 2 Khó. Trả về (spec, đề, nghiệm, regions, cages)."""
    if n not in SIZES[variant]: raise ValueError(f"{VARIANTS[variant]} chỉ hỗ trợ cỡ {SIZES[variant]}")
    if variant == "killer":
        sp, g, sol, cages = make_killer(n, rng, lvl); return sp, g, sol, None, cages
    reg = make_jigsaw(n, rng) if variant == "jigsaw" else None
    sp = get_spec(n, variant, reg); g, sol = make_puzzle(sp, rng, int(sp.N * LEVEL_FRAC[n][lvl]), tlimit)
    return sp, g, sol, reg, None
