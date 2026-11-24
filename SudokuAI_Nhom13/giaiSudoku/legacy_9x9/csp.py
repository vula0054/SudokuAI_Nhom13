"""Sudoku như bài toán CSP: backtracking thuần, AC-3, MAC, MRV. Bitmask domain."""
import random, time
from collections import deque

ALL = 0b111111111  # bit v-1 ứng với giá trị v
def popcount(x): return bin(x).count("1")
def bit(v): return 1 << (v - 1)
def val(mask): return mask.bit_length()  # chỉ dùng khi mask là singleton

# --- cấu trúc CSP ---
def _peers():
    P = []
    for i in range(81):
        r, c = divmod(i, 9)
        s = set()
        for k in range(9):
            s.add(r * 9 + k); s.add(k * 9 + c)
        br, bc = r // 3 * 3, c // 3 * 3
        for a in range(3):
            for b in range(3):
                s.add((br + a) * 9 + bc + b)
        s.discard(i)
        P.append(sorted(s))
    return P
PEERS = _peers()
ARCS = [(i, j) for i in range(81) for j in PEERS[i]]   # 1620 cung có hướng

def parse(s):
    s = s.replace(".", "0")
    return [int(ch) for ch in s]
def to_str(g): return "".join(str(x) if x else "." for x in g)

class Timeout(Exception): pass

class Stats:
    def __init__(self):
        self.assign = 0; self.backtrack = 0; self.revise = 0; self.prune = 0
        self.t = 0.0; self.solved = False; self.timeout = False; self.solutions = 0

class Solver:
    def __init__(self, limit=None, trace=None):
        self.limit = limit; self.trace = trace; self.st = Stats()
        self.t0 = 0.0; self.tick = 0

    def _check_time(self):
        self.tick += 1
        if self.limit and self.tick % 256 == 0 and time.perf_counter() - self.t0 > self.limit:
            raise Timeout()

    # ---------- AC-3 ----------
    def ac3(self, D, queue=None):
        """Lan truyền ràng buộc ≠ ; D sửa tại chỗ. Trả False nếu có miền rỗng."""
        q = deque(ARCS if queue is None else queue)
        st = self.st
        while q:
            xi, xj = q.popleft()
            st.revise += 1
            dj = D[xj]
            # ràng buộc xi != xj: giá trị của xi mất hỗ trợ chỉ khi Dj = {v}
            if dj & (dj - 1) == 0 and D[xi] & dj:
                D[xi] &= ~dj
                st.prune += 1
                if self.trace is not None:
                    self.trace.append(("PRUNE", xi, val(dj), xj))
                if D[xi] == 0:
                    return False
                if D[xi] & (D[xi] - 1) == 0:
                    for xk in PEERS[xi]:
                        if xk != xj:
                            q.append((xk, xi))
        return True

    # ---------- A: backtracking thuần ----------
    def bt_plain(self, grid, count=False, domains=None):
        """Duyệt ô theo thứ tự cố định (hàng-cột), chỉ kiểm tra xung đột khi gán."""
        g = list(grid)
        empties = [i for i in range(81) if g[i] == 0]
        st = self.st
        sols = []
        def rec(k):
            if k == len(empties):
                sols.append(list(g)); return not count or len(sols) >= 2
            i = empties[k]
            used = 0
            for j in PEERS[i]:
                if g[j]: used |= bit(g[j])
            cand = (domains[i] if domains else ALL) & ~used
            if cand == 0:
                st.backtrack += 1; return False
            for v in range(1, 10):
                if cand & bit(v):
                    self._check_time()
                    g[i] = v; st.assign += 1
                    if self.trace is not None: self.trace.append(("ASSIGN", i, v, None))
                    if rec(k + 1): return True
                    g[i] = 0
                    if self.trace is not None: self.trace.append(("BACKTRACK", i, v, None))
            st.backtrack += 1
            return False
        rec(0)
        return sols

    # ---------- C/D: MAC (AC-3 sau mỗi lần gán), tùy chọn MRV ----------
    def mac(self, D, mrv, count=False, rand=None):
        st = self.st
        sols = []
        def pick(D):
            best, bs = -1, 99
            for i in range(81):
                p = popcount(D[i])
                if p > 1:
                    if not mrv: return i
                    if p < bs: best, bs = i, p
                    if p == 2: break
            return best
        def rec(D):
            i = pick(D)
            if i < 0:
                sols.append(list(map(val, D))); return not count or len(sols) >= 2
            vals = [v for v in range(1, 10) if D[i] & bit(v)]
            if rand: rand.shuffle(vals)
            for v in vals:
                self._check_time()
                D2 = D[:]; D2[i] = bit(v); st.assign += 1
                if self.trace is not None: self.trace.append(("ASSIGN", i, v, None))
                if self.ac3(D2, [(k, i) for k in PEERS[i]]):
                    if rec(D2): return True
                if self.trace is not None: self.trace.append(("BACKTRACK", i, v, None))
            st.backtrack += 1
            return False
        rec(D)
        return sols

    def init_domains(self, grid):
        D = [ALL] * 81
        for i, v in enumerate(grid):
            if v: D[i] = bit(v)
        return D

def verify(grid, sol):
    if any(grid[i] and grid[i] != sol[i] for i in range(81)): return False
    for i in range(81):
        if any(sol[i] == sol[j] for j in PEERS[i]): return False
    return all(1 <= x <= 9 for x in sol)

# ------------- giao diện chạy chung cho 4 phiên bản -------------
METHODS = {
  "A": "Backtracking thuần",
  "B": "AC-3 (tiền xử lý) + Backtracking",
  "C": "MAC (AC-3 sau mỗi lần gán)",
  "D": "MAC + MRV",
}
def solve(grid, method, limit=10.0, trace=None):
    s = Solver(limit, trace); st = s.st
    s.t0 = time.perf_counter()
    sol = None
    try:
        if method == "A":
            r = s.bt_plain(grid)
        else:
            D = s.init_domains(grid)
            ok = s.ac3(D)
            if not ok: r = []
            elif method == "B":
                r = s.bt_plain(grid, domains=D)
            elif method == "C": r = s.mac(D, mrv=False)
            else: r = s.mac(D, mrv=True)
        if r: sol = r[0]
    except Timeout:
        st.timeout = True
    st.t = time.perf_counter() - s.t0
    st.solved = sol is not None and verify(grid, sol)
    return sol, st

def count_solutions(grid, cap=2):
    s = Solver(); D = s.init_domains(grid)
    if not s.ac3(D): return 0
    return len(s.mac(D, mrv=True, count=True))

def ac3_only(grid):
    """Trả về (hợp lệ?, số ô xác định chỉ nhờ AC-3, tổng ô trống)."""
    s = Solver(); D = s.init_domains(grid)
    ok = s.ac3(D)
    if not ok: return False, 0, sum(1 for x in grid if x == 0)
    det = sum(1 for i in range(81) if grid[i] == 0 and popcount(D[i]) == 1)
    return True, det, sum(1 for x in grid if x == 0)

# ---------------- sinh đề ----------------
def full_grid(rng):
    s = Solver(); D = s.init_domains([0] * 81)
    r = s.mac(D, mrv=True, rand=rng)
    return r[0]

def make_puzzle(rng, target_clues):
    sol = full_grid(rng); g = sol[:]
    cells = list(range(81)); rng.shuffle(cells)
    clues = 81
    for i in cells:
        if clues <= target_clues: break
        old = g[i]; g[i] = 0
        if count_solutions(g) != 1: g[i] = old
        else: clues -= 1
    return g, sol
