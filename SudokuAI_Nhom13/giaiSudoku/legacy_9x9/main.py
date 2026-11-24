"""Giao diện dòng lệnh: giải Sudoku và hiển thị từng bước suy luận.
Ví dụ: python main.py --puzzle "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79" --method D --steps 15
"""
import argparse
from csp import *

def show(g):
    out = []
    for r in range(9):
        if r % 3 == 0: out.append("+-------+-------+-------+")
        row = "|"
        for c in range(9):
            row += " " + (str(g[r*9+c]) if g[r*9+c] else ".")
            if c % 3 == 2: row += " |"
        out.append(row)
    out.append("+-------+-------+-------+"); return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", required=True, help="81 ký tự, '.' hoặc 0 là ô trống")
    ap.add_argument("--method", default="D", choices=list("ABCD"))
    ap.add_argument("--steps", type=int, default=20, help="số sự kiện suy luận in ra")
    ap.add_argument("--timeout", type=float, default=30)
    a = ap.parse_args()
    g = parse(a.puzzle)
    if len(g) != 81: raise SystemExit("Đề phải có đúng 81 ký tự")
    print("ĐỀ BÀI:\n" + show(g))
    ok, det, emp = ac3_only(g)
    if not ok: raise SystemExit("AC-3 phát hiện đề mâu thuẫn (có miền rỗng): đề vô nghiệm.")
    print(f"\nAC-3: xác định {det}/{emp} ô trống chỉ bằng lan truyền ràng buộc")
    n = count_solutions(g); print(f"Số nghiệm (đếm tới 2): {n}")
    trace = []; sol, st = solve(g, a.method, a.timeout, trace)
    print(f"\nPhương pháp {a.method} - {METHODS[a.method]}")
    if sol is None: raise SystemExit("Không tìm được nghiệm" + (" (quá thời gian)" if st.timeout else ""))
    print(show(sol))
    print(f"Thời gian {st.t*1000:.2f} ms | gán {st.assign} | quay lui {st.backtrack} | revise {st.revise} | loại giá trị {st.prune}")
    rc = lambda i: f"({i//9+1},{i%9+1})"
    print(f"\n{a.steps} bước suy luận đầu tiên:")
    for e in trace[:a.steps]:
        if e[0] == "ASSIGN": print(f"  ASSIGN    {rc(e[1])} = {e[2]}")
        elif e[0] == "PRUNE": print(f"  PRUNE     {rc(e[1])} loại {e[2]} vì {rc(e[3])} = {e[2]}")
        else: print(f"  BACKTRACK {rc(e[1])} thử {e[2]} thất bại")
if __name__ == "__main__": main()
