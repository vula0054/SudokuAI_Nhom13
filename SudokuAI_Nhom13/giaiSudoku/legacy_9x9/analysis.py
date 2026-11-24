import json, csv, statistics as S
from csp import *
from refpuzzles import REF

# 1) đề tham chiếu
rows = []
for n, s in REF.items():
    g = parse(s)
    for m in "ABCD":
        sol, st = solve(g, m, 30)
        rows.append(dict(name=n, method=m, clues=sum(1 for x in g if x), solved=st.solved, timeout=st.timeout,
                         time=st.t, assign=st.assign, backtrack=st.backtrack, revise=st.revise))
with open("results/ref.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)

# 2) AC-3 một mình trên đề sinh ngẫu nhiên
P = json.load(open("results/puzzles.json")); ac = {}
for lvl in ["Dễ", "Trung bình", "Khó"]:
    ps = [parse(p["grid"]) for p in P if p["level"] == lvl]; solved = 0; fr = []
    for g in ps:
        ok, det, emp = ac3_only(g); fr.append(det / emp)
        if det == emp: solved += 1
    ac[lvl] = dict(n=len(ps), solved_by_ac3=solved, mean_det=S.mean(fr))
for n, s in REF.items():
    g = parse(s); ok, det, emp = ac3_only(g); ac[n] = dict(n=1, solved_by_ac3=int(det == emp), mean_det=det / emp)
json.dump(ac, open("results/ac3_only.json", "w"), ensure_ascii=False, indent=1)

# 3) đầu vào lỗi
edge = {}
bad = parse("55..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79")  # hai số 5 cùng hàng
s = Solver(); D = s.init_domains(bad)
# AC-3 phát hiện: ô gán 5 trùng hàng làm miền của ô kia rỗng
ok = s.ac3(D); edge["xung đột ngay từ đề (hai số 5 cùng hàng)"] = dict(ac3_phat_hien=not ok, revise=s.st.revise)
# đề vô nghiệm "tinh vi": hợp lệ cục bộ nhưng không có nghiệm
u = parse("." * 81); u[0:8] = [1,2,3,4,5,6,7,8]; u[17] = 9      # ô (0,8) chỉ có thể là 9 nhưng cột/khối ... cố tình ép
sol, st = solve(u, "D", 10); edge["vô nghiệm kiểu ép miền (8 số hàng 1 + số 9 ở ô (2,8))"] = dict(co_nghiem=sol is not None, backtrack=st.backtrack, assign=st.assign)
empty = parse("." * 81); n_sol = count_solutions(empty)
edge["đề rỗng (nhiều nghiệm)"] = dict(so_nghiem_dem_den_2=n_sol)
json.dump(edge, open("results/edge.json", "w"), ensure_ascii=False, indent=1)

# 4) log suy luận (MAC+MRV) trên AI Escargot
tr = []; g = parse(REF["AI Escargot"]); sol, st = solve(g, "D", 10, trace=tr)
def rc(i): return f"({i//9+1},{i%9+1})"
lines = []
for e in tr:
    if e[0] == "ASSIGN": lines.append(f"ASSIGN   {rc(e[1])} = {e[2]}")
    elif e[0] == "PRUNE": lines.append(f"PRUNE    {rc(e[1])} loại {e[2]} vì {rc(e[3])} = {e[2]}")
    else: lines.append(f"BACKTRACK {rc(e[1])} thử {e[2]} thất bại")
open("results/trace_example.txt", "w").write("\n".join(lines))
