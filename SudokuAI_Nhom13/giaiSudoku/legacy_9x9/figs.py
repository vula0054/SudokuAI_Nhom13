import csv, json, statistics as S
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from csp import *
from refpuzzles import REF
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":"#52514e","axes.labelcolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
COL = {"A":"#2a78d6","B":"#eb6834","C":"#1baf7a","D":"#eda100"}
LAB = {"A":"A. Backtracking thuần","B":"B. AC-3 tiền xử lý + BT","C":"C. MAC","D":"D. MAC + MRV"}
rows = list(csv.DictReader(open("results/raw.csv"))); LV=["Dễ","Trung bình","Khó"]

def grouped(metric, ylabel, fn, title, log=True):
    fig, ax = plt.subplots(figsize=(7.2,3.8), dpi=200); w=0.19
    for k,m in enumerate("ABCD"):
        vals=[]
        for l in LV:
            v=[float(r[metric]) for r in rows if r["level"]==l and r["method"]==m]; vals.append(max(S.mean(v), 1e-3 if metric=="time" else 0.5))
        xs=[i+(k-1.5)*w for i in range(3)]
        b=ax.bar(xs, vals, w*0.92, color=COL[m], label=LAB[m])
        for x,v in zip(xs,vals): ax.text(x, v*1.12, f"{v:,.0f}" if metric!="time" else f"{v*1000:.1f}", ha="center", va="bottom", fontsize=6.5, color="#0b0b0b")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{l}\n(~{c} gợi ý)" for l,c in zip(LV,[40,32,25])])
    if log: ax.set_yscale("log")
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=10.5, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left"); ax.yaxis.grid(True, color="#e5e4e0", lw=.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(fn); plt.close(fig)
grouped("assign","Số lần gán biến (TB, log)","figs/fig_assign.png","Số lần gán biến trung bình theo mức độ khó (30 đề/mức)")
grouped("time","Thời gian giải (ms, TB, log)","figs/fig_time.png","Thời gian giải trung bình (ms) theo mức độ khó (30 đề/mức)")

ref=list(csv.DictReader(open("results/ref.csv"))); names=list(REF)
fig, ax = plt.subplots(figsize=(7.2,4), dpi=200); w=0.19
for k,m in enumerate("ABCD"):
    xs=[]; vs=[]; to=[]
    for i,n in enumerate(names):
        r=[x for x in ref if x["name"]==n and x["method"]==m][0]; xs.append(i+(k-1.5)*w); vs.append(max(int(r["assign"]),0.8)); to.append(r["timeout"]=="True")
    bars=ax.bar(xs, vs, w*0.92, color=COL[m], label=LAB[m])
    for b,t in zip(bars,to):
        if t: b.set_hatch("////"); b.set_edgecolor("#ffffff")
ax.set_yscale("log"); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Số lần gán biến (log)"); ax.set_title("Đề khó tham chiếu: số lần gán biến (cột gạch chéo = quá 30 giây, dừng sớm)", loc="left", fontsize=10.5, color="#0b0b0b")
ax.set_ylim(0.5,4e9); ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", columnspacing=1, handlelength=1.2); ax.yaxis.grid(True,color="#e5e4e0",lw=.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figs/fig_ref.png"); plt.close(fig)

# --- trực quan hóa lưới: đề / sau AC-3 / nghiệm ---
P=json.load(open("results/puzzles.json"))
cand=[parse(p["grid"]) for p in P if p["level"]=="Trung bình"]
g=[x for x in cand if (lambda r:(r[0] and 0<r[1]<r[2]))(ac3_only(x))][0]
s=Solver(); D=s.init_domains(g); s.ac3(D)
sol,_=solve(g,"D")
def draw(ax, cells, title):
    for i in range(81):
        r,c=divmod(i,9); t,colr=cells.get(i,("", "#000"))
        ax.text(c+.5, 8.5-r, t, ha="center", va="center", fontsize=11, color=colr, fontweight="bold" if colr=="#000" else "normal")
    for k in range(10):
        lw=2.2 if k%3==0 else .6
        ax.plot([0,9],[k,k],color="#0b0b0b",lw=lw); ax.plot([k,k],[0,9],color="#0b0b0b",lw=lw)
    ax.set_xlim(0,9); ax.set_ylim(0,9); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9.5)
fig,axs=plt.subplots(1,3,figsize=(9.6,3.5),dpi=200)
c1={i:(str(v),"#000") for i,v in enumerate(g) if v}
c2=dict(c1); c2.update({i:(str(val(D[i])),"#2a78d6") for i in range(81) if g[i]==0 and popcount(D[i])==1})
c3=dict(c2); c3.update({i:(str(sol[i]),"#eb6834") for i in range(81) if i not in c2})
n_ac=len(c2)-len(c1); n_se=len(c3)-len(c2)
draw(axs[0],c1,f"(a) Đề gốc: {len(c1)} gợi ý"); draw(axs[1],c2,f"(b) Sau AC-3: +{n_ac} ô (xanh)"); draw(axs[2],c3,f"(c) Nghiệm: +{n_se} ô do tìm kiếm (cam)")
fig.tight_layout(); fig.savefig("figs/fig_grid.png"); plt.close(fig)
