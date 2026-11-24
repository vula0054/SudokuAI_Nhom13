import random, json, csv
from multiprocessing import Pool
from csp import *

LEVELS = {"Dễ": 40, "Trung bình": 32, "Khó": 25}
N = 30; LIMIT = 10.0

def gen(args):
    lvl, k = args
    rng = random.Random(list(LEVELS).index(lvl) * 1000 + k)
    g, sol = make_puzzle(rng, LEVELS[lvl])
    return lvl, k, g

def run(args):
    lvl, k, g, m = args
    sol, st = solve(g, m, LIMIT)
    return dict(level=lvl, id=k, method=m, clues=sum(1 for x in g if x), solved=st.solved,
                timeout=st.timeout, time=st.t, assign=st.assign, backtrack=st.backtrack,
                revise=st.revise, prune=st.prune)

if __name__ == "__main__":
    with Pool() as p:
        puzzles = p.map(gen, [(l, k) for l in LEVELS for k in range(N)])
        json.dump([dict(level=l, id=k, grid=to_str(g)) for l, k, g in puzzles], open("results/puzzles.json", "w"), ensure_ascii=False)
        jobs = [(l, k, g, m) for l, k, g in puzzles for m in "ABCD"]
        rows = p.map(run, jobs, chunksize=1)
    with open("results/raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("done", len(rows))
