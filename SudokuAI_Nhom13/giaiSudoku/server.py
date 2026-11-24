"""Web UI: python server.py  →  http://localhost:8000  (server dùng thư viện chuẩn; nhận dạng ảnh cần opencv-python, scikit-learn)"""
import json, random, os, base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from csp import *
HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["Dễ", "Trung bình", "Khó"]

def spec_of(d):
    n, v = int(d.get("n", 9)), d.get("variant", "classic")
    if v not in VARIANTS: raise ValueError("Biến thể không tồn tại")
    if n not in SIZES[v]: raise ValueError(f"{VARIANTS[v]} chỉ hỗ trợ cỡ {', '.join(f'{k}×{k}' for k in SIZES[v])}")
    if v == "jigsaw" and not d.get("regions"): raise ValueError("Jigsaw cần bản đồ vùng: hãy bấm 'Sinh đề' trước")
    if v == "killer" and not d.get("cages"): raise ValueError("Killer cần các lồng: hãy bấm 'Sinh đề' trước")
    cages = [(c["cells"], c["sum"]) for c in d["cages"]] if v == "killer" else None
    return get_spec(n, v, d.get("regions") if v == "jigsaw" else None, cages)

def api(path, d):
    if path == "/api/generate":
        n, v = int(d.get("n", 9)), d.get("variant", "classic"); rng = random.Random(d.get("seed"))
        if v not in VARIANTS: raise ValueError("Biến thể không tồn tại")
        sp, g, sol, reg, cages = generate(n, v, LEVELS.index(d.get("level", "Trung bình")), rng)
        return dict(grid=g, box=sp.box_of, n=n, variant=v, regions=reg, suggest=choose_method(sp, g),
                    cages=[dict(cells=list(c), sum=t) for c, t in sp.cages] if v == "killer" else None)
    if path == "/api/recognize":
        from vision import recognize
        raw = d["image"].split(",", 1)[-1]; r = recognize(base64.b64decode(raw), int(d.get("n", 9)))
        spc = get_spec(r["n"], "classic"); r["check"] = check(spc, r["grid"], 5); r["suggest"] = choose_method(spc, r["grid"]); return r
    sp = spec_of(d); g = parse_grid(d["grid"])
    if len(g) != sp.N: raise ValueError("Sai kích thước lưới")
    if any(x < 0 or x > sp.n for x in g): raise ValueError(f"Giá trị phải nằm trong 1–{sp.n}")
    if path == "/api/suggest": return dict(suggest=choose_method(sp, g))
    if path == "/api/check": return dict(check(sp, g), box=sp.box_of, suggest=choose_method(sp, g))
    if path == "/api/solve":
        m = d.get("method", "auto")
        if m != "auto" and m not in METHODS: raise ValueError("Thuật toán không tồn tại")
        tr = []; sol, st = solve(sp, g, m, float(d.get("timeout", 10)), tr)
        return dict(solution=sol, trace=tr, box=sp.box_of, solved=st.solved, timeout=st.timeout,
                    method=st.method, method_name=METHODS[st.method], reason=st.reason, auto=(m == "auto"), fallback=st.fallback,
                    stats=dict(ms=round(st.t*1000, 2), assign=st.assign, backtrack=st.backtrack,
                               revise=st.revise, prune=st.prune), check=check(sp, g, 5))
    raise ValueError("API không tồn tại")

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ct="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", ct + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/api/eval":
            f = os.path.join(HERE, "results", "eval.json")
            return self._send(200, open(f, "rb").read() if os.path.exists(f) else {"error": "Chưa chạy python evaluate.py"})
        self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html")
    def do_POST(self):
        try: d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"); self._send(200, api(self.path, d))
        except Exception as e: self._send(400, {"error": str(e)})
    def log_message(self, *a): pass

if __name__ == "__main__":
    print("Mở http://localhost:8000"); ThreadingHTTPServer(("", 8000), H).serve_forever()
