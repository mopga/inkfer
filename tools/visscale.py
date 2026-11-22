"""Does enlarging a screenshot before the vision tower help Inkling read it? Repeated, graded runs per scale.

    python visscale.py [PORT] [--scales 1,1.5,2] [--reps 3] [--images code,dense,term,table]

The engine gives an image one cell per 40x40 pixels and never enlarges it, so small text shares cells. Each run sends
the scaled PNG with the same question as vischeck.py and grades the answer by the same facts. Prints per image and scale:
passes, mean prompt tokens, mean output tokens, mean time. Results: logs\\visscale-PORT.json.
"""
import base64, io, json, os, re, sys, time, urllib.request
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIS = os.path.join(HERE, "bench", "vision")
a = sys.argv[1:]
PORT = int(a[0]) if a and a[0].isdigit() else 8081
SCALES = [float(s) for s in a[a.index("--scales") + 1].split(",")] if "--scales" in a else [1.0, 1.5, 2.0]
REPS = int(a[a.index("--reps") + 1]) if "--reps" in a else 3
IMAGES = a[a.index("--images") + 1].split(",") if "--images" in a else ["code", "dense", "term", "table"]
BASE = f"http://127.0.0.1:{PORT}"
MODEL = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=10).read())["data"][0]["id"]

CASES = {
    "code": ("Найди ошибку в функции на скриншоте: на какой строке (номер слева) и как исправить?",
             [r"\b39\b", r"count\s*\+=\s*1"]),
    "dense": ("На скриншоте 120 строк склада. У одной строки есть пометка. Какой это item, какой у него sku, qty и bin, и что за пометка?",
              [r"077", r"K90816", r"\b13\b", r"R05", r"ORLOVA|Орлов"]),
    "term": ("Это скриншот терминала. В каком файле и на какой строке упала программа, какое исключение и с каким значением?",
             [r"totals\.py", r"\b214\b", r"KeyError", r"KZT"]),
    "table": ("Какой сервис в состоянии degraded, на каком он порту, какое у него p95 и сколько ошибок в час? Какой сервис остановлен?",
              [r"orders", r"9100", r"388", r"\b57\b", r"reports"]),
}


def ask(im, q):
    b = io.BytesIO(); im.save(b, "PNG")
    body = {"model": MODEL, "max_tokens": 8000, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}},
        {"type": "text", "text": q}]}]}
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=1200).read())
    return r["usage"], r["choices"][0]["message"].get("content") or "", time.time() - t0


rows = []
for name in IMAGES:
    q, facts = CASES[name]
    im = Image.open(os.path.join(VIS, name + ".png")).convert("RGB")
    for s in SCALES:
        x = im if s == 1 else im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
        runs = []
        for _ in range(REPS):
            u, ans, t = ask(x, q)
            ok = all(re.search(p, ans, re.I) for p in facts)
            runs.append(dict(ok=ok, prompt=u["prompt_tokens"], out=u["completion_tokens"], s=round(t, 1), answer=ans[:300]))
        row = dict(image=name, scale=s, size=f"{x.width}x{x.height}", passed=sum(r["ok"] for r in runs), reps=REPS,
                   prompt=round(sum(r["prompt"] for r in runs) / REPS), out=round(sum(r["out"] for r in runs) / REPS),
                   seconds=round(sum(r["s"] for r in runs) / REPS, 1), runs=runs)
        rows.append(row)
        print(f"{name:6s} x{s:<4} {row['size']:>10s}: {row['passed']}/{REPS} pass, prompt {row['prompt']}, out {row['out']}, {row['seconds']} s")
        sys.stdout.flush()
path = os.path.join(HERE, "logs", f"visscale-{PORT}.json")
json.dump(dict(model=MODEL, rows=rows), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(path)
