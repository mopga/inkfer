"""Vision checks of a loaded Inkling profile through the OpenAI-compatible API (images as data: URLs, as agents send them).

    python vischeck.py [PORT] [--long TOKENS] [--only NAME[,NAME]]

Short checks (each a fresh conversation, answers graded by the expected facts):
  natural     one photo-like image: the mailbox number and the sun's side
  chart       a chart: title, count of circles, relative position
  compare     two images in one message: counts in each and the added shape
  term        terminal screenshot: file, line and exception of a traceback
  table       dashboard screenshot: the degraded service, its port and p95
  code        editor screenshot: the bug (count += 1 outside the loop body) and its line
  dense       1920x1080 screen of 14 px text: the one annotated row of 120
  tool        a tool call whose argument has to be read from the image, then the tool result -> answer
  followup    a second turn about an image sent in the first turn (prefix cache with an image in the history)
--long N adds two checks with about N tokens of text: image first then N tokens then the question, and N tokens
then image and question. VRAM is sampled with nvidia-smi during each request. Results: logs\\vischeck-PORT.json.
"""
import base64, json, os, re, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.join(HERE, "src", "ninfer-all", "examples", "cli", "media")
VIS = os.path.join(HERE, "bench", "vision")
args = sys.argv[1:]
PORT = int(args[0]) if args and args[0].isdigit() else 8081
LONG = int(args[args.index("--long") + 1]) if "--long" in args else 0
ONLY = set(args[args.index("--only") + 1].split(",")) if "--only" in args else None
BASE = f"http://127.0.0.1:{PORT}"
MODEL = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=10).read())["data"][0]["id"]
RESULTS = []


def img(path):
    with open(path, "rb") as f:
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(f.read()).decode()}}


def text(s):
    return {"type": "text", "text": s}


class VramPeak:
    def __enter__(self):
        self.peak, self.stop = 0, False
        def run():
            while not self.stop:
                try:
                    v = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                           capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0])
                    self.peak = max(self.peak, v)
                except Exception:
                    pass
                time.sleep(0.25)
        self.t = threading.Thread(target=run, daemon=True); self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True; self.t.join(2)


def chat(messages, tools=None, max_tokens=4000, timeout=3600):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"reasoning": "", "content": "", "tools": {}, "usage": None, "ttft": None, "finish": None}
    t0 = time.time()
    with VramPeak() as vp, urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                out["usage"] = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if out["ttft"] is None and (delta.get("reasoning_content") or delta.get("content") or delta.get("tool_calls")):
                    out["ttft"] = round(time.time() - t0, 2)
                out["reasoning"] += delta.get("reasoning_content") or ""
                out["content"] += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    t = out["tools"].setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                    t["id"] = tc.get("id") or t["id"]
                    fn = tc.get("function") or {}
                    t["name"] += fn.get("name") or ""
                    t["args"] += fn.get("arguments") or ""
                out["finish"] = ch.get("finish_reason") or out["finish"]
    out["total"] = round(time.time() - t0, 2)
    out["vram_peak_mib"] = vp.peak
    return out


def record(name, out, checks, note=""):
    ok = all(checks.values())
    u = out.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    RESULTS.append(dict(name=name, ok=ok, checks=checks, prompt=u.get("prompt_tokens"), cached=cached,
                        output=u.get("completion_tokens"), ttft=out.get("ttft"), total=out.get("total"),
                        vram_peak_mib=out.get("vram_peak_mib"), answer=out["content"][:600],
                        tools=list(out["tools"].values()), note=note))
    bad = [k for k, v in checks.items() if not v]
    print(f"{'PASS' if ok else 'FAIL'} {name:9s} prompt {u.get('prompt_tokens')} cached {cached} ttft {out.get('ttft')} s"
          f" total {out.get('total')} s vram {out.get('vram_peak_mib')} MiB {'missing: ' + ', '.join(bad) if bad else ''}")
    print("     ", re.sub(r"\s+", " ", out["content"])[:300])
    sys.stdout.flush()


def has(s, *pats):
    return all(re.search(p, s, re.I | re.S) for p in pats)


def want(name):
    return ONLY is None or name in ONLY


if want("natural"):
    o = chat([{"role": "user", "content": [img(os.path.join(MEDIA, "natural_scene.png")),
              text("Опиши сцену одним предложением. Затем ответь: какое число на почтовом ящике и с какой стороны кадра солнце?")]}])
    record("natural", o, {"24": has(o["content"], r"\b24\b"), "right": has(o["content"], r"справ|\bправ(ом|ой|ая|ый|ую)\b|right")})

if want("chart"):
    o = chat([{"role": "user", "content": [img(os.path.join(MEDIA, "visual_chart.png")),
              text("Прочитай заголовок, посчитай красные круги и скажи, с какой стороны от зелёного треугольника синий квадрат. Ответ: заголовок; число; сторона.")]}])
    record("chart", o, {"title": has(o["content"], r"NIFER\s*VISION\s*731"), "3": has(o["content"], r"\b3\b|три"),
                        "left": has(o["content"], r"слева|лев|left")})

if want("compare"):
    o = chat([{"role": "user", "content": [img(os.path.join(MEDIA, "compare_left.png")), img(os.path.join(MEDIA, "compare_right.png")),
              text("Сравни две картинки: сколько красных кругов на первой и на второй и какая жёлтая фигура появилась на второй? Ответ: число; число; фигура.")]}])
    record("compare", o, {"2": has(o["content"], r"\b2\b|два|две"), "3": has(o["content"], r"\b3\b|три"),
                          "star": has(o["content"], r"звезд|звёзд|star")})

if want("term"):
    o = chat([{"role": "user", "content": [img(os.path.join(VIS, "term.png")),
              text("Это скриншот терминала. В каком файле и на какой строке упала программа, какое исключение и с каким значением?")]}])
    record("term", o, {"totals.py": has(o["content"], r"totals\.py"), "214": has(o["content"], r"\b214\b"),
                       "KeyError": has(o["content"], r"KeyError"), "KZT": has(o["content"], r"KZT")})

if want("table"):
    o = chat([{"role": "user", "content": [img(os.path.join(VIS, "table.png")),
              text("Какой сервис в состоянии degraded, на каком он порту, какое у него p95 и сколько ошибок в час? Какой сервис остановлен?")]}])
    record("table", o, {"orders": has(o["content"], r"orders"), "9100": has(o["content"], r"9100"), "388": has(o["content"], r"388"),
                        "57": has(o["content"], r"\b57\b"), "reports": has(o["content"], r"reports")})

if want("code"):
    o = chat([{"role": "user", "content": [img(os.path.join(VIS, "code.png")),
              text("Найди ошибку в функции на скриншоте: на какой строке (номер слева) и как исправить?")]}])
    record("code", o, {"line 39": has(o["content"], r"\b39\b"), "count": has(o["content"], r"count\s*\+=\s*1"),
                       "indent": has(o["content"], r"отступ|внутр|цикл|indent|loop")})

if want("dense"):
    o = chat([{"role": "user", "content": [img(os.path.join(VIS, "dense.png")),
              text("На скриншоте 120 строк склада. У одной строки есть пометка. Какой это item, какой у него sku, qty и bin, и что за пометка?")]}])
    record("dense", o, {"077": has(o["content"], r"077"), "K90816": has(o["content"], r"K90816"), "13": has(o["content"], r"\b13\b"),
                        "R05": has(o["content"], r"R05"), "ORLOVA": has(o["content"], r"ORLOVA|Орлов")})

TOOLS = [{"type": "function", "function": {
    "name": "lookup_rate", "description": "Курс валюты к евро на дату.",
    "parameters": {"type": "object", "properties": {"currency": {"type": "string", "description": "код ISO 4217"},
                                                     "date": {"type": "string", "description": "YYYY-MM-DD"}},
                   "required": ["currency"]}}}]
if want("tool"):
    msgs = [{"role": "system", "content": "Ты агент. Если нужны данные, вызывай инструменты."},
            {"role": "user", "content": [img(os.path.join(VIS, "term.png")),
             text("Программа упала из-за отсутствующей валюты в таблице курсов. Узнай курс этой валюты к евро инструментом lookup_rate и скажи, какую строку добавить в RATES.")]}]
    o = chat(msgs, tools=TOOLS)
    calls = list(o["tools"].values())
    arg_ok = bool(calls) and has(calls[0]["args"], r"KZT")
    record("tool", o, {"tool called": bool(calls) and calls[0]["name"] == "lookup_rate", "currency KZT": arg_ok})
    if calls:
        msgs.append({"role": "assistant", "content": o["content"] or None,
                     "tool_calls": [{"id": calls[0]["id"] or "call_0", "type": "function",
                                     "function": {"name": calls[0]["name"], "arguments": calls[0]["args"]}}]})
        msgs.append({"role": "tool", "tool_call_id": calls[0]["id"] or "call_0", "content": json.dumps({"currency": "KZT", "eur": 0.00181})})
        o2 = chat(msgs, tools=TOOLS)
        record("tool-2", o2, {"rate used": has(o2["content"], r"0[.,]00181"), "KZT": has(o2["content"], r"KZT")})

if want("followup"):
    m1 = [{"role": "user", "content": [img(os.path.join(VIS, "table.png")), text("Перечисли сервисы на скриншоте с их портами.")]}]
    o = chat(m1)
    record("follow-1", o, {"gateway 8443": has(o["content"], r"gateway.*8443"), "mailer 9400": has(o["content"], r"mailer.*9400")})
    m1 += [{"role": "assistant", "content": o["content"]},
           {"role": "user", "content": "У какого из них больше всего ошибок в час и во сколько раз его p95 больше, чем у auth?"}]
    o2 = chat(m1)
    record("follow-2", o2, {"orders": has(o2["content"], r"orders"), "~22.8x": has(o2["content"], r"22[.,][78]|23")},
           note="cached tokens show whether the image turn was reused from the cache")


def filler(n_tokens):
    # About 9 tokens per line; numbered notes so that nothing repeats.
    lines = [f"Заметка {i}: склад {i % 17}, партия {i * 31 % 997}, ответственный смены {i % 5}, статус проверен." for i in range(n_tokens // 22)]
    return "\n".join(lines)


if LONG:
    big = filler(LONG)
    o = chat([{"role": "user", "content": [img(os.path.join(VIS, "dense.png")), text("Запомни этот скриншот склада. Ниже журнал заметок.\n" + big
              + "\n\nВопрос по скриншоту из начала сообщения: у какой строки есть пометка, какие у неё sku и qty?")]}])
    record("long-img-first", o, {"077": has(o["content"], r"077"), "K90816": has(o["content"], r"K90816"), "13": has(o["content"], r"\b13\b")},
           note=f"image, then about {LONG} tokens of text, then the question")
    o = chat([{"role": "user", "content": [text("Журнал заметок:\n" + big + "\n\nТеперь скриншот терминала:"), img(os.path.join(VIS, "term.png")),
              text("В каком файле и на какой строке упала программа и почему? И сколько заметок в журнале выше (номер последней)?")]}])
    last = LONG // 22 - 1
    record("long-img-last", o, {"totals.py": has(o["content"], r"totals\.py"), "214": has(o["content"], r"\b214\b"),
                                "KZT": has(o["content"], r"KZT"), f"last note {last}": has(o["content"], rf"\b{last}\b|\b{last + 1}\b")},
           note=f"about {LONG} tokens of text, then the image and the question")

path = os.path.join(HERE, "logs", f"vischeck-{PORT}{'-long' + str(LONG) if LONG else ''}.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(dict(model=MODEL, results=RESULTS), f, ensure_ascii=False, indent=1)
print(f"{sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} passed; {path}")
