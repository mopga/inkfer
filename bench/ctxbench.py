"""Prefill / decode speed at a context that is really filled, for any OpenAI-compatible engine, with a system monitor.

  python ctxbench.py --base http://127.0.0.1:8090 --model inkling --label LABEL
                     [--fills 8000,32000,64000,128000,256000,512000] [--gen 500] [--turns 3] [--temp 0]
                     [--proc llama-server.exe] [--disk PhysicalDrive1] [--think on|off]

For every fill depth N (in the engine's own tokens, counted by the engine):
  cold    the document cut to N tokens + question A, nothing cached   -> prefill speed, time to first token, decode at N
  warm    the same document + question B                               -> cache hit, time to first token, decode at N
  turn1.. an agent-like dialog on top of the same document            -> time to first token of a short turn, decode
Timed on the client; when the engine reports its own timings (llama.cpp `timings`) they are stored next to the client's.
While a request runs, once a second: VRAM used, engine RSS / private bytes, available RAM, page file in use, CPU load,
disk read rate. Results: runs/LABEL.json (+ runs/LABEL.tsv with the monitor's samples).

Needs psutil: run with Strata's bundled python (D:\\MODELS\\Strata\\control\\python\\python.exe).
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

import psutil

CORPUS_DIR = Path(os.environ["LOCALAPPDATA"]) / "Strata" / "install" / "third_party" / "llama.cpp"
OUT = Path(__file__).parent / "runs"

Q_COLD = ("Выше приведены исходные файлы проекта. Напиши связный технический обзор на русском языке: архитектура, ключевые "
          "модули и структуры данных, как они взаимодействуют. Пиши развёрнуто, минимум 500 слов, без списков.")
Q_WARM = ("Based on the source files above, write a complete, well-commented Python module (at least 120 lines) that implements "
          "a minimal GGUF metadata reader with a command-line interface. Output only the code.")
FOLLOW = [
    "Коротко: в каком файле из приведённых выше реализован разбор аргументов командной строки и как называется главная функция?",
    "Покажи на Python функцию из 20–30 строк, которая читает заголовок GGUF-файла (магическое число, версия, число тензоров).",
    "Какие три места в коде выше ты бы проверил первыми, если генерация вдруг стала медленной? Кратко, по пунктам.",
    "Назови два файла выше, где работают с KV-кэшем, и одной фразой — что делает каждый.",
    "Сделай итог всего диалога в одном абзаце на русском.",
]
NEEDLE = "\n\n===== FILE: notes/ops.txt =====\nСлужебная заметка: кодовое слово развёртывания — «{word}», порт стенда — {port}.\n"


def corpus(max_chars):
    """Deterministic corpus: llama.cpp sources and docs in sorted order (the same text the earlier benches used)."""
    parts, total = [], 0
    exts = {".cpp", ".h", ".md", ".py", ".c", ".hpp", ".txt", ".cmake"}
    for root in ("src", "common", "tools", "docs", "examples", "ggml/src", "gguf-py", "include", "tests", "ggml/include"):
        base = CORPUS_DIR / root
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in exts or p.stat().st_size > 300_000:
                continue
            try:
                t = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            chunk = f"\n\n===== FILE: {p.relative_to(CORPUS_DIR).as_posix()} =====\n{t}"
            parts.append(chunk)
            total += len(chunk)
            if total >= max_chars:
                return "".join(parts)[:max_chars]
    return "".join(parts)[:max_chars]


def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def count_tokens(base, text):
    """The engine's own count (llama.cpp /tokenize). None when the engine has no such endpoint."""
    try:
        return len(post(base + "/tokenize", {"content": text}, timeout=600)["tokens"])
    except Exception:
        return None


class Monitor(threading.Thread):
    def __init__(self, proc_names, disk, tsv):
        super().__init__(daemon=True)
        self.names = [n.lower() for n in proc_names]
        self.disk, self.tsv = disk, tsv
        self.samples, self.mark, self.stop_flag = [], "", False
        self.lock = threading.Lock()

    def engine_mem(self):
        rss = priv = 0
        for p in psutil.process_iter(["name"]):
            try:
                if (p.info["name"] or "").lower() in self.names:
                    m = p.memory_info()
                    rss += m.rss
                    priv += getattr(m, "private", 0)
            except psutil.Error:
                pass
        return rss, priv

    def run(self):
        prev, prev_t = psutil.disk_io_counters(perdisk=True), time.time()
        psutil.cpu_percent(None)
        with open(self.tsv, "w", encoding="utf-8") as f:
            f.write("t\tmark\tvram_used_mib\tgpu_pct\tengine_rss_gib\tengine_private_gib\tram_avail_gib\tpagefile_used_gib\tcpu_pct\tdisk_read_MBps\n")
            while not self.stop_flag:
                t = time.time()
                try:
                    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=10).stdout.strip().split(", ")
                    vram, gpu = int(smi[0]), int(smi[1])
                except Exception:
                    vram, gpu = -1, -1
                now = psutil.disk_io_counters(perdisk=True)
                dt = max(time.time() - prev_t, 1e-3)
                rates = {k: (now[k].read_bytes - prev[k].read_bytes) / 1e6 / dt for k in now if k in prev}
                prev, prev_t = now, time.time()
                disk = rates.get(self.disk) if self.disk in rates else max(rates.values() or [0])
                rss, priv = self.engine_mem()
                s = {"t": t, "mark": self.mark, "vram": vram, "gpu": gpu, "rss": rss / 2**30, "priv": priv / 2**30,
                     "avail": psutil.virtual_memory().available / 2**30, "swap": psutil.swap_memory().used / 2**30,
                     "cpu": psutil.cpu_percent(None), "disk": disk}
                with self.lock:
                    self.samples.append(s)
                f.write(f"{time.strftime('%H:%M:%S')}\t{s['mark']}\t{vram}\t{gpu}\t{s['rss']:.2f}\t{s['priv']:.2f}\t{s['avail']:.2f}\t"
                        f"{s['swap']:.2f}\t{s['cpu']:.0f}\t{disk:.0f}\n")
                f.flush()
                time.sleep(max(0.0, 1.0 - (time.time() - t)))

    def window(self, t0, t1):
        with self.lock:
            w = [s for s in self.samples if t0 - 1.0 <= s["t"] <= t1 + 1.0]
        if not w:
            return {}
        return {"vram_used_mib_max": max(s["vram"] for s in w), "gpu_pct_avg": round(sum(s["gpu"] for s in w) / len(w)),
                "engine_rss_gib_max": round(max(s["rss"] for s in w), 2), "engine_private_gib_max": round(max(s["priv"] for s in w), 2),
                "ram_avail_gib_min": round(min(s["avail"] for s in w), 2), "pagefile_used_gib_max": round(max(s["swap"] for s in w), 2),
                "cpu_pct_avg": round(sum(s["cpu"] for s in w) / len(w)), "cpu_pct_max": round(max(s["cpu"] for s in w)),
                "disk_read_MBps_avg": round(sum(s["disk"] for s in w) / len(w)), "disk_read_MBps_max": round(max(s["disk"] for s in w))}


def chat(base, model, messages, max_tokens, temp, extra, timeout=7200):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temp, "stream": True,
            "stream_options": {"include_usage": True}}
    body.update(extra)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = None
    usage, timings, finish, reasoning, content, tool_calls = None, None, None, [], [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                rc = d.get("reasoning_content") or d.get("reasoning") or ""
                c = d.get("content") or ""
                if d.get("tool_calls"):
                    tool_calls += 1
                if rc or c or d.get("tool_calls"):
                    if t_first is None:
                        t_first = time.time()
                    reasoning.append(rc)
                    content.append(c)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.time()
    usage = usage or {}
    gen = usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None and timings and timings.get("cache_n") is not None:
        cached = timings["cache_n"]
    rec = {"t0": t0, "t1": t_end, "ttft_s": round((t_first or t_end) - t0, 2), "total_s": round(t_end - t0, 2),
           "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached, "completion_tokens": gen, "finish": finish}
    if gen and t_first and t_end > t_first and gen > 1:
        rec["decode_tps_client"] = round((gen - 1) / (t_end - t_first), 1)
    if usage.get("prompt_tokens") is not None and cached is not None:
        rec["fresh_tokens"] = usage["prompt_tokens"] - cached
        if rec["ttft_s"] > 0:
            rec["prefill_tps_client"] = round(rec["fresh_tokens"] / rec["ttft_s"])
    if timings:
        rec["engine"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in timings.items()
                         if k in ("cache_n", "prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
                                  "predicted_per_second", "draft_n", "draft_n_accepted")}
    return rec, "".join(reasoning), "".join(content)


def fit_doc(base, text, target, cpt):
    """Cut the corpus so that the document is `target` tokens by the engine's own count (within 1%)."""
    n_chars = int(target * cpt)
    for _ in range(6):
        doc = "Исходные файлы проекта:\n" + text[:n_chars] + "\n\n===== END OF FILES =====\n\n"
        n = count_tokens(base, doc)
        if n is None:
            return doc, None, cpt
        cpt = n_chars / n
        if abs(n - target) <= target * 0.01:
            return doc, n, cpt
        n_chars = int(target * cpt)
        if n_chars > len(text):
            sys.exit(f"corpus too short for {target} tokens: {len(text)} chars")
    return doc, n, cpt


def loopiness(text):
    """Share of repeated 6-word shingles in a long answer: ~0 for normal prose, towards 1 when generation loops."""
    w = text.split()
    sh = [" ".join(w[i:i + 6]) for i in range(max(0, len(w) - 5))]
    return round(1 - len(set(sh)) / len(sh), 3) if sh else None


def lean_fill(fill, doc, run, save, a):
    facts = []
    for k, frac in enumerate((0.9, 0.5, 0.1)):          # insert from the end so earlier offsets stay valid
        word, port = f"{('орион', 'вега', 'сириус')[k]}-{(fill + 131 * k) % 977}", 20000 + (fill + 577 * k) % 9973
        cut = doc.find("\n\n===== FILE:", int(len(doc) * frac))
        if cut < 0:
            cut = doc.rfind("\n\n===== FILE:")
        note = (f"\n\n===== FILE: notes/ops-{k + 1}.txt =====\nСлужебная заметка №{k + 1}: кодовое слово развёртывания — «{word}», "
                f"порт стенда — {port}.\n")
        doc = doc[:cut] + note + doc[cut:]
        facts.append((word, str(port)))
    q = ("Среди файлов выше есть три служебные заметки (notes/ops-1.txt, notes/ops-2.txt, notes/ops-3.txt). Для каждой назови "
         "кодовое слово развёртывания и порт стенда. Ответь тремя строками, без пояснений.")
    messages = [{"role": "user", "content": doc + q}]
    rec, _, content = run("cold", fill, messages, a.turn_tokens)
    if rec.get("error"):
        return
    rec["needles_found"] = sum(1 for w, p in facts if w in content and p in content)
    rec["needle_ok"] = rec["needles_found"] == len(facts)
    rec["answer"] = content[:400]
    save()
    follow = [Q_COLD] + FOLLOW
    for t in range(a.turns):
        messages = messages + [{"role": "assistant", "content": content}, {"role": "user", "content": follow[t % len(follow)]}]
        rec, _, content = run("turn%d" % (t + 1), fill, messages, a.gen if t == 0 else a.turn_tokens)
        if rec.get("error"):
            break
        if t == 0:
            rec["loopiness"] = loopiness(content)
            save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="x")
    ap.add_argument("--label", required=True)
    ap.add_argument("--fills", default="8000,32000,64000,128000")
    ap.add_argument("--gen", type=int, default=500)
    ap.add_argument("--turn-tokens", type=int, default=300)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--cpt", type=float, default=3.0)
    ap.add_argument("--proc", default="llama-server.exe,strata.exe,basalt.exe,ninfer.exe")
    ap.add_argument("--disk", default="")
    ap.add_argument("--extra", default="{}", help="JSON merged into every request body")
    ap.add_argument("--note", default="")
    ap.add_argument("--lean", action="store_true",
                    help="deep-context mode, one cold read per fill: three planted facts (10/50/90 %% of the document) asked "
                         "in the cold request, then a long answer (--gen tokens) and short turns on top of the cached prefix")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    fills = [int(x) for x in a.fills.split(",") if x]
    extra = json.loads(a.extra)
    text = corpus(int(max(fills) * 4.6) + 200000)
    mon = Monitor(a.proc.split(","), a.disk, OUT / f"{a.label}.tsv")
    mon.start()
    time.sleep(2.5)
    idle = mon.window(time.time() - 2, time.time())
    result = {"label": a.label, "base": a.base, "note": a.note, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "temp": a.temp, "extra": extra, "idle": idle, "requests": []}

    def save():
        (OUT / f"{a.label}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    def run(tag, fill, messages, max_tokens):
        mon.mark = f"{fill}:{tag}"
        rec = {"tag": tag, "fill_target": fill}
        reasoning = content = ""
        try:
            r, reasoning, content = chat(a.base, a.model, messages, max_tokens, a.temp, extra)
            rec.update(r)
            rec["reasoning_chars"], rec["content_chars"] = len(reasoning), len(content)
            rec["head"] = (content or reasoning)[:160]
            rec["sys"] = mon.window(rec.pop("t0"), rec.pop("t1"))
        except urllib.error.HTTPError as e:
            rec["error"] = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
        except Exception as e:
            rec["error"] = repr(e)[:400]
        mon.mark = ""
        result["requests"].append(rec)
        save()
        e = rec.get("engine") or {}
        s = rec.get("sys") or {}
        print(f"[{a.label}] {fill:>7} {tag:6} prompt={rec.get('prompt_tokens')} cached={rec.get('cached_tokens')} ttft={rec.get('ttft_s')}s "
              f"prefill={e.get('prompt_per_second') or rec.get('prefill_tps_client')} gen={rec.get('completion_tokens')} "
              f"decode={rec.get('decode_tps_client')}/{e.get('predicted_per_second')} vram={s.get('vram_used_mib_max')} "
              f"rss={s.get('engine_rss_gib_max')} avail={s.get('ram_avail_gib_min')} disk={s.get('disk_read_MBps_avg')} {rec.get('error', '')}",
              flush=True)
        return rec, reasoning, content

    run("warmup", 0, [{"role": "user", "content": "Скажи «привет»."}], 64)
    cpt = a.cpt
    for fill in fills:
        doc, n, cpt = fit_doc(a.base, text, fill, cpt)
        if a.lean:
            lean_fill(fill, doc, run, save, a)
            continue
        word, port = f"орион-{fill % 977}", 20000 + fill % 9973
        cut = int(len(doc) * 0.37)
        cut = doc.find("\n\n===== FILE:", cut)
        doc = doc[:cut] + NEEDLE.format(word=word, port=port) + doc[cut:]
        rec, _, _ = run("cold", fill, [{"role": "user", "content": doc + Q_COLD}], a.gen)
        if rec.get("error"):
            continue
        if n is None and rec.get("prompt_tokens"):
            cpt = len(doc) / rec["prompt_tokens"]
        run("warm", fill, [{"role": "user", "content": doc + Q_WARM}], a.gen)
        q = "Какое кодовое слово развёртывания и какой порт стенда указаны в служебной заметке среди файлов выше? Ответь одной строкой."
        messages = [{"role": "user", "content": doc + q}]
        for t in range(a.turns):
            rec, reasoning, content = run("turn%d" % (t + 1), fill, messages, a.turn_tokens)
            if rec.get("error"):
                break
            if t == 0:
                rec["needle_ok"] = (word in content) and (str(port) in content)
                save()
            messages = messages + [{"role": "assistant", "content": content}, {"role": "user", "content": FOLLOW[t % len(FOLLOW)]}]
    mon.stop_flag = True
    result["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()


if __name__ == "__main__":
    main()
