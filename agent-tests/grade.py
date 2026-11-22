# Оценка агентного прогона run_chain.sh (анализ -> исправление трёх ошибок -> проверка памяти сессии).
#   python grade.py RUN [RUN ...]      -> runs\RUN\grade.json и сводная таблица в Markdown
# Читает runs\RUN\step*\events.jsonl и рабочую копию done\RUN; тесты запускает в отдельных временных копиях.
import json, os, re, shutil, subprocess, sys, tempfile

sys.stdout.reconfigure(encoding="utf-8")
A = os.path.dirname(os.path.abspath(__file__))
TASK_PY = r"C:\WORKDIR\harness\task-python\python.exe"
MODS = ["serve.test_admit_order", "serve.test_parallel", "serve.test_yield_accounting"]
EVENTS = ["queued", "admitted", "reading", "first_token", "finished", "cancelled", "disconnected", "timeout", "error"]
OUTCOMES = ["stop", "length", "cancel", "disconnect", "timeout", "error"]


def has(text, *pats):
    return all(re.search(p, text, re.I | re.S) for p in pats)


def near(text, a, b, span=400):
    """a и b встречаются не дальше span символов друг от друга."""
    for m in re.finditer(a, text, re.I):
        if re.search(b, text[max(0, m.start() - span): m.end() + span], re.I | re.S):
            return True
    return False


def step_metrics(d):
    out = dict(requests=0, input=0, output=0, cache_read=0, max_prompt=0, gen_s=0.0, ttft_s=0.0, tools={}, tool_errors=0,
               denied=0, bad_stops=0, compactions=0, final_text="")
    p = os.path.join(d, "events.jsonl")
    if not os.path.exists(p):
        return None
    for line in open(p, encoding="utf-8", errors="replace"):
        head = line[:60]
        if '"message_update"' in head or '"tool_execution_update"' in head or '"tool_stream_update"' in head:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        t = e.get("type")
        if t == "message_end" and (e.get("message") or {}).get("role") == "assistant":
            m = e["message"]; u = m.get("usage") or {}
            out["requests"] += 1
            out["input"] += u.get("input", 0); out["output"] += u.get("output", 0); out["cache_read"] += u.get("cacheRead", 0)
            out["max_prompt"] = max(out["max_prompt"], u.get("input", 0) + u.get("cacheRead", 0))
            dur, ttft = (m.get("duration") or 0) / 1000.0, (m.get("ttft") or 0) / 1000.0
            out["gen_s"] += max(0.0, dur - ttft); out["ttft_s"] += ttft
            if m.get("stopReason") in ("error", "aborted"):
                out["bad_stops"] += 1
            text = "".join(c.get("text", "") for c in m.get("content") or [] if c.get("type") == "text")
            if text.strip():
                out["final_text"] = text
        elif t == "tool_execution_start":
            n = e.get("toolName", "?"); out["tools"][n] = out["tools"].get(n, 0) + 1
        elif t == "tool_execution_end":
            r = e.get("result") or {}
            if e.get("isError") or r.get("isError"):
                out["tool_errors"] += 1
                txt = json.dumps(r, ensure_ascii=False)[:600]
                if "requires approval" in txt or "denied" in txt.lower():
                    out["denied"] += 1
        elif t == "auto_compaction_start":
            out["compactions"] += 1
    try:
        out["wall_s"] = int(open(os.path.join(d, "finished.epoch")).read()) - int(open(os.path.join(d, "started.epoch")).read())
    except Exception:
        out["wall_s"] = None
    out["tool_calls"] = sum(out["tools"].values())
    tot = out["input"] + out["cache_read"]
    out["cache_pct"] = round(100.0 * out["cache_read"] / tot, 1) if tot else 0.0
    out["decode_tps"] = round(out["output"] / out["gen_s"], 1) if out["gen_s"] > 0 else 0.0
    return out


def read(path):
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""


def grade_analysis(text):
    checks = {
        "1a SHORT_FIRST, по умолчанию 0 = выключено": has(text, r"STRATA_ADMIT_SHORT_FIRST") and near(text, r"STRATA_ADMIT_SHORT_FIRST", r"\b0\b"),
        "1b MAX_WAIT_S = 30": near(text, r"STRATA_ADMIT_MAX_WAIT_S", r"\b30\b"),
        "2a limit = max(1, batch) + max_queue": has(text, r"max_queue") and has(text, r"max\(1|batch"),
        "2b нет предела при max_queue None": near(text, r"max_queue", r"None|не задан|не указан|отсутств"),
        "3 running / queued / limit": has(text, r"running", r"queued", r"limit") and has(text, r"live_reqs|busy"),
        "4a формула 1 + min(queued, …)": has(text, r"1\s*\+\s*min|min\(") and has(text, r"retry_after_for"),
        "4b RETRY_AFTER_MAX = 10": near(text, r"RETRY_AFTER_MAX", r"\b10\b"),
        "5 QUEUE_BEAT_S = 10 с": near(text, r"QUEUE_BEAT_S", r"\b10(\.0)?\b"),
        "6a SOLO_AGAIN_MAX = 2": near(text, r"SOLO_AGAIN_MAX", r"\b2\b", 200),
        "6b SOLO_AGAIN_MIN_LEFT = 32": near(text, r"SOLO_AGAIN_MIN_LEFT", r"\b32\b", 200),
        "6c прочие условия (slot_cache, PARALLEL_SOLO, embeddings)": sum(bool(re.search(p, text)) for p in (r"slot_cache", r"STRATA_PARALLEL_SOLO", r"embeddings")) >= 2,
        "7a _shorter_waiting": has(text, r"_shorter_waiting"),
        "7b YIELDS_MAX / два места вызова": has(text, r"YIELDS_MAX") or len(re.findall(r"server\.py:(230\d|244\d)", text)) >= 2,
        "8a все 9 событий": all(re.search(r"\b%s\b" % w, text) for w in EVENTS),
        "8b все 6 outcome": all(re.search(r"\b%s\b" % w, text) for w in OUTCOMES),
    }
    refs = len(re.findall(r"[\w/\\.]+\.py:\d+", text))
    tail = text[text.lower().rfind("замечани"):] if "замечани" in text.lower() else ""
    found = {
        "A (_admit_next: and вместо or)": bool(re.search(r"_admit_next|ADMIT_MAX_WAIT", tail)) and bool(re.search(r"\band\b|\bor\b|никогда|оба услови|одновременно", tail)),
        "B (Retry-After до 11)": bool(re.search(r"\b11\b", tail)),
        "C (_shorter_waiting: вдвое длиннее)": bool(re.search(r"_shorter_waiting|#656", tail)) and bool(re.search(r"\* ?2|вдвое|половин|half", tail)),
    }
    return dict(score=sum(checks.values()), of=len(checks), checks=checks, refs=refs, noticed=found, noticed_n=sum(found.values()))


PROBE = r'''
import json, sys, threading, time, types
sys.path.insert(0, ".")
import serve.server as s
r = {}
try:
    r["B"] = [s.retry_after_for({"queued": q}) for q in (0, 1, 3, 9, 10, 12, 50)] == [1, 2, 4, 10, 10, 10, 10]
except Exception as e:
    r["B"] = "error: %r" % e
cls = next(c for c in vars(s).values() if isinstance(c, type) and "_shorter_waiting" in vars(c))
def fake(wait):
    o = types.SimpleNamespace(slot_cv=threading.Condition(), wait_lens=wait); return o
try:
    f = cls._shorter_waiting
    r["C"] = (f(fake([[90000, 0, True]]), 100000) is False and f(fake([[150000, 0, True]]), 100000) is False
              and f(fake([[50000, 0, True]]), 100000) is True and f(fake([[1000, 0, True]]), 100000) is True
              and f(fake([[50001, 0, True]]), 100000) is False)
except Exception as e:
    r["C"] = "error: %r" % e
try:
    s.ADMIT_SHORT_FIRST, s.ADMIT_MAX_WAIT_S = 8192, 30.0
    now = time.monotonic()
    long_old, long_new, short_new = [100000, now - 5, True], [120000, now - 3, True], [500, now - 1, True]
    g = cls._admit_next
    a1 = g(fake([long_old, long_new, short_new])) is short_new          # короткий раньше длинных
    stale = [100000, now - 40, True]
    a2 = g(fake([stale, short_new])) is stale                            # длинный, ждавший > 30 с, считается коротким
    a3 = g(fake([long_old, long_new])) is long_old                       # порядок прихода среди длинных
    r["A"] = bool(a1 and a2 and a3); r["A_detail"] = [a1, a2, a3]
except Exception as e:
    r["A"] = "error: %r" % e
print("PROBE " + json.dumps(r))
'''


def run(cmd, cwd, timeout=900):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "timeout"


def unittest_summary(out):
    m = re.search(r"Ran (\d+) tests?", out)
    return dict(ran=int(m.group(1)) if m else 0, ok=bool(re.search(r"^OK\b", out, re.M)),
                fail=re.search(r"FAILED \(([^)]*)\)", out).group(1) if re.search(r"FAILED \(([^)]*)\)", out) else "")


def grade_bugfix(work):
    res = {}
    if not os.path.isdir(work):
        return dict(error="нет рабочей копии")
    rc, out = run([TASK_PY, "-c", PROBE], work, 120)
    m = re.search(r"PROBE (\{.*\})", out)
    res["fixed"] = json.loads(m.group(1)) if m else {"error": out[-400:]}
    res["fixed_n"] = sum(1 for k in "ABC" if res["fixed"].get(k) is True)
    rc, d = run(["git", "diff", "--numstat", "HEAD", "--", "serve/server.py"], work)
    res["server_diff"] = d.strip().split("\t")[:2] if d.strip() else []
    rc, st = run(["git", "status", "--porcelain"], work)
    res["changed"] = [l[3:].strip() for l in st.splitlines() if l.strip()]
    rc, diff = run(["git", "diff", "HEAD", "--", "serve"], work)
    added = re.findall(r"^\+\s+def (test_\w+)", diff, re.M)
    new_files = [f for f in res["changed"] if re.search(r"test_\w+\.py$", f) and f not in diff]
    for f in new_files:
        added += re.findall(r"^\s+def (test_\w+)", read(os.path.join(work, f)), re.M)
    res["new_tests"] = added
    test_files = sorted({f for f in res["changed"] if re.search(r"(^|/)test_\w+\.py$", f)})
    mods = sorted(set(MODS + [f[:-3].replace("/", ".") for f in test_files]))
    rc, out = run([TASK_PY, "-m", "unittest"] + mods, work)
    res["suite"] = unittest_summary(out); res["suite_mods"] = mods
    # Ловят ли новые тесты ошибки: те же тестовые файлы на исходном (ошибочном) коде.
    tmp = tempfile.mkdtemp(prefix="grade-")
    try:
        old = os.path.join(tmp, "old")
        run(["git", "clone", "-q", os.path.join(A, "repos", "_bugseed"), old], tmp)
        for f in test_files:
            os.makedirs(os.path.dirname(os.path.join(old, f)), exist_ok=True)
            shutil.copy2(os.path.join(work, f), os.path.join(old, f))
        rc, out = run([TASK_PY, "-m", "unittest", "-v"] + mods, old)
        bad = re.findall(r"^(?:FAIL|ERROR): (test_\w+)", out, re.M)
        res["old_code"] = dict(unittest_summary(out), failing=sorted(set(bad)),
                               new_failing=sorted(set(bad) & set(added)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bf = read(os.path.join(work, "BUGFIX.md"))
    res["bugfix_md"] = dict(exists=bool(bf), funcs=sum(bool(re.search(p, bf)) for p in (r"_admit_next", r"retry_after_for", r"_shorter_waiting")),
                            tests_named=sum(1 for t in added if t in bf))
    return res


def grade_retention(text, tools, bug, last_ran):
    t = text
    checks = {
        "1 MAX_WAIT_S = 30": near(t, r"STRATA_ADMIT_MAX_WAIT_S", r"\b30\b", 200),
        "2 SOLO_AGAIN_MAX = 2 и MIN_LEFT = 32": near(t, r"SOLO_AGAIN_MAX", r"\b2\b", 120) and near(t, r"SOLO_AGAIN_MIN_LEFT", r"\b32\b", 120),
        "3 все 9 событий": all(re.search(r"\b%s\b" % w, t) for w in EVENTS),
        "4 три функции": has(t, r"_admit_next", r"retry_after_for", r"_shorter_waiting"),
        "4 выражения (or / min(...MAX) / *2 <= plen)": sum([bool(re.search(r"\bor\b", t)), bool(re.search(r"min\([^)]*RETRY_AFTER_MAX", t)),
                                                        bool(re.search(r"\*\s*2\s*<=?\s*plen|plen\s*(//|/)\s*2|<=?\s*plen\s*/", t))]) == 3,
        "5 имена своих тестов (не меньше трёх)": sum(1 for n in bug.get("new_tests", []) if n in t) >= min(3, max(1, len(bug.get("new_tests", [])))),
        "6 число тестов последнего прогона": bool(last_ran) and bool(re.search(r"\b%d\b" % last_ran, t)),
        "7 изменённые файлы": has(t, r"server\.py", r"ANALYSIS\.md", r"BUGFIX\.md"),
    }
    return dict(score=sum(checks.values()), of=len(checks), checks=checks, tool_calls=tools)


def last_test_run(step_dir):
    p = os.path.join(step_dir, "events.jsonl"); last = 0
    if not os.path.exists(p):
        return 0
    for line in open(p, encoding="utf-8", errors="replace"):
        if '"tool_execution_end"' in line[:60] and "Ran " in line:
            m = re.findall(r"Ran (\d+) tests?", line)
            if m:
                last = int(m[-1])
    return last


def grade(run_name):
    rd = os.path.join(A, "runs", run_name); work = os.path.join(A, "done", run_name)
    g = dict(run=run_name, params=read(os.path.join(rd, "step1", "params.txt")).strip())
    g["steps"] = {s: step_metrics(os.path.join(rd, s)) for s in ("step1", "step2", "step3")}
    g["analysis"] = grade_analysis(read(os.path.join(work, "ANALYSIS.md")))
    g["bugfix"] = grade_bugfix(work)
    s3 = g["steps"]["step3"] or {}
    g["retention"] = grade_retention(s3.get("final_text", ""), s3.get("tool_calls", 0), g["bugfix"], last_test_run(os.path.join(rd, "step2")))
    for s in g["steps"].values():
        if s:
            s["final_text"] = s["final_text"][:6000]
    json.dump(g, open(os.path.join(rd, "grade.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return g


def table(gs):
    def tot(g, k):
        return sum((s or {}).get(k) or 0 for s in g["steps"].values())
    rows = [
        ("Анализ: фактов верно", lambda g: "%d/%d" % (g["analysis"]["score"], g["analysis"]["of"])),
        ("Анализ: ссылок файл:строка", lambda g: str(g["analysis"]["refs"])),
        ("Анализ: замечено заложенных ошибок", lambda g: "%d/3" % g["analysis"]["noticed_n"]),
        ("Ошибки исправлены (A/B/C)", lambda g: "%d/3 (%s)" % (g["bugfix"].get("fixed_n", 0), "".join(k if g["bugfix"].get("fixed", {}).get(k) is True else "-" for k in "ABC"))),
        ("Правка server.py (+/− строк)", lambda g: "/".join(g["bugfix"].get("server_diff", []) or ["-"])),
        ("Набор тестов после правки", lambda g: "%s, %d тестов" % ("OK" if g["bugfix"].get("suite", {}).get("ok") else "FAILED " + g["bugfix"].get("suite", {}).get("fail", ""), g["bugfix"].get("suite", {}).get("ran", 0))),
        ("Новых тестов / из них падают на старом коде", lambda g: "%d / %d" % (len(g["bugfix"].get("new_tests", [])), len(g["bugfix"].get("old_code", {}).get("new_failing", [])))),
        ("BUGFIX.md: функций названо", lambda g: "%d/3" % g["bugfix"].get("bugfix_md", {}).get("funcs", 0)),
        ("Память сессии: верно", lambda g: "%d/%d" % (g["retention"]["score"], g["retention"]["of"])),
        ("Память: вызовов инструментов (нужно 0)", lambda g: str(g["retention"]["tool_calls"])),
        ("Время шагов, мин", lambda g: " + ".join("%.1f" % ((s or {}).get("wall_s") or 0) / 60.0 if False else "%.1f" % (((s or {}).get("wall_s") or 0) / 60.0) for s in g["steps"].values())),
        ("Время всего, мин", lambda g: "%.1f" % (tot(g, "wall_s") / 60.0)),
        ("Запросов к модели", lambda g: str(tot(g, "requests"))),
        ("Токены: прочитано / из кэша / сгенерировано", lambda g: "%d / %d / %d" % (tot(g, "input"), tot(g, "cache_read"), tot(g, "output"))),
        ("Доля кэша, %", lambda g: "%.1f" % (100.0 * tot(g, "cache_read") / max(1, tot(g, "cache_read") + tot(g, "input")))),
        ("Наибольший промпт, ток.", lambda g: str(max((s or {}).get("max_prompt") or 0 for s in g["steps"].values()))),
        ("Decode в сессии, ток/с", lambda g: "%.0f" % (tot(g, "output") / max(1e-9, tot(g, "gen_s")))),
        ("Вызовов инструментов / ошибок / отказов", lambda g: "%d / %d / %d" % (tot(g, "tool_calls"), tot(g, "tool_errors"), tot(g, "denied"))),
        ("Сбоев ответа (error/aborted)", lambda g: str(tot(g, "bad_stops"))),
        ("Сжатий истории", lambda g: str(tot(g, "compactions"))),
    ]
    print("| Показатель | " + " | ".join(g["run"] for g in gs) + " |")
    print("|---|" + "---|" * len(gs))
    for name, f in rows:
        cells = []
        for g in gs:
            try:
                cells.append(f(g))
            except Exception as e:
                cells.append("?")
        print("| %s | %s |" % (name, " | ".join(cells)))


if __name__ == "__main__":
    gs = [grade(r) for r in sys.argv[1:]]
    table(gs)
    for g in gs:
        miss = [k for k, v in g["analysis"]["checks"].items() if not v]
        rmiss = [k for k, v in g["retention"]["checks"].items() if not v]
        print("\n%s: анализ — не найдено: %s; память — не найдено: %s" % (g["run"], "; ".join(miss) or "нет", "; ".join(rmiss) or "нет"))
