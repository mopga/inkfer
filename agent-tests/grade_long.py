# Оценка длинной задачи run_long.sh (чтение ~14 тыс. строк -> ENV.md / ROUTES.md / правка / субагент -> вопросы по памяти).
#   python grade_long.py RUN [RUN ...]      -> runs\RUN\grade-long.json и сводная таблица в Markdown
# Эталон берётся из исходников на коммите 1974882 (repos\_seed); тесты запускаются в рабочей копии done\RUN.
import json, os, re, subprocess, sys
from grade import step_metrics, read, run, unittest_summary, TASK_PY, near

sys.stdout.reconfigure(encoding="utf-8")
A = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.join(A, "repos", "_seed")
FILES = ["serve/server.py", "serve/frontend.py", "serve/responses.py", "serve/mcp.py", "serve/telemetry.py"]


def src(path):
    return subprocess.run(["git", "show", "1974882:" + path], cwd=SEED, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


SRC = {f: src(f).splitlines() for f in FILES}


def truth_env():
    names = set()
    for f, lines in SRC.items():
        for l in lines:
            names.update(n for n in re.findall(r"STRATA_[A-Z0-9_]*[A-Z0-9]", l))
    return sorted(n for n in names if not any(o.startswith(n + "_") for o in names))   # STRATA_COALESCE_* in comments


def truth_routes():
    """Пути, которые сравнивает обработчик запросов: path == "...", path in (...), path.startswith(...)."""
    paths = set()
    for l in SRC["serve/server.py"]:
        for m in re.finditer(r"path(?:\.startswith\(|\s*==\s*|\s+in\s+)\(?([^:)]*)", l):
            paths.update(p for p in re.findall(r"\"(/[^\"]*)\"", m.group(1)))
    return sorted(paths)


def coverage(step_dir):
    """Какие строки каждого файла инструмент read действительно вернул модели (номера строк в выводе)."""
    seen = {f: set() for f in FILES}
    p = os.path.join(step_dir, "events.jsonl")
    if not os.path.exists(p):
        return {}
    for line in open(p, encoding="utf-8", errors="replace"):
        if '"tool_execution_end"' not in line[:80] or '"read"' not in line[:200]:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        r = e.get("result") or {}
        d = r.get("details") or {}
        src = ((d.get("meta") or {}).get("source") or {}).get("value") or ""
        nums = (d.get("displayContent") or {}).get("lineNumbers")
        f = next((k for k in FILES if src.replace("\\", "/").endswith(k)), None) if src else None
        if f and nums:                    # both output forms (numbered and raw) list the lines they returned here
            seen[f].update(int(x) for x in nums if isinstance(x, int))
            continue
        for c in r.get("content") or []:
            t = c.get("text") or ""
            m = re.match(r"\[([^\]#]+)", t)
            if not m:
                continue
            f = m.group(1).replace("\\", "/")
            f = next((k for k in FILES if f.endswith(k)), None)
            if f:
                seen[f].update(int(x) for x in re.findall(r"^(\d+):", t, re.M))
    return {f: dict(seen=len(s), total=len(SRC[f]), pct=round(100.0 * len(s) / max(1, len(SRC[f])), 1)) for f, s in seen.items()}


def ref_ok(name, file, line):
    lines = SRC.get(file) or []
    return any(name in lines[i] for i in range(max(0, line - 4), min(len(lines), line + 3)))


def grade_env(text):
    truth = truth_env()
    found = [n for n in truth if re.search(r"\b%s\b" % n, text)]
    extra = sorted(set(re.findall(r"STRATA_[A-Z0-9_]*[A-Z0-9]", text)) - set(truth))
    refs = ok = 0
    for row in text.splitlines():
        names = re.findall(r"STRATA_[A-Z0-9_]*[A-Z0-9]", row)
        if not names:
            continue
        for fm in re.finditer(r"(server|frontend|responses|mcp|telemetry)\.py[:\s#Lстрока]*?(\d{2,5})", row):
            refs += 1
            ok += any(ref_ok(n, "serve/%s.py" % fm.group(1), int(fm.group(2))) for n in names)
    return dict(found=len(found), of=len(truth), missing=[n for n in truth if n not in found], invented=extra,
                line_refs=refs, line_refs_ok=ok)


def grade_routes(text):
    truth = truth_routes()
    found = [p for p in truth if p in text]
    return dict(found=len(found), of=len(truth), missing=[p for p in truth if p not in found])


def grade_code(work):
    res = {}
    rc, diff = run(["git", "diff", "HEAD", "--", "serve"], work)
    rc, st = run(["git", "status", "--porcelain"], work)
    res["changed"] = [l[3:].strip() for l in st.splitlines() if l.strip()]
    plus = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    minus = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    res["uptime_added"] = any("uptime_s" in l for l in plus)
    res["uses_started_at"] = any("started_at" in l for l in plus)
    res["int_like_v1"] = any(re.search(r"int\(\s*(time\.time\(\)|now)\s*-\s*\w*\.?started_at", l) for l in plus)
    res["v1_status_untouched"] = not any("uptime_s" in l for l in minus)
    res["new_tests"] = re.findall(r"^\+\s+def (test_\w+)", diff, re.M)
    test_files = sorted({f for f in res["changed"] if re.search(r"(^|/)test_\w+\.py$", f)})
    mods = sorted(set(["serve.test_parallel", "serve.test_security"] + [f[:-3].replace("/", ".") for f in test_files]))
    rc, out = run([TASK_PY, "-m", "unittest"] + mods, work)
    res["suite"] = unittest_summary(out); res["suite_mods"] = mods
    # Поведение: /status на мок-сервисе отдаёт uptime_s (число), /v1/status — как было (целое).
    return res


def grade_tests4xx(text):
    names = set(re.findall(r"\b(test_\w+)", text))
    real = set(re.findall(r"^\s+def (test_\w+)", "\n".join(src("serve/test_server.py").splitlines()), re.M))
    return dict(exists=bool(text.strip()), named=len(names), real=len(names & real), invented=sorted(names - real)[:10])


def grade_recall(t):
    checks = {
        "1 COALESCE_TOKENS 16384 / 256": near(t, r"COALESCE_TOKENS|(?m:^)\s*1\.", r"16\s?384", 250) and near(t, r"COALESCE_TOKENS|(?m:^)\s*1\.", r"\b256\b", 250),
        "2 HTTP_BACKLOG 256 / 5 / request_queue_size": near(t, r"BACKLOG|(?m:^)\s*2\.", r"\b256\b", 250) and near(t, r"BACKLOG|(?m:^)\s*2\.", r"\b5\b", 250)
                                                       and bool(re.search(r"request_queue_size", t)),
        "3 KEEP_EMPTY_TURNS; последняя реплика": bool(re.search(r"STRATA_KEEP_EMPTY_TURNS", t)) and near(t, r"KEEP_EMPTY_TURNS", r"послед|final|last", 400),
        "4 unloaded, llama-server": near(t, r"\bunloaded\b", r"llama", 300),
        "5 issue #55": bool(re.search(r"#\s?55\b|issue\s+55\b|\b55\b", t)),
        "6 text; конфиг важнее": near(t, r"UNDECLARED_TOOL_CALLS|(?m:^)\s*6\.", r"\btext\b", 250) and near(t, r"UNDECLARED_TOOL_CALLS|(?m:^)\s*6\.", r"конфиг|config", 300),
        "7 BadBody, 400/413, #893": bool(re.search(r"BadBody", t)) and bool(re.search(r"#?\s?893\b", t)) and bool(re.search(r"\b(400|413)\b", t)),
        "8 60 и 3": near(t, r"STREAM_WRITE_TIMEOUT", r"\b60\b", 200) and near(t, r"START_RETRIES", r"\b3\b", 200),
        "9 ProgramFiles, C:\\Program Files": bool(re.search(r"ProgramFiles", t)) and bool(re.search(r"Program Files", t)),
        "10 uptime_s int от started_at": near(t, r"uptime_s", r"started_at", 300) and near(t, r"uptime_s", r"\bint\b|цел", 300),
    }
    return dict(score=sum(checks.values()), of=len(checks), checks=checks)


def grade(run_name):
    rd = os.path.join(A, "runs", run_name); work = os.path.join(A, "done", run_name)
    g = dict(run=run_name, L1=step_metrics(os.path.join(rd, "L1")), L2=step_metrics(os.path.join(rd, "L2")))
    for k in ("L1", "L2"):
        if g[k]:
            g[k].pop("final_text_full", None)
    g["coverage"] = coverage(os.path.join(rd, "L1"))
    g["env"] = grade_env(read(os.path.join(work, "docs", "ENV.md")))
    g["routes"] = grade_routes(read(os.path.join(work, "docs", "ROUTES.md")))
    g["code"] = grade_code(work) if os.path.isdir(work) else dict(error="нет рабочей копии")
    g["tests4xx"] = grade_tests4xx(read(os.path.join(work, "docs", "TESTS-4XX.md")))
    g["result_md"] = bool(read(os.path.join(work, "RESULT.md")).strip())
    g["recall"] = grade_recall((g["L2"] or {}).get("final_text", ""))
    g["recall"]["tool_calls"] = (g["L2"] or {}).get("tool_calls")
    with open(os.path.join(rd, "grade-long.json"), "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False, indent=1)
    return g


def table(gs):
    rows = [("Показатель", [g["run"] for g in gs])]
    def cov(g):
        c = g["coverage"]; s = sum(v["seen"] for v in c.values()); t = sum(v["total"] for v in c.values())
        return "%d/%d строк (%.0f %%); server.py %.0f %%" % (s, t, 100.0 * s / max(1, t), c.get("serve/server.py", {}).get("pct", 0)) if c else "—"
    def m(g, k, f):
        return f(g[k]) if g.get(k) else "—"
    rows += [
        ("Прочитано строк из 5 файлов", [cov(g) for g in gs]),
        ("ENV.md: переменных найдено", ["%d/%d (лишних %d)" % (g["env"]["found"], g["env"]["of"], len(g["env"]["invented"])) for g in gs]),
        ("ENV.md: ссылок на строки верно", ["%d/%d" % (g["env"]["line_refs_ok"], g["env"]["line_refs"]) for g in gs]),
        ("ROUTES.md: путей найдено", ["%d/%d" % (g["routes"]["found"], g["routes"]["of"]) for g in gs]),
        ("uptime_s: добавлено / от started_at / как в /v1/status (int)", ["%s / %s / %s" % (g["code"].get("uptime_added"), g["code"].get("uses_started_at"), g["code"].get("int_like_v1")) for g in gs]),
        ("Тесты (новые / прогон)", ["%d / %s %s" % (len(g["code"].get("new_tests", [])), g["code"].get("suite", {}).get("ran"), "OK" if g["code"].get("suite", {}).get("ok") else "FAIL " + g["code"].get("suite", {}).get("fail", "")) for g in gs]),
        ("Субагент: TESTS-4XX.md (тестов названо / существуют)", ["%s %d/%d" % ("есть" if g["tests4xx"]["exists"] else "нет", g["tests4xx"]["named"], g["tests4xx"]["real"]) for g in gs]),
        ("RESULT.md", ["да" if g["result_md"] else "нет" for g in gs]),
        ("Вопросы по памяти (без инструментов)", ["%d/%d (инструментов %s)" % (g["recall"]["score"], g["recall"]["of"], g["recall"]["tool_calls"]) for g in gs]),
        ("Время L1 + L2, мин", [m(g, "L1", lambda x: "%.1f" % ((x["wall_s"] or 0) / 60)) + " + " + m(g, "L2", lambda x: "%.1f" % ((x["wall_s"] or 0) / 60)) for g in gs]),
        ("Запросов к модели", [m(g, "L1", lambda x: str(x["requests"])) for g in gs]),
        ("Наибольший промпт", [m(g, "L1", lambda x: "{:,}".format(x["max_prompt"]).replace(",", " ")) + " / " + m(g, "L2", lambda x: "{:,}".format(x["max_prompt"]).replace(",", " ")) for g in gs]),
        ("Доля кэша", [m(g, "L1", lambda x: "%.1f %%" % x["cache_pct"]) for g in gs]),
        ("Токены: прочитано / из кэша / сгенерировано", [m(g, "L1", lambda x: "%dk / %.1fM / %dk" % (x["input"] // 1000, x["cache_read"] / 1e6, x["output"] // 1000)) for g in gs]),
        ("Decode в сессии, ток/с", [m(g, "L1", lambda x: str(x["decode_tps"])) for g in gs]),
        ("Вызовов инструментов / ошибок / отказов", [m(g, "L1", lambda x: "%d / %d / %d" % (x["tool_calls"], x["tool_errors"], x["denied"])) for g in gs]),
        ("Сжатий истории", [m(g, "L1", lambda x: str(x["compactions"])) for g in gs]),
    ]
    out = "| " + " | ".join([rows[0][0]] + rows[0][1]) + " |\n|" + "---|" * (len(gs) + 1) + "\n"
    for name, vals in rows[1:]:
        out += "| " + " | ".join([name] + vals) + " |\n"
    return out


if __name__ == "__main__":
    gs = [grade(r) for r in sys.argv[1:]]
    print(table(gs))
    for g in gs:
        print(g["run"], "recall:", {k: v for k, v in g["recall"]["checks"].items() if not v}, "env missing:", g["env"]["missing"][:20],
              "routes missing:", g["routes"]["missing"][:20])
