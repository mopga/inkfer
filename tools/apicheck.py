"""Checks of what the agents need from the engine's OpenAI-compatible API (run against a loaded profile).

    python apicheck.py [PORT] [MODEL]          defaults: 8081, the first model of /v1/models

Checks: health / models / metrics, streaming with separate reasoning and answer, tool call -> tool result -> answer,
two tool calls in one reply, the output limit, stop in the middle of a stream (the engine must get free at once),
continuing the stopped conversation from the cache. Prints one line per check and writes logs\\apicheck.json.
"""
import http.client, json, os, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"http://127.0.0.1:{PORT}"
RESULTS = []


def get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def post(body, timeout=600):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def stream(body, stop_after_chars=None, timeout=600):
    """Returns reasoning, content, tool calls, finish reason, usage, time to first token, chunks, stopped flag."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    t0 = time.time()
    conn.request("POST", "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode("utf-8"),
                 {"Content-Type": "application/json", "Accept": "text/event-stream"})
    resp = conn.getresponse()
    out = {"status": resp.status, "reasoning": "", "content": "", "tools": {}, "finish": None, "usage": None,
           "ttft": None, "chunks": 0, "stopped": False, "done_marker": False}
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            out["done_marker"] = True
            break
        d = json.loads(data)
        out["chunks"] += 1
        if d.get("usage"):
            out["usage"] = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if piece or delta.get("content") or delta.get("tool_calls"):
                out["ttft"] = out["ttft"] or round(time.time() - t0, 2)
            out["reasoning"] += piece
            out["content"] += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                slot = out["tools"].setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
        if stop_after_chars and len(out["content"]) >= stop_after_chars:      # the answer itself, not the reasoning
            out["stopped"] = True
            break
    out["seconds"] = round(time.time() - t0, 2)
    conn.close()                                   # for a stopped stream this is what the Stop button does
    return out


def check(name, ok, detail):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("OK   " if ok else "FAIL ") + name + ": " + detail)


TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file of the project and return its contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path of the file"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List the files of a directory of the project.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]


def main():
    code, body = get("/health")
    check("health", code == 200 and "ok" in body.lower(), body.strip()[:120])
    code, body = get("/v1/models")
    models = [m["id"] for m in json.loads(body).get("data", [])]
    model = sys.argv[2] if len(sys.argv) > 2 else models[0]
    check("models", model in models, ", ".join(models))
    for path in ("/metrics", "/stats"):
        try:
            code, body = get(path)
            check("state " + path, code == 200 and len(body) > 50, f"{len(body)} bytes: " + " ".join(body.split())[:110])
        except Exception as e:
            check("state " + path, False, repr(e)[:150])

    base = {"model": model, "temperature": 0}

    # 1. streaming: reasoning and the answer arrive in separate fields, usage at the end
    s = stream(dict(base, max_tokens=1500, messages=[
        {"role": "user", "content": "Сколько будет 17 * 23? Сначала подумай, потом дай ответ одной строкой."}]))
    check("stream: reasoning apart from the answer",
          s["status"] == 200 and s["reasoning"].strip() and "391" in s["content"] and "<think" not in s["content"].lower()
          and s["finish"] == "stop" and s["done_marker"],
          f"reasoning {len(s['reasoning'])} chars, answer {s['content'].strip()[:60]!r}, finish {s['finish']}, "
          f"chunks {s['chunks']}, first token {s['ttft']} s")
    check("stream: usage in the last chunk", bool(s["usage"]) and s["usage"].get("completion_tokens", 0) > 0, json.dumps(s["usage"])[:160])

    # 2. reasoning turned down: Inkling opens its reasoning block by itself at any effort (the template only passes the
    #    level), so the check is that the request is accepted and the answer still arrives apart from the reasoning
    s = stream(dict(base, max_tokens=300, reasoning_effort="none", messages=[
        {"role": "user", "content": "Ответь одним словом: столица Франции?"}]))
    check("stream: reasoning_effort none", "Париж" in s["content"],
          f"reasoning {len(s['reasoning'])} chars, answer {s['content'].strip()[:40]!r}")

    # 3. tool call, then the tool's result, then the answer
    msgs = [{"role": "system", "content": "Ты агент-программист. Для чтения файлов используй инструменты."},
            {"role": "user", "content": "Прочитай файл config/app.toml и скажи, какой там порт."}]
    s = stream(dict(base, max_tokens=1500, tools=TOOLS, messages=msgs))
    tc = s["tools"].get(0) or {}
    try:
        args = json.loads(tc.get("arguments") or "")
    except ValueError:
        args = None
    check("tools: call in a stream", s["finish"] == "tool_calls" and tc.get("name") == "read_file" and isinstance(args, dict)
          and "app.toml" in str(args.get("path")) and not s["content"].strip().startswith("{"),
          f"finish {s['finish']}, {tc.get('name')}({tc.get('arguments')}), id {tc.get('id')}, reasoning {len(s['reasoning'])} chars")
    msgs += [{"role": "assistant", "content": s["content"] or None, "reasoning_content": s["reasoning"],
              "tool_calls": [{"id": tc.get("id") or "call_1", "type": "function",
                              "function": {"name": tc.get("name") or "read_file", "arguments": tc.get("arguments") or "{}"}}]},
             {"role": "tool", "tool_call_id": tc.get("id") or "call_1", "content": "[server]\nhost = \"0.0.0.0\"\nport = 8457\nworkers = 4\n"}]
    s2 = stream(dict(base, max_tokens=1500, tools=TOOLS, messages=msgs))
    cached = ((s2["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    check("tools: answer after the tool result", s2["finish"] == "stop" and "8457" in s2["content"],
          f"answer {s2['content'].strip()[:80]!r}, cached {cached} of {(s2['usage'] or {}).get('prompt_tokens')}")

    # 4. two calls in one reply
    s = stream(dict(base, max_tokens=1500, tools=TOOLS, messages=[
        {"role": "system", "content": "Ты агент-программист. Независимые вызовы инструментов делай одним ответом."},
        {"role": "user", "content": "Одновременно: прочитай файл README.md и выведи список каталога src."}]))
    names = sorted(t["name"] for t in s["tools"].values())
    ok_args = all(isinstance(json.loads(t["arguments"] or "null"), dict) for t in s["tools"].values()) if s["tools"] else False
    check("tools: two calls in one reply", names == ["list_dir", "read_file"] and ok_args and s["finish"] == "tool_calls",
          f"finish {s['finish']}, calls {[(t['name'], t['arguments']) for t in s['tools'].values()]}")

    # 5. the output limit
    d = post(dict(base, max_tokens=40, reasoning_effort="none", messages=[
        {"role": "user", "content": "Напиши длинный рассказ про маяк, не меньше тысячи слов."}]))
    ch = d["choices"][0]
    check("limit: finish_reason length", ch["finish_reason"] == "length" and d["usage"]["completion_tokens"] <= 40,
          f"finish {ch['finish_reason']}, completion {d['usage']['completion_tokens']}")

    # 6. stop in the middle of a stream; the next request must not wait for the stopped one
    notes = "\n".join(f"Заметка {i}: маяк {i % 7 + 1}, смена {i % 3 + 1}, ветер {5 + i % 20} м/с, видимость {2 + i % 9} миль."
                      for i in range(160))
    story = [{"role": "system", "content": "Ты писатель. Пиши подробно. Справочные заметки смотрителя:\n" + notes},
             {"role": "user", "content": "Напиши очень длинный рассказ про смотрителя маяка, не меньше трёх тысяч слов."}]
    s = stream(dict(base, max_tokens=6000, reasoning_effort="none", messages=story), stop_after_chars=600)
    t0 = time.time()
    d = post(dict(base, max_tokens=20, reasoning_effort="none", messages=[{"role": "user", "content": "Ответь одним словом: 2+2?"}]))
    waited = round(time.time() - t0, 2)
    check("stop: stream closed by the client", s["stopped"] and waited < 15,
          f"stopped after {len(s['content'])} chars of the answer in {s['seconds']} s; the next request answered in {waited} s "
          f"({d['choices'][0]['message'].get('content', '').strip()[:20]!r})")

    # 7. continue the stopped conversation: the prompt and the part already written come from the cache
    cont = story + [{"role": "assistant", "content": s["content"]}, {"role": "user", "content": "Продолжай с того же места."}]
    s3 = stream(dict(base, max_tokens=900, reasoning_effort="none", messages=cont))
    u = s3["usage"] or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    check("continue after stop", s3["finish"] in ("length", "stop") and len(s3["content"]) > 100
          and cached >= (u.get("prompt_tokens") or 0) // 2,
          f"stopped part {len(s['content'])} chars; cached {cached} of {u.get('prompt_tokens')}, first token {s3['ttft']} s, "
          f"text {s3['content'].strip()[:50]!r}")

    bad = [r for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    json.dump({"port": PORT, "model": model, "results": RESULTS}, open(os.path.join(HERE, "logs", f"apicheck-{model}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    sys.exit(1 if bad else 0)


main()
