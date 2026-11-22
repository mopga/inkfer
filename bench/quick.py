"""One or more quick requests to an OpenAI-compatible server; prints the engine's and the client's speed.

  python quick.py [--base http://127.0.0.1:8090] [--n 2] [--max-tokens 400] [--effort low] [--prompt TEXT] [--show]
"""
import argparse, json, sys, time, urllib.request

PROMPTS = [
    "Write a Python class implementing an LRU cache with get/put in O(1), with type hints and a short usage example.",
    "Объясни подробно, чем B-дерево отличается от LSM-дерева и когда что выбирать. Пиши связным текстом, около 300 слов.",
    "Write a C++17 function that parses a CSV line with quoted fields and escaped quotes into a vector of strings.",
]


def one(base, model, prompt, max_tokens, effort, temp):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if effort:
        body["reasoning_effort"] = effort
    if temp is not None:
        body["temperature"] = temp
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t1 = None
    usage = timings = None
    rc, c = [], []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("reasoning_content") or d.get("content") or d.get("tool_calls"):
                    t1 = t1 or time.time()
                rc.append(d.get("reasoning_content") or "")
                c.append(d.get("content") or "")
    t2 = time.time()
    gen = (usage or {}).get("completion_tokens") or 0
    out = {"ttft": round((t1 or t2) - t0, 2), "gen": gen, "client_tps": round((gen - 1) / (t2 - t1), 1) if t1 and gen > 1 and t2 > t1 else None}
    if timings:
        out["engine_decode_tps"] = round(timings.get("predicted_per_second", 0), 1)
        out["engine_prefill_tps"] = round(timings.get("prompt_per_second", 0), 1)
        for k in ("draft_n", "draft_n_accepted"):
            if k in timings:
                out[k] = timings[k]
    return out, "".join(rc), "".join(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="inkling")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--effort", default="")
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    for i in range(a.n):
        p = a.prompt or PROMPTS[i % len(PROMPTS)]
        out, rc, c = one(a.base, a.model, p, a.max_tokens, a.effort, a.temp)
        print(json.dumps(out, ensure_ascii=False), flush=True)
        if a.show:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            print("--- reasoning:", rc[:500].replace("\n", " "))
            print("--- content:", c[:1200])


if __name__ == "__main__":
    main()
