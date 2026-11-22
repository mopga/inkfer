"""Greedy (temperature 0) answers to a fixed set of prompts, saved to a file, to compare engine settings:
the same engine with and without speculative decoding must give the same text.

  python detcheck.py --base URL --label LABEL [--max-tokens 300]      -> bench/runs/det-LABEL.json
  python detcheck.py --compare LABEL_A LABEL_B                        -> per prompt: identical / first difference
"""
import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = [
    "Write a Python function that merges overlapping intervals. Include a docstring and three asserts.",
    "Объясни в пяти предложениях, почему быстрая сортировка в среднем работает за O(n log n).",
    "Given the JSON {\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}, list every path to a leaf value, one per line.",
    "A train leaves at 14:35 and arrives at 17:10 after two stops of 7 and 12 minutes. How long was it moving? Show the steps.",
    "Напиши SQL-запрос: для таблицы orders(id, customer_id, total, created_at) выведи трёх клиентов с наибольшей суммой заказов за 2025 год.",
    "Write a Rust function that returns the longest common prefix of a slice of strings, with two unit tests.",
]
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a text file and return its content.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}},
                   "required": ["path"]}}}]


def ask(base, model, prompt, max_tokens, tools=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "seed": 1, "stream": False, "reasoning_effort": "low"}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.loads(r.read())
    m = j["choices"][0]["message"]
    return {"prompt": prompt, "reasoning": m.get("reasoning_content") or "", "content": m.get("content") or "",
            "tool_calls": [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])],
            "finish": j["choices"][0].get("finish_reason"), "usage": j.get("usage"), "timings": j.get("timings"),
            "seconds": round(time.time() - t0, 2)}


def compare(a, b):
    ra = json.load(open(os.path.join(HERE, "runs", f"det-{a}.json"), encoding="utf-8"))
    rb = json.load(open(os.path.join(HERE, "runs", f"det-{b}.json"), encoding="utf-8"))
    same = 0
    for x, y in zip(ra, rb):
        ta = x["reasoning"] + "\x00" + x["content"] + "\x00" + json.dumps(x["tool_calls"])
        tb = y["reasoning"] + "\x00" + y["content"] + "\x00" + json.dumps(y["tool_calls"])
        if ta == tb:
            same += 1
            print("identical |", x["prompt"][:60])
        else:
            n = next((i for i, (p, q) in enumerate(zip(ta, tb)) if p != q), min(len(ta), len(tb)))
            print(f"DIFFERENT at char {n} of {len(ta)}/{len(tb)} |", x["prompt"][:60])
    print(f"{same} of {len(ra)} identical")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="inkling")
    ap.add_argument("--label")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    out = []
    for p in PROMPTS:
        out.append(ask(a.base, a.model, p, a.max_tokens))
    out.append(ask(a.base, a.model, "Read the file config/app.yaml and tell me which port the service listens on.", a.max_tokens, TOOLS))
    for r in out:
        print(r["seconds"], "s |", r["finish"], "|", (r["content"] or str(r["tool_calls"]))[:70].replace("\n", " "))
    json.dump(out, open(os.path.join(HERE, "runs", f"det-{a.label}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
