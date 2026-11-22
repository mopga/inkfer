"""Two conversations taking turns on one engine (an agent and its subagent): does the first one keep its cache while
the second makes several requests in a row?  The engine's host store is filled on purpose (start the engine with a
small --host-cache-mib, e.g. 3072 for the default sizes) to reproduce what a 16 GiB store meets at 180k + 125k tokens.

    python alttest.py PORT MODEL [A_TOKENS] [B_TOKENS] [B_RUN]      defaults: 40000 30000 7

Order: A, B, A, B, A (plain alternation), then B_RUN requests of B in a row, then A again - the request that matters:
its prompt must come from the cache as well as after a single request of B, not be read again from the start.
Writes logs\\alttest.json; exit 1 when A lost its cache.
"""
import json, os, random, sys, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT, MODEL = sys.argv[1], sys.argv[2]
A_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
B_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 30000
B_RUN = int(sys.argv[5]) if len(sys.argv) > 5 else 7


def doc(seed, lines):
    rnd = random.Random(seed)
    words = ("amber basalt cedar delta ember fjord garnet harbor indigo jasper kelp lumen marble nectar onyx prism "
             "quartz river sable tundra umber velvet willow xenon yarrow zephyr").split()
    out = []
    for i in range(lines):
        w = [rnd.choice(words) for _ in range(6)]
        out.append(f"record {seed}-{i:05d}: owner {w[0]} {w[1]}, site {w[2]}, load {rnd.randint(100, 9999)} kg, "
                   f"route {w[3]}-{w[4]}, status {w[5]}, code {rnd.randint(100000, 999999)}")
    return "\n".join(out)


class Chat:
    def __init__(self, name, seed, tokens):
        self.name, self.seed, self.turn = name, seed, 0
        self.messages = [
            {"role": "system", "content": f"Ты помощник по учёту грузов, журнал {name}. Отвечай одной короткой строкой."},
            {"role": "user", "content": f"Журнал {name}:\n" + doc(seed, max(40, tokens // 38)) + "\n\nПодтверди получение."}]

    def step(self):
        self.turn += 1
        if self.turn > 1:
            # like a tool result of an agent: some new text and the next question
            self.messages.append({"role": "user", "content": f"Дополнение {self.turn} к журналу {self.name}:\n"
                                  + doc(self.seed * 100 + self.turn, 12)
                                  + f"\n\nНазови статус записи {self.seed}-{self.turn * 7:05d}."})
        body = {"model": MODEL, "messages": self.messages, "temperature": 0, "max_tokens": 60, "stream": False,
                "reasoning_effort": "none"}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.loads(r.read().decode("utf-8"))
        m = d["choices"][0]["message"]
        self.messages.append({"role": "assistant", "content": m.get("content") or ""})
        u = d.get("usage", {})
        row = {"chat": self.name, "turn": self.turn, "prompt": u.get("prompt_tokens"),
               "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
               "seconds": round(time.time() - t0, 2), "answer": (m.get("content") or "").strip()[:60]}
        print(f"{row['chat']}{row['turn']}: prompt {row['prompt']}, cached {row['cached']} "
              f"({100 * row['cached'] / max(row['prompt'], 1):.1f} %), {row['seconds']} s | {row['answer']!r}", flush=True)
        return row


def main():
    a, b = Chat("A", 11, A_TOKENS), Chat("B", 22, B_TOKENS)
    rows = [c.step() for c in (a, b, a, b, a)]
    rows += [b.step() for _ in range(B_RUN)]
    last = a.step()
    rows.append(last)
    plain = rows[4]                              # A after one request of B: what plain alternation gives
    ok = last["cached"] > 0 and last["cached"] >= plain["cached"] - 8192
    print(("OK: " if ok else "FAIL: ") + f"A after {B_RUN} requests of B in a row: {last['cached']} of {last['prompt']} "
          f"tokens from the cache, {last['seconds']} s (after one request of B: {plain['cached']} of {plain['prompt']}, "
          f"{plain['seconds']} s)")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    json.dump({"a_tokens": A_TOKENS, "b_tokens": B_TOKENS, "b_run": B_RUN, "ok": ok, "rows": rows},
              open(os.path.join(HERE, "logs", "alttest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if ok else 1)


main()
