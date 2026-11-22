"""Check of the NInfer "trail points" patch (engines\\ninfer-trail): a prompt that parts from the cached conversation
in the middle must resume from a trail point and give the same greedy answer as a prompt computed from scratch.

    python trailtest.py PORT MODEL STAGE [TOKENS]
        STAGE  trail : cold request, then requests that diverge at the end / after a side request / in the middle
               ref   : the same diverging requests on an engine without the trail (NINFER_TRAIL_GRID=0), each computed
                       from whatever the stock cache gives
        TOKENS approximate size of the conversation in tokens (default 45000)
Writes logs\\trailtest-STAGE.json; compare with:  python trailtest.py - - compare
"""
import json, os, random, sys, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(HERE, "logs")


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


def conversation(tokens, question, middle="Принято, первая часть журнала сохранена."):
    lines = max(40, tokens // 2 // 38)          # one record line is about 38 tokens
    return [
        {"role": "system", "content": "Ты аккуратный помощник по учёту грузов. Отвечай кратко и точно."},
        {"role": "user", "content": "Первая часть журнала:\n" + doc(11, lines)},
        {"role": "assistant", "content": middle},
        {"role": "user", "content": "Вторая часть журнала:\n" + doc(22, lines)},
        {"role": "assistant", "content": "Принято, вторая часть журнала сохранена."},
        {"role": "user", "content": question},
    ]


def ask(port, model, messages, max_tokens=400):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "reasoning_effort": "low"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read().decode("utf-8"))
    u = d.get("usage", {})
    m = d["choices"][0]["message"]
    return {"seconds": round(time.time() - t0, 2), "prompt": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "text": (m.get("reasoning_content") or m.get("reasoning") or "") + " ## " + (m.get("content") or "")}


QUESTIONS = {
    "q1": "Назови код и статус записи 22-00007 и владельца записи 11-00003.",
    "q2": "Сколько килограммов в записи 22-00011 и какой у неё маршрут? Назови также площадку записи 11-00005.",
    "q3": "Какой статус у записи 11-00009 и какой код у записи 22-00002?",
    "q4": "Назови владельца и груз записи 22-00004 и маршрут записи 11-00001.",
}


def main():
    port, model, stage = sys.argv[1], sys.argv[2], sys.argv[3]
    if stage == "compare":
        a = json.load(open(os.path.join(LOGS, "trailtest-trail.json"), encoding="utf-8"))
        b = json.load(open(os.path.join(LOGS, "trailtest-ref.json"), encoding="utf-8"))
        for k in b:
            same = a[k]["text"] == b[k]["text"]
            print(f"{k}: same answer {same} | trail: cached {a[k]['cached']}/{a[k]['prompt']} in {a[k]['seconds']} s"
                  f" | ref: cached {b[k]['cached']}/{b[k]['prompt']} in {b[k]['seconds']} s")
            if not same:
                print("   trail:", a[k]["text"][:300].replace("\n", " "))
                print("   ref  :", b[k]["text"][:300].replace("\n", " "))
        return
    tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 45000
    out = {}
    side = [{"role": "system", "content": "Ты придумываешь короткие заголовки."},
            {"role": "user", "content": "Придумай заголовок из трёх слов для этого текста:\n" + doc(33, 60)}]
    if stage == "trail":
        out["cold_q1"] = ask(port, model, conversation(tokens, QUESTIONS["q1"]))
    out["end_q2"] = ask(port, model, conversation(tokens, QUESTIONS["q2"]))
    out["side"] = ask(port, model, side, 40)
    out["after_side_q3"] = ask(port, model, conversation(tokens, QUESTIONS["q3"]))
    out["middle_q4"] = ask(port, model, conversation(tokens, QUESTIONS["q4"], middle="Первая часть получена."))
    for k, v in out.items():
        print(f"{k}: prompt {v['prompt']} cached {v['cached']} in {v['seconds']} s | {v['text'][-110:]!r}")
    json.dump(out, open(os.path.join(LOGS, f"trailtest-{stage}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


main()
