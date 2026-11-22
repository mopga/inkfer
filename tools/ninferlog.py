"""Totals of a NInfer request log (ninfer-serve --log-stats-panel on): requests, cache share, prefill and decode.

    python ninferlog.py LOG [MIN_PROMPT]      MIN_PROMPT: list the requests with less than 50 % from the cache and a
                                              prompt of at least so many tokens (default 20000)
"""
import re, sys

LINE = re.compile(r"^(\S+ \S+)\s+INFO\s+req#(\d+) done \| (.*)$")


def seconds(text):
    m = re.fullmatch(r"(?:(\d+)m )?([\d.]+)(ms|s)", text.strip())
    if not m:
        return 0.0
    v = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
    return v + 60 * int(m.group(1) or 0)


def number(text):
    text = text.replace(",", "")
    return float(text[:-1]) * 1000 if text.endswith("k") else float(text)


def main():
    path = sys.argv[1]
    floor = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = LINE.match(line)
        if not m:
            continue
        f = {"time": m.group(1), "id": int(m.group(2)), "path": ""}
        for part in m.group(3).split(" | "):
            part = part.strip()
            if part.startswith("prompt "):
                f["prompt"] = int(number(part.split()[1]))
            elif part.startswith("output "):
                f["output"] = int(number(part.split()[1]))
            elif part.startswith("cache "):
                f["cache"] = int(number(part.split()[1]))
                p = re.search(r"\(([^)]*)\)", part)
                f["path"] = p.group(1) if p else ""
            elif part.startswith("TTFT "):
                f["ttft"] = seconds(part[5:])
            elif part.startswith("total "):
                f["total"] = seconds(part[6:])
            elif part.startswith("decode "):
                f["decode"] = number(part.split()[1])
            elif part.startswith("dspark accepted "):
                a, b = part.split()[2].split("/")
                f["acc"], f["drafted"] = int(number(a)), int(number(b))
        if "prompt" in f:
            rows.append(f)
    if not rows:
        print("no finished requests in the log")
        return
    prompt = sum(r["prompt"] for r in rows)
    cache = sum(r.get("cache", 0) for r in rows)
    out = sum(r.get("output", 0) for r in rows)
    gen_s = sum(r["output"] / r["decode"] for r in rows if r.get("decode") and r.get("output"))
    total = sum(r.get("total", 0) for r in rows)
    ttft = sum(r.get("ttft", 0) for r in rows)
    print(f"requests {len(rows)}; first {rows[0]['time']}, last {rows[-1]['time']}")
    print(f"prompt tokens {prompt:,}: from cache {cache:,} ({100 * cache / prompt:.1f} %), read {prompt - cache:,}")
    print(f"generated {out:,} tokens in {gen_s:,.0f} s of decode: {out / gen_s:.1f} tok/s" if gen_s else "no decode")
    print(f"sum of request times {total:,.0f} s, of which before the first token {ttft:,.0f} s")
    print(f"largest prompt {max(r['prompt'] for r in rows):,}; largest output {max(r.get('output', 0) for r in rows):,}")
    if any("drafted" in r for r in rows):
        acc, dr = sum(r.get("acc", 0) for r in rows), sum(r.get("drafted", 0) for r in rows)
        print(f"dspark drafts accepted {acc:,} of {dr:,} ({100 * acc / max(dr, 1):.1f} %)")
    paths = {}
    for r in rows:
        key = r["path"].split(", ")[-1] if r["path"] else "no cache"
        paths[key] = paths.get(key, 0) + 1
    print("cache paths:", ", ".join(f"{k}: {v}" for k, v in sorted(paths.items(), key=lambda x: -x[1])))
    low = [r for r in rows if r["prompt"] >= floor and r.get("cache", 0) < 0.5 * r["prompt"]]
    print(f"requests with a prompt of {floor:,}+ tokens and under 50 % from the cache: {len(low)}")
    for r in low:
        print(f"  {r['time']} req#{r['id']}: prompt {r['prompt']:,}, cache {r.get('cache', 0):,} ({r['path']}), "
              f"first token after {r.get('ttft', 0):.0f} s")


main()
