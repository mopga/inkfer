"""Markdown table from ctxbench runs: one row per (run, context fill).

  python summarize.py [LABEL ...]        (no labels: every bench/runs/*.json that is a ctxbench run)

Columns: prefill tok/s of the cold read, decode tok/s (engine's figure) min..max over the requests of that fill,
TTFT of the cold read and of the cached follow-up turn, share of the follow-up prompt taken from the cache,
whether the planted fact was found, VRAM, engine RSS, free RAM, CPU, disk reads.
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(path):
    d = json.load(open(path, encoding="utf-8"))
    if "requests" not in d:
        return
    fills = []
    for r in d["requests"]:
        if r.get("fill_target") and r["fill_target"] not in fills:
            fills.append(r["fill_target"])
    for f in fills:
        rs = [r for r in d["requests"] if r.get("fill_target") == f]
        cold = next((r for r in rs if r["tag"] == "cold"), None)
        last = rs[-1]
        dec = [(r.get("engine") or {}).get("predicted_per_second") or r.get("decode_tps_client") for r in rs]
        dec = [x for x in dec if x]
        pre = (cold.get("engine") or {}).get("prompt_per_second") or cold.get("prefill_tps_client") if cold else None
        nr = next((r for r in rs if "needle_ok" in r), None)
        needle = nr["needle_ok"] if nr else None
        loop = next((r["loopiness"] for r in rs if r.get("loopiness") is not None), None)
        sysm = [r["sys"] for r in rs if r.get("sys")]
        cached = (100.0 * last["cached_tokens"] / last["prompt_tokens"]) if last.get("prompt_tokens") else 0
        yield (d["label"], cold["prompt_tokens"] if cold else f, round(pre) if pre else "-",
               f"{min(dec):.0f}–{max(dec):.0f}" if dec else "-", f"{cold['ttft_s']:.1f}" if cold else "-",
               f"{last['ttft_s']:.2f}", f"{cached:.0f}%", (f"{nr['needles_found']}/3" if nr and "needles_found" in nr else {True: "да", False: "НЕТ", None: "-"}[needle])
               + (f", повторы {loop:.2f}" if loop is not None else ""),
               max(s["vram_used_mib_max"] for s in sysm), max(s["engine_rss_gib_max"] for s in sysm),
               min(s["ram_avail_gib_min"] for s in sysm), max(s["cpu_pct_max"] for s in sysm),
               max(s["disk_read_MBps_max"] for s in sysm))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    labels = sys.argv[1:]
    paths = ([os.path.join(HERE, "runs", l + ".json") for l in labels] if labels
             else sorted(p for p in glob.glob(os.path.join(HERE, "runs", "*.json")) if not os.path.basename(p).startswith("det-")))
    print("| Прогон | Промпт, ток. | Prefill, ток/с | Decode, ток/с | TTFT холодный, с | TTFT след. хода, с | Из кэша | Факт найден | VRAM, МиБ | RSS, ГиБ | Своб. RAM, ГиБ | CPU макс, % | Диск макс, МБ/с |")
    print("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for p in paths:
        for r in rows(p) or ():
            print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    main()
