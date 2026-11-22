"""Spot check of the "function - line number" references in a Markdown document written by an agent.

    python refcheck.py DOC.md SOURCE.py

A reference is a `name` in backticks followed within 12 characters by a line number or a range (optionally with the
source's file name): `Service.admission_limit()` (4866-4869), `reserve_inference` - serve/server.py:4894, and so on.
It matches when the source defines `name` (def / class) and the number points at the definition (3 lines of slack)
or inside its body. Names the source does not define are skipped, so this is a spot check, not a proof.
"""
import re, sys

doc = open(sys.argv[1], encoding="utf-8").read().splitlines()
src = open(sys.argv[2], encoding="utf-8", errors="replace").read().splitlines()
base = re.escape(sys.argv[2].replace("\\", "/").split("/")[-1])

spans = {}                                   # name -> [(first line, last line of the body)]
stack = []                                   # (indent, name, first line)
for i, line in enumerate(src + ["x"], 1):
    if not line.strip():
        continue
    indent = len(line) - len(line.lstrip())
    while stack and indent <= stack[-1][0] and not line.lstrip().startswith(("#", ")")):
        ind, name, first = stack.pop()
        spans.setdefault(name, []).append((first, i - 1))
    m = re.match(r"\s*(?:async\s+)?(?:def|class)\s+(\w+)", line)
    if m:
        stack.append((indent, m.group(1), i))

REF = re.compile(r"`(?:\w+\.)*(\w+)(?:\([^`]*\))?`[^`\n\d]{0,12}?(?:[\w/\\]*" + base + r":)?(\d{3,5})(?:\s*[-–]\s*(\d{3,5}))?")
good, bad = 0, []
for n, line in enumerate(doc, 1):
    for m in REF.finditer(line):
        name, lo, hi = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        if name not in spans:
            continue
        if any(first - 3 <= lo <= last + 3 and hi <= last + 3 for first, last in spans[name]):
            good += 1
        else:
            bad.append(f"  doc line {n}: {name} -> {lo}-{hi}, defined at {[s[0] for s in spans[name]]}")
print(f"{sys.argv[1]}: {good} references match, {len(bad)} do not")
print("\n".join(bad))
