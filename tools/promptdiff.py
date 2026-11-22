"""Where consecutive requests of one conversation, recorded by logproxy.py, stop being a continuation of each other.

    python promptdiff.py OUT_DIR

Conversations are told apart by their first user message. For each request: number of messages, size, and against the
previous request of the same conversation: the first message that differs (index, role, the place in its text).
"""
import glob, json, os, sys


def flat(message):
    return json.dumps(message, ensure_ascii=False, sort_keys=True)


def main():
    last = {}
    for path in sorted(glob.glob(os.path.join(sys.argv[1], "[0-9]*.json"))):
        try:
            body = json.load(open(path, encoding="utf-8"))
        except ValueError:
            continue
        messages = body.get("messages") or []
        if not messages:
            continue
        first_user = next((flat(m) for m in messages if m.get("role") == "user"), "")
        key = hash(first_user[:2000])
        head = {k: v for k, v in body.items() if k != "messages"}
        tools = flat(head.pop("tools", None))
        line = f"{os.path.basename(path)} conv {abs(key) % 1000:03d} messages {len(messages)} chars {sum(len(flat(m)) for m in messages)}"
        prev = last.get(key)
        if prev:
            pm, ptools, phead = prev
            same = 0
            while same < min(len(pm), len(messages)) and flat(pm[same]) == flat(messages[same]):
                same += 1
            if same == len(pm):
                line += f" | continues the previous one (+{len(messages) - same} messages)"
            else:
                a, b = flat(pm[same]), flat(messages[same])
                at = 0
                while at < min(len(a), len(b)) and a[at] == b[at]:
                    at += 1
                before = sum(len(flat(m)) for m in messages[:same]) + at
                line += (f" | DIFFERS at message {same} ({messages[same].get('role')}), char {at} of the message, "
                         f"{before} chars from the start\n      was: {a[max(0, at - 60):at + 160]!r}\n      now: {b[max(0, at - 60):at + 160]!r}")
            if tools != ptools:
                line += " | TOOLS CHANGED"
            changed = [k for k in set(head) | set(phead) if head.get(k) != phead.get(k)]
            if changed:
                line += f" | other fields changed: {changed}"
        print(line)
        last[key] = (messages, tools, head)


main()
