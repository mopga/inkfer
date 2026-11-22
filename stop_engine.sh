#!/usr/bin/env bash
# Stop a test engine and everything it started. stop_engine.sh basalt|llama|ninfer
case "${1:-}" in
  basalt) IMG=basalt.exe ;;
  llama)  IMG=llama-server.exe ;;
  ninfer) IMG=ninfer-serve.exe ;;
  *) echo "usage: stop_engine.sh basalt|llama|ninfer"; exit 2 ;;
esac
taskkill //F //T //IM "$IMG" 2>&1 | tail -3
for i in 1 2 3 4 5 6 7 8 9 10; do tasklist 2>/dev/null | grep -qi "$IMG" || break; ping -n 2 127.0.0.1 >/dev/null; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
