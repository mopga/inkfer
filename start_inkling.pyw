"""Start Inkling 2 Mini in one of the launch profiles (inkling\\profiles.json) and, optionally, the agent's window.

    pythonw start_inkling.pyw PROFILE [--web | --tui] [--no-switch]
        PROFILE     a key of "profiles" in profiles.json (speed / balanced / max)
        --web       when the model is ready, open omp-web (browser window of the oh-my-pi agent)
        --tui       when the model is ready, open oh-my-pi in a terminal window
        --no-switch leave the agents' default model as it is (by default it becomes inkling/<model id of the profile>)

What it does:
  1. frees the video memory: asks the Strata host (if it runs) to unload Qwen - the host itself and the desktop app stay
     open - and stops another Inkling engine if one is running;
  2. starts the engine of the profile in a minimised console window on http://127.0.0.1:8081 (this computer only);
     closing that window unloads the model;
  3. waits until the model is loaded, then points oh-my-pi and omp-web at it (modelRoles.default in their config.yml).
Back to Qwen: start_qwen.pyw next to this file (or the old Strata shortcuts after "Inkling: stop").

Run with Strata's bundled pythonw.exe (no console window). Standard library only. Log of the last start: start_inkling.log
"""
import ctypes, json, os, re, socket, subprocess, sys, time, urllib.request, webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
TUNING = HERE.parent
PROFILES = HERE / "profiles.json"
OWN_LOG = Path(__file__).with_suffix(".log")
STRATA = "http://127.0.0.1:47831"
QWEN_ROLE = "strata/iq3_s"
CONFIGS = [TUNING / "harness" / "omp-home" / "config.yml", TUNING / "harness" / "omp-web" / "home" / "config.yml"]
WEB_CMD = TUNING / "harness" / "omp-web.cmd"
WEB_DIR = TUNING / "harness" / "omp-web"
OMP_CMD = TUNING / "harness" / "omp.cmd"
WEB_URL = "http://127.0.0.1:30141"
ENGINE_IMAGES = ("basalt.exe", "ninfer-serve.exe", "llama-server.exe")
NEW_CONSOLE, NO_WINDOW, SW_SHOWMINNOACTIVE = 0x00000010, 0x08000000, 7
TITLE = "Inkling 2 Mini"

_lines = []


def note(text):
    _lines.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}")
    try:
        OWN_LOG.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    except OSError:
        pass


def fail(text):
    note("FAILED: " + text)
    if not os.environ.get("INKLING_NO_DIALOG"):              # unattended checks: the reason is in the log only
        ctypes.windll.user32.MessageBoxW(None, text, TITLE, 0x10)
    sys.exit(1)


def call(base, path, body=None, timeout=20):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, method="GET" if body is None else "POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def running(image):
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"], capture_output=True,
                             text=True, creationflags=NO_WINDOW, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return image.lower() in out.lower()


def stop_inkling_engines():
    """Stop every local Inkling engine with what it started. Returns the images that were running."""
    was = [i for i in ENGINE_IMAGES if running(i)]
    for image in was:
        subprocess.run(["taskkill", "/F", "/T", "/IM", image], capture_output=True, creationflags=NO_WINDOW, timeout=60)
    end = time.time() + 30
    while time.time() < end and any(running(i) for i in was):
        time.sleep(0.5)
    return was


def unload_qwen():
    """Ask the Strata host to unload its models (the Unload button). The host and the app stay open."""
    try:
        state = call(STRATA, "/api/native", timeout=5)
    except Exception:
        return 0                                             # no host: nothing of Strata is in video memory
    n = 0
    for inst in state.get("instances") or []:
        if inst.get("status") in ("ready", "loading") and inst.get("id"):
            try:
                call(STRATA, f"/api/native/instances/{inst['id']}/unload", {}, timeout=120)
                n += 1
            except Exception as e:
                note(f"Strata instance {inst.get('id')} was not unloaded: {e!r}")
    end = time.time() + 60
    while n and time.time() < end:
        try:
            left = [i for i in call(STRATA, "/api/native", timeout=5).get("instances") or []
                    if i.get("status") in ("ready", "loading", "unloading")]
        except Exception:
            break
        if not left:
            break
        time.sleep(1)
    return n


def vram_free_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, creationflags=NO_WINDOW, timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def set_default_model(role):
    """modelRoles.default - and modelRoles.vision where the file has it - in the agents' config.yml files (one line
    each; everything else stays as it is). The vision role describes pictures for a model without image input; it
    follows the loaded model so that it never points at one that is not in video memory (Inkling and Qwen both see).
    Returns True when a file was changed: a running omp-web read the old value at its start and has to be restarted
    (restart_stale_web), or its new sessions ask the engine for a model that is no longer loaded (404)."""
    changed = False
    for cfg in CONFIGS:
        try:
            with open(cfg, encoding="utf-8", newline="") as f:      # newline="": line ends stay as they are
                text = f.read()
        except OSError:
            continue
        new = text
        for key in ("default", "vision"):
            new, n = re.subn(r"(?m)^(modelRoles:[ \t]*\r?\n(?:[ \t]+(?:#.*|\w+:.*)\r?\n)*?[ \t]+" + key + r":[ \t]*)[^\s#]+",
                             lambda m: m.group(1) + role, new, count=1)
            if n:
                note(f"{cfg.parent.name}: modelRoles.{key} = {role}")
        if new != text:
            with open(cfg, "w", encoding="utf-8", newline="") as f:
                f.write(new)
            changed = True
    return changed


def set_console_title(pid, title):
    try:
        k = ctypes.WinDLL("kernel32")
        k.FreeConsole()
        if k.AttachConsole(int(pid)):
            k.SetConsoleTitleW(title)
        k.FreeConsole()
    except Exception as e:
        note(f"console title not set: {e!r}")


def expand(value, port, model_id):
    return (str(value).replace("{HERE}", str(HERE)).replace("{PORT}", str(port)).replace("{MODEL_ID}", model_id)
            .replace("{LOG}", str(HERE / "logs")))


def start_engine(name, prof, port):
    exe = Path(expand(prof["exe"], port, prof["model_id"]))
    if not exe.exists():
        fail(f"Не найден движок профиля «{name}»:\n{exe}")
    args = [expand(a, port, prof["model_id"]) for a in prof["args"]]
    for a in args:
        if re.match(r"^[A-Za-z]:[\\/].*\.(basalt|ninfer|gguf)$", a) and not Path(a).exists():
            fail(f"Не найден файл модели профиля «{name}»:\n{a}")
    env = dict(os.environ)
    for k, v in (prof.get("env") or {}).items():
        env[k] = expand(v, port, prof["model_id"])
    if prof.get("path_prepend"):
        env["PATH"] = os.pathsep.join(expand(p, port, prof["model_id"]) for p in prof["path_prepend"]) + os.pathsep + env.get("PATH", "")
    env["PYTHONUTF8"] = "1"
    (HERE / "logs").mkdir(exist_ok=True)
    title = f"Inkling движок ({name}): закрыть это окно = выгрузить модель"
    quoted = " ".join(f'"{a}"' if (" " in a or not a) else a for a in args)
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = SW_SHOWMINNOACTIVE
    cmd = f'cmd.exe /c "title {title} & "{exe}" {quoted}"'
    pid = subprocess.Popen(cmd, cwd=str(Path(expand(prof.get("cwd") or str(exe.parent), port, prof["model_id"]))), env=env,
                           creationflags=NEW_CONSOLE, startupinfo=si, close_fds=True).pid
    note(f"engine started: {exe.name} {quoted}")
    return pid, title, exe.name


def model_ready(port):
    """Loaded and answering: /health says ok and does not say "loaded": false (Basalt answers ok while it still loads)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:
        return False
    if '"loaded": false' in body or '"loaded":false' in body:
        return False
    return "ok" in body.lower()


def listening(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def web_server_pids():
    """The omp-web of this harness that listens on 30141: its server process and the launcher process above it, top
    first. Only processes started from harness\\omp-web count - anything else on that port is left alone."""
    ps = ("$c = Get-NetTCPConnection -LocalPort 30141 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; "
          "if ($c) { $p = Get-CimInstance Win32_Process -Filter \"ProcessId=$($c.OwningProcess)\"; "
          "$q = if ($p) { Get-CimInstance Win32_Process -Filter \"ProcessId=$($p.ParentProcessId)\" }; "
          "@($q, $p) | Where-Object { $_ } | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" } }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW, timeout=40).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    mark = str(WEB_DIR).lower()
    pids = []
    for line in out.splitlines():
        pid, _, cmd = line.partition("\t")
        if pid.strip().isdigit() and mark in cmd.lower().replace("/", "\\"):
            pids.append(int(pid))
    return pids


def stop_web():
    """Stop the running omp-web of this harness (sessions are on disk and stay). True when the port is free after it."""
    for pid in web_server_pids():
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, creationflags=NO_WINDOW, timeout=60)
    end = time.time() + 20
    while time.time() < end and listening(30141):
        time.sleep(0.5)
    return not listening(30141)


def restart_stale_web(open_page):
    """After the default model changed: omp-web keeps the one it read at its start, so a running omp-web is restarted
    (without opening a page unless asked). Nothing happens when omp-web is not running. True when it was restarted."""
    if not listening(30141):
        return False
    if not stop_web():
        note("omp-web was not restarted: the listener on 30141 is not this harness's omp-web; "
             "choose the model by hand in its window")
        return False
    note("omp-web stopped: it has to read the new default model")
    start_web(open_page)
    return True


def open_web():
    if listening(30141):
        webbrowser.open(WEB_URL)
        note("omp-web already running, page opened")
        return
    start_web(True)


def start_web(open_page):
    title = "omp-web: закрыть это окно = остановить"
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = SW_SHOWMINNOACTIVE
    env = dict(os.environ)
    if not open_page:
        env["OMP_WEB_NO_OPEN"] = "1"                         # omp-web opens the browser itself unless told not to
    pid = subprocess.Popen(f'cmd.exe /c "title {title} & "{WEB_CMD}""', cwd=str(WEB_DIR), creationflags=NEW_CONSOLE,
                           startupinfo=si, close_fds=True, env=env).pid
    end = time.time() + 90
    while time.time() < end:
        if listening(30141):
            time.sleep(3)
            set_console_title(pid, title)
            note(f"omp-web started: {WEB_URL}")
            return
        time.sleep(1)
    fail(f"Сервер omp-web не ответил за 90 секунд на {WEB_URL}.")


def open_tui():
    subprocess.Popen(f'cmd.exe /k "title oh-my-pi + Inkling & "{OMP_CMD}""', cwd=str(Path.home()), creationflags=NEW_CONSOLE,
                     close_fds=True)
    note("oh-my-pi terminal opened")


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    names = [a for a in argv if not a.startswith("--")]
    conf = json.loads(PROFILES.read_text(encoding="utf-8"))
    port = int(conf.get("port", 8081))
    name = names[0] if names else conf.get("default", "balanced")
    prof = (conf.get("profiles") or {}).get(name)
    if not prof:
        fail(f"Нет профиля «{name}» в {PROFILES.name}. Есть: {', '.join(conf.get('profiles') or [])}")
    global TITLE
    TITLE = f"Inkling 2 Mini — {prof.get('title', name)}"
    current = None
    try:
        current = (HERE / "logs" / "current-profile.txt").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    if current == name and model_ready(port) and any(running(i) for i in ENGINE_IMAGES):
        note(f"profile {name} is already loaded on port {port}")
    else:
        n = unload_qwen()
        if n:
            note(f"Strata: unloaded {n} model(s) to free video memory")
        was = stop_inkling_engines()
        if was:
            note("stopped: " + ", ".join(was))
        free = vram_free_mib()
        note(f"video memory free: {free} MiB")
        need = int(prof.get("min_free_vram_mib", 14500))
        if free is not None and free < need:
            fail(f"Свободно {free} МиБ видеопамяти, профилю «{name}» нужно около {need} МиБ.\n"
                 "Закройте программы, занимающие видеопамять (игры, другие модели), и запустите ярлык ещё раз.")
        pid, title, image = start_engine(name, prof, port)
        end = time.time() + int(prof.get("load_timeout_s", 420))
        while time.time() < end:
            if model_ready(port):
                break
            if not running(image):
                fail(f"Движок профиля «{name}» завершился при запуске.\nЖурнал: {HERE / 'logs'}")
            time.sleep(2)
        else:
            fail(f"Модель не загрузилась за {prof.get('load_timeout_s', 420)} с (профиль «{name}»).")
        set_console_title(pid, title)
        (HERE / "logs" / "current-profile.txt").write_text(name, encoding="utf-8")
        note(f"model ready: profile {name}, http://127.0.0.1:{port}/v1, model id {prof['model_id']}")
    restarted = False
    if "--no-switch" not in flags and set_default_model("inkling/" + prof["model_id"]):
        restarted = restart_stale_web("--web" in flags)
    if restarted and "--web" in flags:
        pass                                                 # the restarted omp-web has opened its page itself
    elif "--web" in flags:
        open_web()
    elif "--tui" in flags:
        open_tui()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except SystemExit:
        raise
    except Exception as e:                                    # never die silently: there is no console
        fail(f"Неожиданная ошибка лаунчера:\n{e!r}")
