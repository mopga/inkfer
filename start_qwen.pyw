"""Back to Qwen3.8-Flash-Next (Strata): unload Inkling, point the agents at Qwen again, start the old launcher.

    pythonw start_qwen.pyw [--web | --tui | --engine | --app | --stop-only]
        --web        Strata engine without the app's window + omp-web      (start_strata_ompweb.pyw)   - default
        --tui        Strata engine + oh-my-pi in a terminal                (start_strata_omp.pyw)
        --engine     Strata engine only                                    (start_engine_iq3s.pyw)
        --app        the Strata desktop app with Qwen loaded               (start_strata_iq3s.pyw)
        --stop-only  only unload Inkling and switch the agents' default model back to Qwen

The old launchers and the Strata settings are not changed; this file only frees the video memory first.
Run with Strata's bundled pythonw.exe. Standard library only.
"""
import importlib.machinery, importlib.util, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TUNING = HERE.parent
OLD = {"--web": "start_strata_ompweb.pyw", "--tui": "start_strata_omp.pyw", "--engine": "start_engine_iq3s.pyw",
       "--app": "start_strata_iq3s.pyw"}


def load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def main(argv):
    ink = load(HERE / "start_inkling.pyw", "inkling_start")
    ink.TITLE = "Возврат на Qwen (Strata)"
    was = ink.stop_inkling_engines()
    ink.note("stopped: " + (", ".join(was) or "nothing was running"))
    try:
        (HERE / "logs" / "current-profile.txt").unlink()
    except OSError:
        pass
    changed = ink.set_default_model(ink.QWEN_ROLE)
    mode = next((a for a in argv if a in OLD or a == "--stop-only"), "--web")
    if changed and ink.listening(30141):                  # a running omp-web still points its new sessions at Inkling
        if mode == "--web":
            if ink.stop_web():                            # the old launcher below starts it again and opens the page
                ink.note("omp-web stopped: it has to read the new default model")
        else:
            ink.restart_stale_web(False)
    if mode == "--stop-only":
        return
    old = load(TUNING / OLD[mode], "qwen_start")
    old.main()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except SystemExit:
        raise
    except Exception as e:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, f"Неожиданная ошибка лаунчера:\n{e!r}", "Возврат на Qwen (Strata)", 0x10)
        sys.exit(1)
