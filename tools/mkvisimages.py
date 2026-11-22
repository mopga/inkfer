"""Test images for vischeck.py: synthetic "screenshots" with text whose content is known exactly.

    python mkvisimages.py            writes bench\\vision\\*.png

- term.png      a terminal window with a Python traceback (file, line number, exception text)
- table.png     a dashboard-like table of services with ports, states and numbers
- code.png      an editor screenshot with a small function containing a bug
- dense.png     a 1920x1080 "full screen" with small 14 px text in two columns
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "bench", "vision")
os.makedirs(OUT, exist_ok=True)


def font(size, mono=True):
    for name in (["consola.ttf", "cour.ttf"] if mono else ["segoeui.ttf", "arial.ttf"]):
        try:
            return ImageFont.truetype(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name), size)
        except OSError:
            pass
    return ImageFont.load_default()


def term():
    lines = [
        "PS C:\\work\\billing> python -m billing.run --month 2026-10",
        "Loading 18,204 invoices from invoices_2026-10.csv",
        "Traceback (most recent call last):",
        '  File "C:\\work\\billing\\billing\\run.py", line 88, in main',
        "    total = summarize(rows, currency=args.currency)",
        '  File "C:\\work\\billing\\billing\\totals.py", line 214, in summarize',
        "    rate = RATES[currency]",
        "KeyError: 'KZT'",
        "PS C:\\work\\billing> ",
    ]
    img = Image.new("RGB", (1100, 330), (12, 12, 12))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1100, 28], fill=(40, 40, 40))
    d.text((12, 5), "Windows PowerShell", font=font(15, False), fill=(220, 220, 220))
    f = font(18)
    for i, s in enumerate(lines):
        col = (255, 110, 110) if s.startswith("KeyError") else (204, 204, 204)
        d.text((12, 40 + i * 31), s, font=f, fill=col)
    img.save(os.path.join(OUT, "term.png"))


def table():
    rows = [("Service", "Port", "State", "p95, ms", "Errors/h"),
            ("gateway", "8443", "running", "41", "0"),
            ("auth", "9001", "running", "17", "2"),
            ("orders", "9100", "degraded", "388", "57"),
            ("search", "9200", "running", "63", "1"),
            ("reports", "9310", "stopped", "-", "-"),
            ("mailer", "9400", "running", "22", "0")]
    img = Image.new("RGB", (900, 420), (250, 250, 252))
    d = ImageDraw.Draw(img)
    d.text((24, 16), "Cluster status - eu-north-2", font=font(26, False), fill=(20, 20, 30))
    xs = [24, 250, 380, 560, 720]
    f = font(20, False)
    for r, row in enumerate(rows):
        y = 80 + r * 46
        if r == 0:
            d.rectangle([16, y - 6, 884, y + 32], fill=(225, 230, 240))
        for x, cell in zip(xs, row):
            col = (20, 20, 30)
            if cell == "degraded":
                col = (200, 120, 0)
            elif cell == "stopped":
                col = (190, 30, 30)
            d.text((x, y), cell, font=f, fill=col)
    img.save(os.path.join(OUT, "table.png"))


def code():
    src = [
        "def average_rating(reviews):",
        '    """Mean of the 1..5 star ratings; reviews without a rating are skipped."""',
        "    total = 0",
        "    count = 0",
        "    for r in reviews:",
        "        if r.get('stars') is None:",
        "            continue",
        "        total += r['stars']",
        "    count += 1",
        "    return total / count if count else 0.0",
    ]
    img = Image.new("RGB", (1000, 400), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1000, 30], fill=(45, 45, 48))
    d.text((12, 6), "ratings.py - editor", font=font(15, False), fill=(200, 200, 200))
    f = font(18)
    for i, s in enumerate(src):
        d.text((12, 44 + i * 33), f"{31 + i:>3}", font=f, fill=(110, 110, 110))
        d.text((64, 44 + i * 33), s, font=f, fill=(212, 212, 212))
    img.save(os.path.join(OUT, "code.png"))


def dense():
    img = Image.new("RGB", (1920, 1080), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = font(14)
    for col in range(2):
        for i in range(60):
            n = col * 60 + i + 1
            s = f"item-{n:03d}  sku=K{(n * 7919) % 100000:05d}  qty={(n * 37) % 91:2d}  bin=R{n % 12:02d}"
            if n == 77:
                s = "item-077  sku=K90816  qty=13  bin=R05  <- RECOUNT REQUESTED BY M. ORLOVA"
            d.text((20 + col * 960, 20 + i * 17), s, font=f, fill=(30, 30, 30))
    img.save(os.path.join(OUT, "dense.png"))


if __name__ == "__main__":
    term(); table(); code(); dense()
    print("written to", OUT, sorted(os.listdir(OUT)))
