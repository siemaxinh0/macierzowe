"""Rysunki do sprawozdania z zadania 3 (z wynikow programu compress.exe).

Uruchomienie (z folderu task3/report):  python make_figures.py
Wejscie: data/bitmapa.ppm, data/singular_values.csv, data/grid.csv, out/<konfiguracja>_*.ppm
Wynik:   fig/singular_values.pdf, fig/grid.pdf, fig/oryginal.png, fig/<konfiguracja>.png
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 8})
CONFIGS = ["r1_s1", "r1_slast", "r1_shalf", "r4_s1", "r4_slast", "r4_shalf", "best"]
GAP = 8
COLORS = {"R": "tab:red", "G": "tab:green", "B": "tab:blue"}


def montage(top, bottom, big):
    """Dwa rzedy po 3 kafelki 512 x 512 i duzy kafelek 1024 x 1024 po prawej."""
    t = top[0].width
    canvas = Image.new("RGB", (3 * t + 3 * GAP + big.width, 2 * t + GAP), "white")
    for k, im in enumerate(top):
        canvas.paste(im, (k * (t + GAP), 0))
    for k, im in enumerate(bottom):
        canvas.paste(im, (k * (t + GAP), t + GAP))
    canvas.paste(big, (3 * (t + GAP), 0))
    return canvas


def tinted(arr, ch):
    out = np.zeros_like(arr)
    out[:, :, ch] = arr[:, :, ch]
    return Image.fromarray(out)


# Bitmapa nieskompresowana: calosc i osobno R, G, B.
orig = np.asarray(Image.open("data/bitmapa.ppm").convert("RGB"))
tiles = [Image.fromarray(orig)] + [tinted(orig, ch) for ch in range(3)]
canvas = Image.new("RGB", (4 * 512 + 3 * GAP, 512), "white")
for k, im in enumerate(tiles):
    canvas.paste(im, (k * (512 + GAP), 0))
canvas.save("fig/oryginal.png")

# Skompresowane macierze, kanaly i bitmapa RGB dla kazdej konfiguracji.
for c in CONFIGS:
    top = [Image.open(f"out/{c}_tree_{ch}.ppm").convert("RGB") for ch in "RGB"]
    bottom = [Image.open(f"out/{c}_{ch}.ppm").convert("RGB") for ch in "RGB"]
    big = Image.open(f"out/{c}_rgb.ppm").convert("RGB").resize((1024 + GAP, 1024 + GAP), Image.NEAREST)
    montage(top, bottom, big).save(f"fig/{c}.png", optimize=True)

# Wartosci osobliwe calych macierzy R, G, B.
sv = np.genfromtxt("data/singular_values.csv", delimiter=",", names=True)
fig, ax = plt.subplots(figsize=(5.6, 2.6))
for ch in "RGB":
    ax.semilogy(sv["i"], sv[ch], lw=1, color=COLORS[ch], label=ch)
ax.set_xlabel("indeks $i$")
ax.set_ylabel(r"wartość osobliwa $\sigma_i$")
ax.set_xlim(1, len(sv))
ax.legend()
fig.tight_layout()
fig.savefig("fig/singular_values.pdf")

# Przeszukanie parametrow: pamiec a jakosc.
grid = np.genfromtxt("data/grid.csv", delimiter=",", names=True)
fig, ax = plt.subplots(figsize=(5.6, 2.6))
for r in np.unique(grid["r"]):
    m = grid["r"] == r
    ax.plot(100 * grid["ratio"][m], grid["psnr"][m], "-o", ms=3, lw=1, label=f"$r={int(r)}$")
    for x, y, d in zip(100 * grid["ratio"][m], grid["psnr"][m], grid["delta"][m]):
        if r == 1:
            ax.annotate(f"{int(d)}", (x, y), textcoords="offset points", xytext=(3, -9), fontsize=6)
ax.set_xlabel("pamięć drzewa [% liczby pikseli]")
ax.set_ylabel("PSNR [dB]")
ax.legend(ncol=5, loc="lower right")
fig.tight_layout()
fig.savefig("fig/grid.pdf")
print("OK")
