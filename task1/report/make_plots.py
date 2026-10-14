"""Generuje wykresy do sprawozdania z plików CSV w folderze data/.

Uruchomienie (z folderu task1/report):  python make_plots.py
Wynik: fig_time.pdf, fig_flops.pdf, fig_loglog.pdf oraz statystyki na stdout.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 8, "figure.dpi": 150})


def load(name):
    """Wczytuje CSV (nagłówek w pierwszej linii) jako tablicę numpy."""
    return np.genfromtxt(f"data/{name}", delimiter=",", names=True)


def load_time(alg):
    """Łączy dwa pliki pomiarowe (gęsty dla małych n i rzadki dla dużych n)."""
    a, b = load(f"{alg}-time-a.csv"), load(f"{alg}-time-b.csv")
    return np.concatenate([a, b])


flops = {alg: load(f"{alg}-flops.csv") for alg in ("binet", "strassen")}
timing = {alg: load_time(alg) for alg in ("binet", "strassen")}
label = {"binet": "Binét", "strassen": "Strassen"}
color = {"binet": "tab:blue", "strassen": "tab:red"}

# --- Kontrola: model liczby operacji == licznik zmierzony w czasie działania ---
for alg in flops:
    n = timing[alg]["n"].astype(int)
    model = flops[alg]["flops"][n - 1]
    assert np.array_equal(model, timing[alg]["flops"]), f"model != pomiar ({alg})"
print("Kontrola: model liczby operacji == licznik zmierzony (wszystkie zmierzone n).")

# --- Kontrola wzorów teoretycznych ---
n_ax = flops["binet"]["n"].astype(np.int64)
assert np.array_equal(flops["binet"]["flops"].astype(np.int64), 2 * n_ax ** 3 - n_ax ** 2), "Binet != 2n^3-n^2"
print("Kontrola: Binet = 2n^3 - n^2 dla wszystkich n = 1..1000.")
for k in range(0, 10):  # Strassen dla n = 2^k: 7*7^k - 6*4^k
    n2 = 2 ** k
    assert int(flops["strassen"]["flops"][n2 - 1]) == 7 * 7 ** k - 6 * 4 ** k, f"Strassen k={k}"
print("Kontrola: Strassen(2^k) = 7*7^k - 6*4^k dla k = 0..9.")

# --- Wykres 1: czas działania (lewy panel: n<=200, prawy: cały zakres) ---
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.5))
for alg in ("binet", "strassen"):
    t = timing[alg]
    m = t["n"] <= 200
    axs[0].plot(t["n"][m], t["time_ms"][m], "-", lw=1, color=color[alg], label=label[alg])
    axs[1].plot(t["n"], t["time_ms"] / 1000.0, "-o", ms=2.5, lw=1, color=color[alg], label=label[alg])
axs[0].set_xlabel("rozmiar macierzy $n$")
axs[0].set_ylabel("czas działania [ms]")
axs[0].set_title("$n = 1,\\ldots,200$", fontsize=8)
axs[1].set_xlabel("rozmiar macierzy $n$")
axs[1].set_ylabel("czas działania [s]")
axs[1].set_title("$n = 1,\\ldots,1000$", fontsize=8)
axs[0].legend()
fig.tight_layout()
fig.savefig("fig_time.pdf")

# --- Wykres 2: liczba operacji zmiennoprzecinkowych ---
fig, ax = plt.subplots(figsize=(5.2, 2.5))
for alg in ("binet", "strassen"):
    ax.plot(flops[alg]["n"], flops[alg]["flops"], lw=1, color=color[alg], label=label[alg])
ax.set_xlabel("rozmiar macierzy $n$")
ax.set_ylabel("liczba operacji $+-*/$")
ax.legend()
fig.tight_layout()
fig.savefig("fig_flops.pdf")


def fit(x, y):
    """Nachylenie prostej log-log: y ~ c * x^p. Zwraca p."""
    p, _ = np.polyfit(np.log(x), np.log(y), 1)
    return p


# --- Wykres 3: skala log-log do oszacowania złożoności ---
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.4))
stats = {}
for alg in ("binet", "strassen"):
    f = flops[alg]
    m = f["n"] >= 16
    p_f = fit(f["n"][m], f["flops"][m])
    t = timing[alg]
    mt = t["n"] >= 50
    p_t = fit(t["n"][mt], t["time_ms"][mt])
    stats[alg] = (p_f, p_t)
    axs[0].loglog(f["n"], f["flops"], lw=1, color=color[alg], label=f"{label[alg]}: nachylenie {p_f:.2f}")
    axs[1].loglog(t["n"], t["time_ms"] / 1000.0, "o", ms=2.5, color=color[alg],
                  label=f"{label[alg]}: nachylenie {p_t:.2f}")
    coef = np.polyfit(np.log(t["n"][mt]), np.log(t["time_ms"][mt] / 1000.0), 1)
    axs[1].loglog(t["n"][mt], np.exp(np.polyval(coef, np.log(t["n"][mt]))), "-", lw=0.7, color=color[alg])
axs[0].set_xlabel("$n$")
axs[0].set_ylabel("liczba operacji")
axs[0].set_title("liczba operacji", fontsize=8)
axs[1].set_xlabel("$n$")
axs[1].set_ylabel("czas [s]")
axs[1].set_title("czas działania", fontsize=8)
for a in axs:
    a.legend(fontsize=6, loc="upper left")
fig.tight_layout()
fig.savefig("fig_loglog.pdf")

# --- Statystyki do tekstu sprawozdania ---
print("\nNachylenia log-log (flops dla n>=16, czas dla n>=50):")
for alg, (p_f, p_t) in stats.items():
    print(f"  {alg:9s} flops: {p_f:.3f}   czas: {p_t:.3f}")

# Strassen tylko dla potęg dwójki (czysta rekurencja, bez obierania)
ns = 2 ** np.arange(2, 10)
fs = flops["strassen"]["flops"][ns - 1]
print(f"  strassen (n=2^k, k=2..9) flops: {fit(ns, fs):.3f}   (teoria: log2(7) = {np.log2(7):.3f})")

print("\nWybrane wartości (n, binet flops, strassen flops, stosunek S/B):")
for n in (16, 64, 100, 128, 256, 500, 512, 1000):
    b = flops["binet"]["flops"][n - 1]
    s = flops["strassen"]["flops"][n - 1]
    print(f"  n={n:5d}  B={b:>12.0f}  S={s:>12.0f}  S/B={s / b:.3f}")

diff = flops["strassen"]["flops"] < flops["binet"]["flops"]
first_ok = [int(n) for n, d in zip(flops["binet"]["n"], diff) if d]
print("\nPierwsze n z Strassen < Binet:", first_ok[:5], "...")
print("Liczba n (1..1000) z Strassen < Binet:", int(diff.sum()))
last_worse = int(flops["binet"]["n"][~diff].max())
print("Ostatnie n z Strassen >= Binet:", last_worse)

print("\nWybrane czasy [ms]:")
for alg in ("binet", "strassen"):
    t = timing[alg]
    for n in (10, 50, 100, 200, 500, 1000):
        i = np.where(t["n"] == n)[0]
        if len(i):
            print(f"  {alg:9s} n={n:5d}  {t['time_ms'][i[0]]:.3f}")
print("Maksymalne zmierzone n:", {a: int(timing[a]['n'].max()) for a in timing})
