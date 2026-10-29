"""Generuje wykresy do sprawozdania z zadania 2 z plików CSV w folderze data/.

Uruchomienie (z folderu task2/report):  python make_plots.py
Pliki wejściowe: data/<op>-<alg>-{flops,time-a,time-b}.csv, gdzie
  op  = inv (odwracanie) | lu (faktoryzacja LU),
  alg = binet | strassen | laderman.
Rysowane są te pary (op, alg), dla których istnieje plik <op>-<alg>-flops.csv.
Wynik: fig_<op>_time.pdf, fig_<op>_flops.pdf, fig_<op>_error.pdf oraz statystyki na stdout.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 8, "figure.dpi": 150})

DATA = "data"
OPS = {"inv": "odwracanie", "lu": "faktoryzacja LU"}
ALGS = ("binet", "strassen", "laderman")
label = {"binet": "Binét", "strassen": "Strassen", "laderman": "Laderman"}
color = {"binet": "tab:blue", "strassen": "tab:red", "laderman": "tab:green"}


def load(name):
    """Wczytuje CSV (nagłówek w pierwszej linii) jako tablicę numpy."""
    return np.genfromtxt(f"{DATA}/{name}", delimiter=",", names=True)


def fit(x, y):
    """Nachylenie prostej log-log: y ~ c * x^p. Zwraca p."""
    p, _ = np.polyfit(np.log(x), np.log(y), 1)
    return p


for op, op_name in OPS.items():
    algs = [a for a in ALGS if os.path.exists(f"{DATA}/{op}-{a}-flops.csv")]
    if not algs:
        continue
    print(f"=== {op_name} ({op}): {', '.join(algs)} ===")
    flops = {a: load(f"{op}-{a}-flops.csv") for a in algs}
    timing = {a: np.concatenate([load(f"{op}-{a}-time-a.csv"), load(f"{op}-{a}-time-b.csv")]) for a in algs}

    # --- Kontrola: model liczby operacji == licznik zmierzony w czasie działania ---
    for a in algs:
        n = timing[a]["n"].astype(int)
        assert np.array_equal(flops[a]["flops"][n - 1], timing[a]["flops"]), f"model != pomiar ({op}, {a})"
    print("Kontrola: model liczby operacji == licznik zmierzony (wszystkie zmierzone n).")

    if op == "inv" and "binet" in flops:
        n_ax = flops["binet"]["n"].astype(np.int64)
        assert np.array_equal(flops["binet"]["flops"].astype(np.int64), 2 * n_ax ** 3 - n_ax ** 2)
        print("Kontrola: odwracanie z Binétem = 2n^3 - n^2 dla wszystkich n = 1..1000.")

    # --- Wykres 1: czas działania (lewy panel: n<=200, prawy: cały zakres) ---
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.5))
    for a in algs:
        t = timing[a]
        m = t["n"] <= 200
        axs[0].plot(t["n"][m], t["time_ms"][m], "-", lw=1, color=color[a], label=label[a])
        axs[1].plot(t["n"], t["time_ms"] / 1000.0, "-o", ms=2.5, lw=1, color=color[a], label=label[a])
    axs[0].set_xlabel("rozmiar macierzy $n$")
    axs[0].set_ylabel("czas działania [ms]")
    axs[0].set_title("$n = 1,\\ldots,200$", fontsize=8)
    axs[1].set_xlabel("rozmiar macierzy $n$")
    axs[1].set_ylabel("czas działania [s]")
    axs[1].set_title(f"$n = 1,\\ldots,{int(max(timing[a]['n'].max() for a in algs))}$", fontsize=8)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(f"fig_{op}_time.pdf")

    # --- Wykres 2: liczba operacji zmiennoprzecinkowych ---
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    for a in algs:
        ax.plot(flops[a]["n"], flops[a]["flops"], lw=1, color=color[a], label=label[a])
    ax.set_xlabel("rozmiar macierzy $n$")
    ax.set_ylabel("liczba operacji $+-*/$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"fig_{op}_flops.pdf")

    # --- Wykres 3: błąd wyniku (skala logarytmiczna) ---
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    for a in algs:
        t = timing[a]
        ax.semilogy(t["n"], np.maximum(t["error"], 1e-17), ".", ms=2.5, color=color[a], label=label[a])
    ax.set_xlabel("rozmiar macierzy $n$")
    ax.set_ylabel("błąd (max. moduł elementu)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"fig_{op}_error.pdf")

    # --- Statystyki do tekstu sprawozdania ---
    print("Nachylenia log-log (flops dla n>=16, czas dla n>=50):")
    for a in algs:
        f, t = flops[a], timing[a]
        mf, mt = f["n"] >= 16, t["n"] >= 50
        print(f"  {a:9s} flops: {fit(f['n'][mf], f['flops'][mf]):.3f}   czas: {fit(t['n'][mt], t['time_ms'][mt]):.3f}")

    print("Wybrane wartości flops:")
    for n in (3, 10, 100, 243, 500, 729, 1000):
        row = "  ".join(f"{a}={flops[a]['flops'][n - 1]:>12.0f}" for a in algs)
        ratio = ""
        if "binet" in algs and len(algs) > 1:
            ratio = "  " + "  ".join(f"{a[0].upper()}/B={flops[a]['flops'][n - 1] / flops['binet']['flops'][n - 1]:.3f}"
                                     for a in algs if a != "binet")
        print(f"  n={n:5d}  {row}{ratio}")
    for a in algs:
        if a != "binet" and "binet" in algs:
            better = flops[a]["flops"] < flops["binet"]["flops"]
            print(f"Liczba n (1..1000) z {label[a]} < Binét:", int(better.sum()))

    print("Wybrane czasy [ms] i błędy:")
    for a in algs:
        t = timing[a]
        for n in (10, 50, 100, 200, 500, 1000):
            i = np.where(t["n"] == n)[0]
            if len(i):
                print(f"  {a:9s} n={n:5d}  {t['time_ms'][i[0]]:12.3f} ms   błąd {t['error'][i[0]]:.2e}")
        print(f"  {a:9s} mediana błędu: {np.median(t['error']):.2e}, max: {t['error'].max():.2e}"
              f" (n = {int(t['n'][np.argmax(t['error'])])})")
    print("Maksymalne zmierzone n:", {a: int(timing[a]["n"].max()) for a in algs})
    print()
