"""Przygotowuje bitmape 512 x 512 w formacie PPM dla programu compress.exe.

Uruchomienie (z folderu task3/report):  python prepare_image.py [plik_obrazu]
Bez argumentu bierze zdjecie Grace Hopper z przykladowych danych matplotlib
(domena publiczna, US Navy). Obraz jest przycinany do kwadratu i skalowany do 512 x 512.
Wynik: data/bitmapa.ppm oraz fig/bitmapa.png (do sprawozdania).
"""
import os
import sys

from matplotlib import cbook
from PIL import Image

N = 512
os.makedirs("data", exist_ok=True)
os.makedirs("fig", exist_ok=True)

path = sys.argv[1] if len(sys.argv) > 1 else cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
img = Image.open(path).convert("RGB")
w, h = img.size
side = min(w, h)
top = (h - side) // 4          # przy zdjeciu pionowym tniemy glownie od dolu
left = (w - side) // 2
img = img.crop((left, top, left + side, top + side)).resize((N, N), Image.LANCZOS)

img.save("data/bitmapa.ppm")
img.save("fig/bitmapa.png")
print(f"{path} -> data/bitmapa.ppm ({N} x {N})")
