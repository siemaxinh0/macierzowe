#!/bin/sh
# Kompilacja na macOS/Linux do folderu bin/ (na Windows: build.ps1).
cd "$(dirname "$0")" && mkdir -p bin || exit 1
for p in inverse test_mul lu; do
    [ -f $p.cpp ] && { g++ -O2 -std=c++17 -o bin/$p $p.cpp || exit 1; }
done
echo "OK: $(ls bin | tr '\n' ' ')"
