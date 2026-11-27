// MNOZENIE METODA BINET'A (z task1/binet.cpp, bez zmian w algorytmie)
//
// Podzial na 4 bloki w punktach ceil(x/2), 8 rekurencyjnych mnozen i 4 dodawania.
// Dziala dla macierzy prostokatnych (p x q) * (q x r); przypadek bazowy, gdy
// ktorys wymiar wynosi 1.

#pragma once

#include <map>
#include <tuple>

#include "common.hpp"

inline Matrix multiplyBinet(const Matrix& A, const Matrix& B) {
    const int p = A.rows, q = A.cols, r = B.cols;

    if (p == 1 || q == 1 || r == 1) return multiplyNaive(A, B);

    // Gorna/lewa czesc dostaje ceil(x/2) elementow, dolna/prawa - reszte.
    const int ph = (p + 1) / 2, qh = (q + 1) / 2, rh = (r + 1) / 2;

    Matrix A11 = subMatrix(A, 0,  0,  ph,     qh);
    Matrix A12 = subMatrix(A, 0,  qh, ph,     q - qh);
    Matrix A21 = subMatrix(A, ph, 0,  p - ph, qh);
    Matrix A22 = subMatrix(A, ph, qh, p - ph, q - qh);

    Matrix B11 = subMatrix(B, 0,  0,  qh,     rh);
    Matrix B12 = subMatrix(B, 0,  rh, qh,     r - rh);
    Matrix B21 = subMatrix(B, qh, 0,  q - qh, rh);
    Matrix B22 = subMatrix(B, qh, rh, q - qh, r - rh);

    Matrix C11 = add(multiplyBinet(A11, B11), multiplyBinet(A12, B21));
    Matrix C12 = add(multiplyBinet(A11, B12), multiplyBinet(A12, B22));
    Matrix C21 = add(multiplyBinet(A21, B11), multiplyBinet(A22, B21));
    Matrix C22 = add(multiplyBinet(A21, B12), multiplyBinet(A22, B22));

    Matrix C(p, r);
    placeBlock(C, C11, 0,  0);
    placeBlock(C, C12, 0,  rh);
    placeBlock(C, C21, ph, 0);
    placeBlock(C, C22, ph, rh);
    return C;
}

// Model liczby operacji multiplyBinet (ta sama rekurencja, tylko sumuje koszty).
// Dla macierzy kwadratowych daje 2n^3 - n^2, tak jak w task1.
inline unsigned long long flopsBinet(int p, int q, int r) {
    static std::map<std::tuple<int, int, int>, unsigned long long> memo;
    if (p == 1 || q == 1 || r == 1) return naiveCost(p, q, r);
    auto key = std::make_tuple(p, q, r);
    auto it = memo.find(key);
    if (it != memo.end()) return it->second;

    const int ph = (p + 1) / 2, qh = (q + 1) / 2, rh = (r + 1) / 2;
    const int pl = p - ph, ql = q - qh, rl = r - rh;
    unsigned long long total = 1ULL * p * r;  // 4 dodawania blokow = p*r operacji
    total += flopsBinet(ph, qh, rh) + flopsBinet(ph, ql, rh);  // C11
    total += flopsBinet(ph, qh, rl) + flopsBinet(ph, ql, rl);  // C12
    total += flopsBinet(pl, qh, rh) + flopsBinet(pl, ql, rh);  // C21
    total += flopsBinet(pl, qh, rl) + flopsBinet(pl, ql, rl);  // C22
    memo[key] = total;
    return total;
}
