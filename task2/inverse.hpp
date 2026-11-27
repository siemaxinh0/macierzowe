// REKURENCYJNE ODWRACANIE MACIERZY (eliminacja blokowa, dopelnienie Schura)
//
//        | A11 A12 |      A11: h x h,  A12: h x k,     h = ceil(n/2)
//    A = |         |      A21: k x h,  A22: k x k,     k = n - h
//        | A21 A22 |
//
//    A11i = A11^-1                         (rekurencja)
//    S    = A22 - A21*A11i*A12             (dopelnienie Schura, k x k)
//    Si   = S^-1                           (rekurencja)
//
//            | A11i + A11i*A12*Si*A21*A11i    -A11i*A12*Si |
//    A^-1 =  |                                             |
//            |        -Si*A21*A11i                 Si      |
//
// Wspolne iloczyny liczymy raz: T = A11i*A12, U = A21*A11i, W = T*Si, V = Si*U.
// Wtedy B11 = A11i + W*U, B12 = -W, B21 = -V, B22 = Si. Na poziom rekurencji:
// 2 odwrocenia, 6 mnozen metoda `mul` (Binet, Strassen albo Laderman),
// 1 odejmowanie, 1 dodawanie i 2 zmiany znaku.
//
// PRZYPADEK BAZOWY: n == 1, A^-1 = [1/a] (1 dzielenie).
//
// Algorytm nie wybiera elementu glownego, wiec wymaga, zeby wszystkie wiodace
// minory A byly niezerowe - wtedy A11 i S sa odwracalne na kazdym poziomie
// (det A[1..h+j] = det A11 * det S[1..j]). Dla macierzy losowych zachodzi to
// z prawdopodobienstwem 1.
//
// Zlozonosc: I(n) = 2*I(n/2) + 6*M(n/2) + O(n^2) = O(M(n)) - odwracanie jest
// rzedu mnozenia uzyta metoda.

#pragma once

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "common.hpp"

// Tryb --example: wypisuje bloki posrednie z kazdego poziomu rekurencji.
inline bool g_trace = false;

inline void traceBlocks(int depth, int n, std::initializer_list<std::pair<const char*, const Matrix*>> blocks) {
    const std::string indent(4 * depth, ' ');
    std::cout << indent << "-- n = " << n << " --\n";
    for (const auto& [name, M] : blocks) {
        std::cout << indent << name << " =";
        for (int i = 0; i < M->rows; ++i) {
            std::cout << (i == 0 ? " [" : "; ");
            for (int j = 0; j < M->cols; ++j) std::cout << (j ? " " : "") << M->operator()(i, j) + 0.0;  // + 0.0: bez "-0"
        }
        std::cout << "]\n";
    }
}

inline Matrix inverse(const Matrix& A, MulFn mul, int depth = 0) {
    const int n = A.rows;

    if (n == 1) {
        Matrix R(1, 1);
        R(0, 0) = 1.0 / A(0, 0);
        g_flops += 1;
        if (g_trace) traceBlocks(depth, n, {{"A", &A}, {"A^-1", &R}});
        return R;
    }

    const int h = (n + 1) / 2, k = n - h;
    Matrix A11 = subMatrix(A, 0, 0, h, h), A12 = subMatrix(A, 0, h, h, k);
    Matrix A21 = subMatrix(A, h, 0, k, h), A22 = subMatrix(A, h, h, k, k);

    Matrix A11i = inverse(A11, mul, depth + 1);
    Matrix T = mul(A11i, A12);           // A11^-1 * A12            (h x k)
    Matrix U = mul(A21, A11i);           // A21 * A11^-1            (k x h)
    Matrix S = sub(A22, mul(A21, T));    // A22 - A21*A11^-1*A12    (k x k)
    Matrix Si = inverse(S, mul, depth + 1);
    Matrix W = mul(T, Si);               // A11^-1 * A12 * S^-1     (h x k)
    Matrix V = mul(Si, U);               // S^-1 * A21 * A11^-1     (k x h)

    Matrix R(n, n);
    placeBlock(R, add(A11i, mul(W, U)), 0, 0);
    placeBlock(R, negate(W), 0, h);
    placeBlock(R, negate(V), h, 0);
    placeBlock(R, Si, h, h);

    if (g_trace)
        traceBlocks(depth, n, {{"A", &A}, {"A11^-1", &A11i}, {"T = A11^-1*A12", &T}, {"U = A21*A11^-1", &U},
                               {"S", &S}, {"S^-1", &Si}, {"W = T*S^-1", &W}, {"V = S^-1*U", &V}, {"A^-1", &R}});
    return R;
}

// Model liczby operacji inverse() dla metody mnozenia o modelu `mulFlops`
// (ta sama rekurencja, tylko sumuje koszty). memo - osobna mapa dla kazdej metody.
inline unsigned long long flopsInverse(int n, FlopsFn mulFlops, std::map<int, unsigned long long>& memo) {
    if (n == 1) return 1;
    auto it = memo.find(n);
    if (it != memo.end()) return it->second;

    const int h = (n + 1) / 2, k = n - h;
    unsigned long long total = flopsInverse(h, mulFlops, memo) + flopsInverse(k, mulFlops, memo)
                             + mulFlops(h, h, k) + mulFlops(k, h, h)         // T, U
                             + mulFlops(k, h, k) + 1ULL * k * k              // S = A22 - A21*T
                             + mulFlops(h, k, k) + mulFlops(k, k, h)         // W, V
                             + mulFlops(h, k, h) + 1ULL * h * h              // B11 = A11i + W*U
                             + 2ULL * h * k;                                 // B12 = -W, B21 = -V
    memo[n] = total;
    return total;
}
