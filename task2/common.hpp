// WSPOLNE NARZEDZIA DLA ZADANIA 2 (odwracanie, LU)
//
// Klasa Matrix, licznik operacji g_flops i podstawowe operacje na blokach,
// przeniesione z task1 (binet.cpp, strassen.cpp, laderman.cpp) bez zmian w
// sposobie liczenia operacji. Do tego:
//   * sub(), negate()   - odejmowanie i zmiana znaku bloku (1 operacja na element),
//   * MulFn, FlopsFn    - wspolny interfejs metod mnozenia: kazda metoda
//                         (Binet, Strassen, Laderman) mnozy macierze PROSTOKATNE
//                         (p x q) * (q x r) i ma model liczby operacji F(p, q, r).
//
// LICZENIE OPERACJI
// Zliczamy + - * / na liczbach. Kopiowanie blokow, generowanie macierzy i
// weryfikacja wyniku (multiplyReference, maxDifference) NIE sa liczone.

#pragma once

#include <algorithm>   // std::max, std::min
#include <cmath>       // std::fabs
#include <random>      // generator liczb losowych
#include <vector>

// Licznik operacji zmiennoprzecinkowych (+, -, *, /).
inline unsigned long long g_flops = 0;

// Prosta klasa macierzy prostokatnej, elementy w jednym wektorze wierszami:
// element (i, j) lezy pod indeksem i*cols + j.
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> data;

    Matrix() = default;
    // Tworzy macierz r x c wypelniona zerami.
    Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0) {}

    double& operator()(int i, int j) { return data[static_cast<size_t>(i) * cols + j]; }
    double operator()(int i, int j) const { return data[static_cast<size_t>(i) * cols + j]; }
};

// Metoda mnozenia (p x q) * (q x r) i jej model liczby operacji.
using MulFn = Matrix (*)(const Matrix&, const Matrix&);
using FlopsFn = unsigned long long (*)(int p, int q, int r);

// Macierz n x n z liczbami losowymi z przedzialu (1e-8, 1.0), rozklad rownomierny.
inline Matrix randomMatrix(int n, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(0.00000001, 1.0);
    Matrix m(n, n);
    for (double& x : m.data) x = dist(gen);
    return m;
}

inline Matrix identity(int n) {
    Matrix I(n, n);
    for (int i = 0; i < n; ++i) I(i, i) = 1.0;
    return I;
}

// Wycina blok o lewym gornym rogu (r0, c0) i rozmiarze h x w (tylko kopiowanie).
inline Matrix subMatrix(const Matrix& M, int r0, int c0, int h, int w) {
    Matrix R(h, w);
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) R(i, j) = M(r0 + i, c0 + j);
    return R;
}

// Wkleja blok S do macierzy M w pozycji (r0, c0) (odwrotnosc subMatrix).
inline void placeBlock(Matrix& M, const Matrix& S, int r0, int c0) {
    for (int i = 0; i < S.rows; ++i)
        for (int j = 0; j < S.cols; ++j) M(r0 + i, c0 + j) = S(i, j);
}

// Dodawanie macierzy o tych samych wymiarach: 1 operacja na element.
inline Matrix add(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, A.cols);
    for (size_t k = 0; k < C.data.size(); ++k) C.data[k] = A.data[k] + B.data[k];
    g_flops += C.data.size();
    return C;
}

// Odejmowanie macierzy o tych samych wymiarach: 1 operacja na element.
inline Matrix sub(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, A.cols);
    for (size_t k = 0; k < C.data.size(); ++k) C.data[k] = A.data[k] - B.data[k];
    g_flops += C.data.size();
    return C;
}

// Zmiana znaku -A. Liczymy ja jak odejmowanie 0 - a: 1 operacja na element.
inline Matrix negate(const Matrix& A) {
    Matrix C(A.rows, A.cols);
    for (size_t k = 0; k < C.data.size(); ++k) C.data[k] = -A.data[k];
    g_flops += C.data.size();
    return C;
}

// Liczba operacji klasycznego mnozenia (p x q) * (q x r): p*r*(2q - 1).
inline unsigned long long naiveCost(unsigned long long p, unsigned long long q, unsigned long long r) {
    return p * r * (2 * q - 1);
}

// Klasyczne mnozenie macierzy prostokatnych (p x q) * (q x r).
// Uzywane w przypadkach bazowych rekurencji. Liczba operacji: naiveCost(p, q, r).
inline Matrix multiplyNaive(const Matrix& A, const Matrix& B) {
    const int p = A.rows, q = A.cols, r = B.cols;
    Matrix C(p, r);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < r; ++j) {
            double sum = A(i, 0) * B(0, j);
            for (int k = 1; k < q; ++k) sum += A(i, k) * B(k, j);
            C(i, j) = sum;
        }
    }
    g_flops += naiveCost(p, q, r);
    return C;
}

// Referencyjne mnozenie klasyczne - NIE zwieksza licznika g_flops.
// Sluzy tylko do sprawdzania poprawnosci wynikow.
inline Matrix multiplyReference(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k)
            for (int j = 0; j < B.cols; ++j) C(i, j) += A(i, k) * B(k, j);
    return C;
}

// Najwieksza roznica bezwzgledna miedzy odpowiadajacymi sobie elementami.
inline double maxDifference(const Matrix& A, const Matrix& B) {
    double d = 0.0;
    for (size_t k = 0; k < A.data.size(); ++k) d = std::max(d, std::fabs(A.data[k] - B.data[k]));
    return d;
}
