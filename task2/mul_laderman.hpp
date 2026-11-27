// MNOZENIE METODA LADERMANA DLA MACIERZY PROSTOKATNYCH
//
// Uogolnienie task1/laderman.cpp z (n x n) * (n x n) na (p x q) * (q x r).
// Odwracanie i LU dziela macierz na bloki o roznych wymiarach (np. A12 ma
// h x (n-h)), wiec mnozenie musi przyjmowac macierze prostokatne.
//
// p, q, r PODZIELNE PRZEZ 3: kazda macierz dzielimy na 9 blokow:
// bloki A maja (p/3) x (q/3), bloki B (q/3) x (r/3), bloki C (p/3) x (r/3).
// Wzory Ladermana (23 mnozenia) sa tozsamosciami w pierscieniu nieprzemiennym
// (w kazdym iloczynie czynnik z A stoi po lewej, z B po prawej), wiec dzialaja
// dla dowolnych blokow, byle wymiary do siebie pasowaly.
// Koszt dodawan na poziom: 28 sum blokow A, 28 sum blokow B i 42 dodawania
// przy skladaniu C (razem 98, tak jak w task1 dla p = q = r).
//
// W PRZECIWNYM RAZIE - "obieranie" po kazdym wymiarze osobno: odcinamy
// p mod 3, q mod 3 i r mod 3 ostatnich wierszy/kolumn. Rdzen ma wszystkie
// wymiary podzielne przez 3, a pozostale iloczyny maja ktorys wymiar <= 2:
//
//    C11 = A11*B11 + a12*b21        A11: pm x qm   a12: pm x qs
//    C12 = A11*b12 + a12*b22        a21: ps x qm   a22: ps x qs
//    C21 = a21*B11 + a22*b21        (pm = p - p mod 3, ps = p mod 3, itd.)
//    C22 = a21*b12 + a22*b22
//
// Skladniki z pustym blokiem (np. qs = 0) pomijamy. Wszystkie iloczyny liczymy
// rekurencyjnie, a te "cienkie" od razu trafiaja do przypadku bazowego.
// Dla p = q = r daje to dokladnie te same operacje co task1/laderman.cpp.
//
// PRZYPADEK BAZOWY: min(p, q, r) <= 2 - mnozenie klasyczne.

#pragma once

#include <functional>         // std::reference_wrapper
#include <initializer_list>
#include <map>
#include <tuple>

#include "common.hpp"

// Kombinacja liniowa blokow: plus[0] + plus[1] + ... - minus[0] - minus[1] - ...
// Liczba operacji: (liczba skladnikow - 1) na kazdy element wyniku.
using MatrixList = std::initializer_list<std::reference_wrapper<const Matrix>>;

inline Matrix combine(MatrixList plus, MatrixList minus = {}) {
    auto it = plus.begin();
    Matrix C = *it;  // kopia pierwszego skladnika (bez operacji)
    for (++it; it != plus.end(); ++it) {
        const Matrix& X = *it;
        for (size_t k = 0; k < C.data.size(); ++k) C.data[k] += X.data[k];
        g_flops += C.data.size();
    }
    for (const Matrix& X : minus) {
        for (size_t k = 0; k < C.data.size(); ++k) C.data[k] -= X.data[k];
        g_flops += C.data.size();
    }
    return C;
}

inline Matrix multiplyLaderman(const Matrix& A, const Matrix& B) {
    const int p = A.rows, q = A.cols, r = B.cols;

    if (std::min({p, q, r}) <= 2) return multiplyNaive(A, B);

    // Obieranie: rdzen pm x qm razy qm x rm i paski o szerokosci 1 lub 2.
    if (p % 3 != 0 || q % 3 != 0 || r % 3 != 0) {
        const int pm = p - p % 3, qm = q - q % 3, rm = r - r % 3;
        const int ps = p - pm, qs = q - qm, rs = r - rm;

        Matrix A11 = subMatrix(A, 0, 0, pm, qm), a12 = subMatrix(A, 0, qm, pm, qs);
        Matrix a21 = subMatrix(A, pm, 0, ps, qm), a22 = subMatrix(A, pm, qm, ps, qs);
        Matrix B11 = subMatrix(B, 0, 0, qm, rm), b12 = subMatrix(B, 0, rm, qm, rs);
        Matrix b21 = subMatrix(B, qm, 0, qs, rm), b22 = subMatrix(B, qm, rm, qs, rs);

        Matrix C(p, r);
        Matrix C11 = multiplyLaderman(A11, B11);
        if (qs > 0) C11 = add(C11, multiplyLaderman(a12, b21));
        placeBlock(C, C11, 0, 0);
        if (rs > 0) {
            Matrix C12 = multiplyLaderman(A11, b12);
            if (qs > 0) C12 = add(C12, multiplyLaderman(a12, b22));
            placeBlock(C, C12, 0, rm);
        }
        if (ps > 0) {
            Matrix C21 = multiplyLaderman(a21, B11);
            if (qs > 0) C21 = add(C21, multiplyLaderman(a22, b21));
            placeBlock(C, C21, pm, 0);
        }
        if (ps > 0 && rs > 0) {
            Matrix C22 = multiplyLaderman(a21, b12);
            if (qs > 0) C22 = add(C22, multiplyLaderman(a22, b22));
            placeBlock(C, C22, pm, rm);
        }
        return C;
    }

    // p, q, r podzielne przez 3: wlasciwy algorytm Ladermana.
    const int pb = p / 3, qb = q / 3, rb = r / 3;
    auto blockA = [&](int i, int j) { return subMatrix(A, i * pb, j * qb, pb, qb); };
    auto blockB = [&](int i, int j) { return subMatrix(B, i * qb, j * rb, qb, rb); };

    Matrix A11 = blockA(0, 0), A12 = blockA(0, 1), A13 = blockA(0, 2);
    Matrix A21 = blockA(1, 0), A22 = blockA(1, 1), A23 = blockA(1, 2);
    Matrix A31 = blockA(2, 0), A32 = blockA(2, 1), A33 = blockA(2, 2);

    Matrix B11 = blockB(0, 0), B12 = blockB(0, 1), B13 = blockB(0, 2);
    Matrix B21 = blockB(1, 0), B22 = blockB(1, 1), B23 = blockB(1, 2);
    Matrix B31 = blockB(2, 0), B32 = blockB(2, 1), B33 = blockB(2, 2);

    // 23 rekurencyjne mnozenia (zamiast 27). combine({plus...}, {minus...}).
    Matrix M1  = multiplyLaderman(combine({A11, A12, A13}, {A21, A22, A32, A33}), B22);
    Matrix M2  = multiplyLaderman(combine({A11}, {A21}), combine({B22}, {B12}));
    Matrix M3  = multiplyLaderman(A22, combine({B12, B21, B33}, {B11, B22, B23, B31}));
    Matrix M4  = multiplyLaderman(combine({A21, A22}, {A11}), combine({B11, B22}, {B12}));
    Matrix M5  = multiplyLaderman(add(A21, A22), combine({B12}, {B11}));
    Matrix M6  = multiplyLaderman(A11, B11);
    Matrix M7  = multiplyLaderman(combine({A31, A32}, {A11}), combine({B11, B23}, {B13}));
    Matrix M8  = multiplyLaderman(combine({A31}, {A11}), combine({B13}, {B23}));
    Matrix M9  = multiplyLaderman(add(A31, A32), combine({B13}, {B11}));
    Matrix M10 = multiplyLaderman(combine({A11, A12, A13}, {A22, A23, A31, A32}), B23);
    Matrix M11 = multiplyLaderman(A32, combine({B13, B21, B32}, {B11, B22, B23, B31}));
    Matrix M12 = multiplyLaderman(combine({A32, A33}, {A13}), combine({B22, B31}, {B32}));
    Matrix M13 = multiplyLaderman(combine({A13}, {A33}), combine({B22}, {B32}));
    Matrix M14 = multiplyLaderman(A13, B31);
    Matrix M15 = multiplyLaderman(add(A32, A33), combine({B32}, {B31}));
    Matrix M16 = multiplyLaderman(combine({A22, A23}, {A13}), combine({B23, B31}, {B33}));
    Matrix M17 = multiplyLaderman(combine({A13}, {A23}), combine({B23}, {B33}));
    Matrix M18 = multiplyLaderman(add(A22, A23), combine({B33}, {B31}));
    Matrix M19 = multiplyLaderman(A12, B21);
    Matrix M20 = multiplyLaderman(A23, B32);
    Matrix M21 = multiplyLaderman(A21, B13);
    Matrix M22 = multiplyLaderman(A31, B12);
    Matrix M23 = multiplyLaderman(A33, B33);

    Matrix C11 = combine({M6, M14, M19});
    Matrix C12 = combine({M1, M4, M5, M6, M12, M14, M15});
    Matrix C13 = combine({M6, M7, M9, M10, M14, M16, M18});
    Matrix C21 = combine({M2, M3, M4, M6, M14, M16, M17});
    Matrix C22 = combine({M2, M4, M5, M6, M20});
    Matrix C23 = combine({M14, M16, M17, M18, M21});
    Matrix C31 = combine({M6, M7, M8, M11, M12, M13, M14});
    Matrix C32 = combine({M12, M13, M14, M15, M22});
    Matrix C33 = combine({M6, M7, M8, M9, M23});

    Matrix C(p, r);
    placeBlock(C, C11, 0, 0);       placeBlock(C, C12, 0, rb);       placeBlock(C, C13, 0, 2 * rb);
    placeBlock(C, C21, pb, 0);      placeBlock(C, C22, pb, rb);      placeBlock(C, C23, pb, 2 * rb);
    placeBlock(C, C31, 2 * pb, 0);  placeBlock(C, C32, 2 * pb, rb);  placeBlock(C, C33, 2 * pb, 2 * rb);
    return C;
}

// Model liczby operacji multiplyLaderman (ta sama rekurencja, tylko sumuje koszty).
inline unsigned long long flopsLaderman(int p, int q, int r) {
    static std::map<std::tuple<int, int, int>, unsigned long long> memo;
    if (std::min({p, q, r}) <= 2) return naiveCost(p, q, r);
    auto key = std::make_tuple(p, q, r);
    auto it = memo.find(key);
    if (it != memo.end()) return it->second;

    unsigned long long total;
    if (p % 3 != 0 || q % 3 != 0 || r % 3 != 0) {
        const int pm = p - p % 3, qm = q - q % 3, rm = r - r % 3;
        const int ps = p - pm, qs = q - qm, rs = r - rm;
        total = flopsLaderman(pm, qm, rm);
        if (qs > 0) total += flopsLaderman(pm, qs, rm) + 1ULL * pm * rm;
        if (rs > 0) {
            total += flopsLaderman(pm, qm, rs);
            if (qs > 0) total += flopsLaderman(pm, qs, rs) + 1ULL * pm * rs;
        }
        if (ps > 0) {
            total += flopsLaderman(ps, qm, rm);
            if (qs > 0) total += flopsLaderman(ps, qs, rm) + 1ULL * ps * rm;
        }
        if (ps > 0 && rs > 0) {
            total += flopsLaderman(ps, qm, rs);
            if (qs > 0) total += flopsLaderman(ps, qs, rs) + 1ULL * ps * rs;
        }
    } else {
        const unsigned long long pb = p / 3, qb = q / 3, rb = r / 3;
        total = 23 * flopsLaderman(p / 3, q / 3, r / 3)
              + 28 * pb * qb + 28 * qb * rb + 42 * pb * rb;
    }
    memo[key] = total;
    return total;
}
