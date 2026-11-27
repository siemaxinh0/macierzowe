// MACIERZ I CZESCIOWE SVD
//
// truncatedSVD(A, k) zwraca k najwiekszych wartosci osobliwych A wraz z wektorami:
//   A ~ U * diag(sigma) * V^T,   U: m x k,  V: n x k,  sigma[0] >= sigma[1] >= ...
// Gdy k > min(m, n), brakujace wartosci osobliwe sa rowne 0.
//
// Metoda: jednostronny algorytm Jacobiego (Hestenesa). Obracamy pary kolumn macierzy
// tak dlugo, az wszystkie kolumny sa do siebie prostopadle. Wtedy dlugosci kolumn to
// wartosci osobliwe, znormalizowane kolumny to wektory U, a zlozenie obrotow to V.
// Metoda jest prosta i dokladna (takze dla malych wartosci osobliwych).

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

struct Matrix {
    int rows = 0, cols = 0;
    std::vector<double> a;  // wierszami

    Matrix() = default;
    Matrix(int r, int c, double v = 0.0) : rows(r), cols(c), a(static_cast<size_t>(r) * c, v) {}
    double& operator()(int i, int j) { return a[static_cast<size_t>(i) * cols + j]; }
    double operator()(int i, int j) const { return a[static_cast<size_t>(i) * cols + j]; }
};

// Blok A[t0:t1, s0:s1] (przedzialy polotwarte).
inline Matrix subMatrix(const Matrix& A, int t0, int t1, int s0, int s1) {
    Matrix B(t1 - t0, s1 - s0);
    for (int i = t0; i < t1; ++i)
        for (int j = s0; j < s1; ++j) B(i - t0, j - s0) = A(i, j);
    return B;
}

struct SVD {
    Matrix U;                   // m x k
    std::vector<double> sigma;  // k
    Matrix V;                   // n x k
};

inline double dot(const std::vector<double>& x, const std::vector<double>& y) {
    double s = 0.0;
    for (size_t i = 0; i < x.size(); ++i) s += x[i] * y[i];
    return s;
}

inline SVD truncatedSVD(const Matrix& A, int k) {
    // Pracujemy na macierzy B (m x n, m >= n): B = A albo B = A^T.
    const bool transposed = A.rows < A.cols;
    const int m = transposed ? A.cols : A.rows;
    const int n = transposed ? A.rows : A.cols;

    std::vector<std::vector<double>> w(n, std::vector<double>(m));  // kolumny B
    std::vector<std::vector<double>> v(n, std::vector<double>(n, 0.0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) w[j][i] = transposed ? A(j, i) : A(i, j);
        v[j][j] = 1.0;
    }

    const double tol = 1e-13;
    std::vector<double> norm2(n);
    for (int sweep = 0; sweep < 60; ++sweep) {
        for (int j = 0; j < n; ++j) norm2[j] = dot(w[j], w[j]);
        bool rotated = false;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double alpha = norm2[p], beta = norm2[q];
                if (alpha == 0.0 || beta == 0.0) continue;
                const double gamma = dot(w[p], w[q]);
                if (std::fabs(gamma) <= tol * std::sqrt(alpha * beta)) continue;
                rotated = true;

                // Obrot, po ktorym kolumny p i q sa prostopadle.
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = c * t;
                for (int i = 0; i < m; ++i) {
                    const double x = w[p][i], y = w[q][i];
                    w[p][i] = c * x - s * y;
                    w[q][i] = s * x + c * y;
                }
                for (int i = 0; i < n; ++i) {
                    const double x = v[p][i], y = v[q][i];
                    v[p][i] = c * x - s * y;
                    v[q][i] = s * x + c * y;
                }
                norm2[p] = alpha - t * gamma;
                norm2[q] = beta + t * gamma;
            }
        }
        if (!rotated) break;
    }

    // Wartosci osobliwe = dlugosci kolumn; sortujemy malejaco.
    std::vector<double> sig(n);
    for (int j = 0; j < n; ++j) sig[j] = std::sqrt(dot(w[j], w[j]));
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int x, int y) { return sig[x] > sig[y]; });

    SVD r;
    r.sigma.assign(k, 0.0);
    Matrix Ub(m, k), Vb(n, k);  // rozklad B = Ub * diag(sigma) * Vb^T
    for (int c = 0; c < std::min(k, n); ++c) {
        const int j = order[c];
        r.sigma[c] = sig[j];
        for (int i = 0; i < m; ++i) Ub(i, c) = sig[j] > 0 ? w[j][i] / sig[j] : 0.0;
        for (int i = 0; i < n; ++i) Vb(i, c) = v[j][i];
    }
    // A = B^T = Vb * diag(sigma) * Ub^T, wiec przy transpozycji U i V zamieniaja sie rolami.
    r.U = transposed ? Vb : Ub;
    r.V = transposed ? Ub : Vb;
    return r;
}
