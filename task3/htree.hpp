// REKURENCYJNA (HIERARCHICZNA) KOMPRESJA MACIERZY Z CZESCIOWYM SVD
//
// CreateTree(blok, r, eps):
//   [U, D, V] = truncatedSVD(blok, r + 1)
//   jesli D(r+1, r+1) < eps  -> lisc: blok przyblizony macierza rzedu r (CompressMatrix)
//   w przeciwnym razie       -> wezel z czterema synami (cwiartki bloku), rekurencja
//
// Lisc rzedu k przechowuje U (wiersze x k) oraz V = D * V^T (k x kolumny), czyli
// k * (wiersze + kolumny) liczb zamiast wiersze * kolumny. Blok zerowy ma rzad 0.
// Zakresy indeksow sa polotwarte: wiersze [t0, t1), kolumny [s0, s1).

#pragma once

#include <memory>
#include <vector>

#include "image.hpp"
#include "matrix.hpp"

struct Node {
    int t0, t1, s0, s1;
    int rank = 0;
    std::vector<double> sigma;  // zachowane wartosci osobliwe
    Matrix U;                   // (t1 - t0) x rank
    Matrix V;                   // rank x (s1 - s0), juz pomnozone przez D
    std::vector<std::unique_ptr<Node>> sons;
    bool leaf() const { return sons.empty(); }
};

inline bool isZero(const Matrix& M) {
    for (double x : M.a)
        if (x != 0.0) return false;
    return true;
}

inline std::unique_ptr<Node> compressMatrix(int t0, int t1, int s0, int s1, const SVD& svd, int r) {
    auto v = std::make_unique<Node>();
    v->t0 = t0, v->t1 = t1, v->s0 = s0, v->s1 = s1;
    // Rzad r, ale bez zerowych wartosci osobliwych (np. blok 1 x 1 ma tylko jedna).
    int rank = 0;
    while (rank < r && svd.sigma[rank] > 1e-9 * svd.sigma[0]) ++rank;
    v->rank = rank;
    v->sigma.assign(svd.sigma.begin(), svd.sigma.begin() + rank);
    v->U = Matrix(t1 - t0, rank);
    v->V = Matrix(rank, s1 - s0);
    for (int k = 0; k < rank; ++k) {
        for (int i = 0; i < t1 - t0; ++i) v->U(i, k) = svd.U(i, k);
        for (int j = 0; j < s1 - s0; ++j) v->V(k, j) = svd.sigma[k] * svd.V(j, k);
    }
    return v;
}

inline std::unique_ptr<Node> createTree(const Matrix& A, int t0, int t1, int s0, int s1, int r, double eps) {
    Matrix block = subMatrix(A, t0, t1, s0, s1);
    if (isZero(block)) {
        auto v = std::make_unique<Node>();
        v->t0 = t0, v->t1 = t1, v->s0 = s0, v->s1 = s1;
        return v;
    }
    SVD svd = truncatedSVD(block, r + 1);
    const bool single = (t1 - t0 == 1 && s1 - s0 == 1);  // bloku 1 x 1 nie da sie podzielic
    if (svd.sigma[r] < eps || single) return compressMatrix(t0, t1, s0, s1, svd, r);

    auto v = std::make_unique<Node>();
    v->t0 = t0, v->t1 = t1, v->s0 = s0, v->s1 = s1;
    const int tm = t0 + (t1 - t0 + 1) / 2, sm = s0 + (s1 - s0 + 1) / 2;
    const int rows[3] = {t0, tm, t1}, cols[3] = {s0, sm, s1};
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            if (rows[a] < rows[a + 1] && cols[b] < cols[b + 1])  // pomijamy puste cwiartki
                v->sons.push_back(createTree(A, rows[a], rows[a + 1], cols[b], cols[b + 1], r, eps));
    return v;
}

// Odtworzenie macierzy z drzewa: w kazdym lisciu blok = U * V.
inline void decompress(const Node& v, Matrix& out) {
    if (!v.leaf()) {
        for (const auto& s : v.sons) decompress(*s, out);
        return;
    }
    for (int i = v.t0; i < v.t1; ++i)
        for (int j = v.s0; j < v.s1; ++j) {
            double x = 0.0;
            for (int k = 0; k < v.rank; ++k) x += v.U(i - v.t0, k) * v.V(k, j - v.s0);
            out(i, j) = x;
        }
}

struct TreeStats {
    long long leaves = 0, storage = 0;  // liczba lisci i liczba przechowywanych liczb
    int maxRank = 0, depth = 0;
};

inline void stats(const Node& v, TreeStats& st, int depth = 0) {
    st.depth = std::max(st.depth, depth);
    if (!v.leaf()) {
        for (const auto& s : v.sons) stats(*s, st, depth + 1);
        return;
    }
    st.leaves += 1;
    st.storage += static_cast<long long>(v.rank) * ((v.t1 - v.t0) + (v.s1 - v.s0));
    st.maxRank = std::max(st.maxRank, v.rank);
}

// RYSOWACZ SKOMPRESOWANEJ MACIERZY
// Kazdy lisc to prostokat z szara ramka. Lisc rzedu k ma czarny pasek szerokosci k
// wzdluz lewej krawedzi (kolumny U) i czarny pasek wysokosci k wzdluz gornej krawedzi
// (wiersze V). Czarne pole odpowiada wiec pamieci zajmowanej przez lisc.
inline void drawNode(const Node& v, Image& img) {
    if (!v.leaf()) {
        for (const auto& s : v.sons) drawNode(*s, img);
        return;
    }
    auto paint = [&](int i, int j, uint8_t c) {
        for (int ch = 0; ch < 3; ++ch) img.at(i, j, ch) = c;
    };
    for (int j = v.s0; j < v.s1; ++j) paint(v.t0, j, 170);  // ramka (gorna i lewa krawedz)
    for (int i = v.t0; i < v.t1; ++i) paint(i, v.s0, 170);
    for (int i = v.t0; i < v.t1; ++i)
        for (int j = v.s0; j < v.s1; ++j)
            if (i - v.t0 < v.rank || j - v.s0 < v.rank) paint(i, j, 0);
}

inline Image drawTree(const Node& root) {
    Image img(root.s1 - root.s0, root.t1 - root.t0, 255);
    drawNode(root, img);
    for (int i = 0; i < img.height; ++i)  // prawa i dolna krawedz calej macierzy
        for (int ch = 0; ch < 3; ++ch) img.at(i, img.width - 1, ch) = std::min<uint8_t>(img.at(i, img.width - 1, ch), 170);
    for (int j = 0; j < img.width; ++j)
        for (int ch = 0; ch < 3; ++ch) img.at(img.height - 1, j, ch) = std::min<uint8_t>(img.at(img.height - 1, j, ch), 170);
    return img;
}
