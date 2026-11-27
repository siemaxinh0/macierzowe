// ZADANIE 3: hierarchiczna kompresja bitmapy z czesciowym SVD
// (algorytm: htree.hpp, SVD: matrix.hpp, bitmapy PPM: image.hpp).
//
// KOMPILACJA
//   g++ -O2 -std=c++17 -o compress.exe compress.cpp
//
// TRYBY
//   compress.exe sv bitmapa.ppm
//       -> wartosci osobliwe calych macierzy R, G, B (CSV: i,R,G,B)
//   compress.exe run bitmapa.ppm r delta prefiks
//       -> kompresja kanalow R, G, B z parametrami r (maksymalny rzad) i delta (prog),
//          zapisuje rysunki prefiks_tree_{R,G,B}.ppm (skompresowane macierze),
//          prefiks_{R,G,B}.ppm (skompresowane kanaly) i prefiks_rgb.ppm (cala bitmapa),
//          na stdout statystyki w CSV.
//       delta: liczba albo wartosc osobliwa danego kanalu: s1, s256, slast, z opcjonalnym
//              dzieleniem, np. s1/2 (sigma_1 / 2).
//   compress.exe example
//       -> maly przyklad z wykladu (macierz 4 x 4 rzedu 2)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

#include "htree.hpp"
#include "image.hpp"
#include "matrix.hpp"

const char* NAMES[3] = {"R", "G", "B"};

// Prog kompresji z opisu delta i wartosci osobliwych calego kanalu.
double parseDelta(const std::string& spec, const std::vector<double>& sigma) {
    if (spec.empty() || spec[0] != 's') return std::stod(spec);
    std::string idx = spec.substr(1);
    double divisor = 1.0;
    const size_t slash = idx.find('/');
    if (slash != std::string::npos) {
        divisor = std::stod(idx.substr(slash + 1));
        idx = idx.substr(0, slash);
    }
    const int i = idx == "last" ? static_cast<int>(sigma.size()) : std::stoi(idx);
    return sigma.at(i - 1) / divisor;
}

std::vector<double> allSingularValues(const Matrix& M) {
    return truncatedSVD(M, std::min(M.rows, M.cols)).sigma;
}

struct ChannelResult {
    double eps = 0, relError = 0, seconds = 0;
    TreeStats st;
    Matrix compressed;
    Image tree;
};

ChannelResult compressChannel(const Matrix& M, int r, const std::string& delta) {
    ChannelResult res;
    auto t0 = std::chrono::steady_clock::now();
    res.eps = parseDelta(delta, allSingularValues(M));
    auto root = createTree(M, 0, M.rows, 0, M.cols, r, res.eps);
    auto t1 = std::chrono::steady_clock::now();
    res.seconds = std::chrono::duration<double>(t1 - t0).count();

    stats(*root, res.st);
    res.compressed = Matrix(M.rows, M.cols);
    decompress(*root, res.compressed);
    res.tree = drawTree(*root);

    double diff = 0, norm = 0;  // blad wzgledny w normie Frobeniusa
    for (size_t i = 0; i < M.a.size(); ++i) {
        diff += (M.a[i] - res.compressed.a[i]) * (M.a[i] - res.compressed.a[i]);
        norm += M.a[i] * M.a[i];
    }
    res.relError = std::sqrt(diff / norm);
    return res;
}

void printTree(const Node& v, int depth = 0) {
    std::printf("%*s[%d:%d, %d:%d] ", 2 * depth, "", v.t0 + 1, v.t1, v.s0 + 1, v.s1);
    if (!v.leaf()) {
        std::printf("wezel, %zu synow\n", v.sons.size());
        for (const auto& s : v.sons) printTree(*s, depth + 1);
        return;
    }
    std::printf("lisc, rzad %d, sigma =", v.rank);
    for (double s : v.sigma) std::printf(" %.4f", s);
    std::printf("\n");
}

int runExample() {
    const double a[4][4] = {{9, 9, 9, 9}, {2, 6, 10, 14}, {11, 15, 19, 23}, {32, 33, 34, 35}};
    Matrix A(4, 4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) A(i, j) = a[i][j];
    std::printf("Wartosci osobliwe A:");
    for (double s : allSingularValues(A)) std::printf(" %.6g", s);
    std::printf("\n");
    for (int r : {1, 2}) {
        auto root = createTree(A, 0, 4, 0, 4, r, 1e-6);
        Matrix C(4, 4);
        decompress(*root, C);
        double err = 0;
        for (size_t i = 0; i < A.a.size(); ++i) err = std::max(err, std::fabs(A.a[i] - C.a[i]));
        TreeStats st;
        stats(*root, st);
        std::printf("\nr = %d, eps = 1e-6: liczb w drzewie = %lld (bez kompresji 16), max |A - A~| = %.2e\n", r,
                    st.storage, err);
        printTree(*root);
    }
    return 0;
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    if (args.size() == 1 && args[0] == "example") return runExample();

    if (args.size() == 2 && args[0] == "sv") {
        Image img = readPPM(args[1]);
        std::vector<double> sv[3];
        std::vector<std::thread> pool;
        for (int ch = 0; ch < 3; ++ch) pool.emplace_back([&, ch] { sv[ch] = allSingularValues(channel(img, ch)); });
        for (auto& t : pool) t.join();
        std::printf("i,R,G,B\n");
        for (size_t i = 0; i < sv[0].size(); ++i) std::printf("%zu,%.10g,%.10g,%.10g\n", i + 1, sv[0][i], sv[1][i], sv[2][i]);
        return 0;
    }

    if (args.size() == 5 && args[0] == "run") {
        Image img = readPPM(args[1]);
        const int r = std::stoi(args[2]);
        const std::string delta = args[3], prefix = args[4];

        ChannelResult res[3];  // trzy kanaly liczymy rownolegle
        std::vector<std::thread> pool;
        for (int ch = 0; ch < 3; ++ch) pool.emplace_back([&, ch] { res[ch] = compressChannel(channel(img, ch), r, delta); });
        for (auto& t : pool) t.join();

        Image out = fromChannels(res[0].compressed, res[1].compressed, res[2].compressed);
        writePPM(prefix + "_rgb.ppm", out);
        std::printf("channel,r,delta,eps,leaves,storage,ratio,max_rank,depth,rel_error,time_s\n");
        long long storage = 0;
        for (int ch = 0; ch < 3; ++ch) {
            writePPM(prefix + "_tree_" + NAMES[ch] + ".ppm", res[ch].tree);
            writePPM(prefix + "_" + NAMES[ch] + ".ppm", singleChannel(res[ch].compressed, ch));
            const ChannelResult& c = res[ch];
            storage += c.st.storage;
            std::printf("%s,%d,%s,%.6g,%lld,%lld,%.4f,%d,%d,%.4e,%.2f\n", NAMES[ch], r, delta.c_str(), c.eps,
                        c.st.leaves, c.st.storage, static_cast<double>(c.st.storage) / (img.width * img.height),
                        c.st.maxRank, c.st.depth, c.relError, c.seconds);
        }
        double mse = 0;  // PSNR bitmapy wynikowej (po zaokragleniu do bajtow)
        for (size_t i = 0; i < img.rgb.size(); ++i) mse += std::pow(double(img.rgb[i]) - out.rgb[i], 2);
        mse /= img.rgb.size();
        std::printf("RGB,%d,%s,,,%lld,%.4f,,,psnr=%.2f,\n", r, delta.c_str(), storage,
                    static_cast<double>(storage) / (3.0 * img.width * img.height), 10 * std::log10(255.0 * 255.0 / mse));
        return 0;
    }

    std::cerr << "Uzycie: compress.exe sv BITMAPA.ppm | run BITMAPA.ppm r delta PREFIKS | example\n";
    return 1;
}
