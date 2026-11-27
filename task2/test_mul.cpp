// TEST METOD MNOZENIA DLA MACIERZY PROSTOKATNYCH
//
//   test_mul.exe [N]                  -> dla kazdej metody z algorithms.hpp i wszystkich
//                                        p, q, r = 1..N (domyslnie 30) oraz 40 losowych
//                                        wiekszych rozmiarow: wynik == mnozenie
//                                        referencyjne i g_flops == model
//   test_mul.exe --square ALG nMax    -> CSV "n,flops" z modelu dla n x n (do porownania
//                                        z task1: <alg>.exe --flops 1 nMax)

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "algorithms.hpp"

Matrix randomRect(int p, int q, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(0.00000001, 1.0);
    Matrix m(p, q);
    for (double& x : m.data) x = dist(gen);
    return m;
}

// Zwraca false i wypisuje komunikat, jesli wynik albo licznik sie nie zgadza.
bool checkOne(const Algorithm& alg, int p, int q, int r, std::mt19937& gen, double& worst) {
    Matrix A = randomRect(p, q, gen), B = randomRect(q, r, gen);
    g_flops = 0;
    Matrix C = alg.mul(A, B);
    const unsigned long long counted = g_flops;
    const double err = maxDifference(C, multiplyReference(A, B)) / q;  // elementy wyniku sa rzedu q
    worst = std::max(worst, err);
    if (C.rows != p || C.cols != r || err > 1e-12 || counted != alg.flops(p, q, r)) {
        std::cout << "  BLAD " << alg.name << " " << p << "x" << q << " * " << q << "x" << r
                  << ": blad/q = " << err << ", licznik " << counted << ", model " << alg.flops(p, q, r) << "\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--square") {
        const Algorithm* alg = findAlgorithm(argv[2]);
        if (!alg) return 1;
        std::cout << "n,flops\n";
        for (int n = 1; n <= std::atoi(argv[3]); ++n) std::cout << n << "," << alg->flops(n, n, n) << "\n";
        return 0;
    }

    const int N = argc > 1 ? std::atoi(argv[1]) : 30;
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> big(31, 200);
    int failures = 0;
    for (const Algorithm& alg : algorithms()) {
        int tested = 0, failed = 0;
        double worst = 0.0;
        for (int p = 1; p <= N; ++p)
            for (int q = 1; q <= N; ++q)
                for (int r = 1; r <= N; ++r, ++tested) failed += !checkOne(alg, p, q, r, gen, worst);
        for (int t = 0; t < 40; ++t, ++tested) failed += !checkOne(alg, big(gen), big(gen), big(gen), gen, worst);
        std::cout << std::setw(9) << alg.name << ": " << tested << " rozmiarow, bledow " << failed
                  << ", max |C - C_ref| / q = " << std::scientific << std::setprecision(2) << worst << "\n";
        failures += failed;
    }
    return failures == 0 ? 0 : 1;
}
