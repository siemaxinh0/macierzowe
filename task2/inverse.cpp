// ZADANIE 2, pkt 1 i 3: rekurencyjne odwracanie macierzy
// z mnozeniem metoda Binet'a, Strassena albo Ladermana (algorytm: inverse.hpp).
//
// KOMPILACJA I URUCHOMIENIE
//   g++ -O2 -std=c++17 -o inverse.exe inverse.cpp
//   inverse.exe --alg laderman               -> test dla domyslnej listy rozmiarow
//   inverse.exe --alg laderman 5 16 100      -> test dla podanych n
//
// Tryby uzywane do sprawozdania (wynik w formacie CSV na stdout):
//   inverse.exe --alg X --flops nMin nMax             -> liczba operacji z modelu
//   inverse.exe --alg X --time nMin nMax krok powt    -> czas (minimum z powtorzen),
//                                                        licznik operacji i blad
//   inverse.exe --alg X --check nMin nMax             -> licznik == model i blad dla kazdego n
//   inverse.exe --alg X --example                     -> przyklad 3x3 krok po kroku
// X = binet | laderman | strassen (gdy jest mul_strassen.hpp). Domyslnie binet.
//
// Blad = max |(A * A^-1 - I)_ij|, liczony mnozeniem referencyjnym (poza licznikiem).

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "algorithms.hpp"
#include "inverse.hpp"

double inverseError(const Matrix& A, const Matrix& Ainv) {
    return maxDifference(multiplyReference(A, Ainv), identity(A.rows));
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    std::string algName = "binet";
    if (args.size() >= 2 && args[0] == "--alg") {
        algName = args[1];
        args.erase(args.begin(), args.begin() + 2);
    }
    const Algorithm* alg = findAlgorithm(algName);
    if (!alg) {
        std::cerr << "Nieznana metoda: " << algName << ". Dostepne:";
        for (const Algorithm& a : algorithms()) std::cerr << " " << a.name;
        std::cerr << "\n";
        return 1;
    }
    std::map<int, unsigned long long> memo;
    auto model = [&](int n) { return flopsInverse(n, alg->flops, memo); };

    // --flops nMin nMax: CSV "n,flops" z modelu (bez odwracania macierzy).
    if (args.size() == 3 && args[0] == "--flops") {
        std::cout << "n,flops\n";
        for (int n = std::atoi(args[1].c_str()); n <= std::atoi(args[2].c_str()); ++n)
            std::cout << n << "," << model(n) << "\n";
        return 0;
    }

    // --time nMin nMax krok powtorzenia: CSV "n,flops,time_ms,error".
    if (args.size() == 5 && args[0] == "--time") {
        const int nMin = std::atoi(args[1].c_str()), nMax = std::atoi(args[2].c_str());
        const int step = std::atoi(args[3].c_str()), reps = std::atoi(args[4].c_str());
        std::mt19937 gen(12345);
        std::cout << "n,flops,time_ms,error\n";
        for (int n = nMin; n <= nMax; n += step) {
            Matrix A = randomMatrix(n, gen);
            Matrix Ainv;
            double best = 1e300;
            unsigned long long flops = 0;
            for (int k = 0; k < reps; ++k) {
                g_flops = 0;
                auto t0 = std::chrono::steady_clock::now();
                Ainv = inverse(A, alg->mul);
                auto t1 = std::chrono::steady_clock::now();
                flops = g_flops;
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::cout << n << "," << flops << "," << std::fixed << std::setprecision(4) << best << ","
                      << std::scientific << std::setprecision(3) << inverseError(A, Ainv)
                      << std::defaultfloat << std::endl;
        }
        return 0;
    }

    // --check nMin nMax: dla kazdego n licznik g_flops musi byc rowny modelowi.
    if (args.size() == 3 && args[0] == "--check") {
        const int nMin = std::atoi(args[1].c_str()), nMax = std::atoi(args[2].c_str());
        std::mt19937 gen(12345);
        int mismatches = 0, worstN = nMin;
        double worst = 0.0;
        for (int n = nMin; n <= nMax; ++n) {
            Matrix A = randomMatrix(n, gen);
            g_flops = 0;
            Matrix Ainv = inverse(A, alg->mul);
            if (g_flops != model(n)) {
                ++mismatches;
                std::cout << "n = " << n << ": licznik " << g_flops << " != model " << model(n) << "\n";
            }
            const double err = inverseError(A, Ainv);
            if (err > worst) worst = err, worstN = n;
        }
        std::cout << alg->name << ", n = " << nMin << ".." << nMax << ": niezgodnosci licznik/model = "
                  << mismatches << ", max blad = " << std::scientific << std::setprecision(3) << worst
                  << " (n = " << worstN << ")\n";
        return mismatches == 0 ? 0 : 1;
    }

    // --example: macierz 3x3 z przykladu recznego w sprawozdaniu.
    if (args.size() == 1 && args[0] == "--example") {
        const double a[3][3] = {{2, 1, 1}, {4, 3, 3}, {8, 7, 9}};
        Matrix A(3, 3);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) A(i, j) = a[i][j];
        g_flops = 0;
        g_trace = true;
        Matrix Ainv = inverse(A, alg->mul);
        g_trace = false;
        std::cout << "\n% " << alg->name << ", liczba operacji = " << g_flops << " (model: " << model(3)
                  << "), max |A*A^-1 - I| = " << inverseError(A, Ainv) << "\n";
        std::cout << std::setprecision(17);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) std::cout << Ainv(i, j) + 0.0 << (j < 2 ? " " : "\n");
        return 0;
    }

    std::vector<int> sizes;
    for (const std::string& s : args) sizes.push_back(std::atoi(s.c_str()));
    if (sizes.empty()) sizes = {1, 2, 3, 4, 5, 7, 8, 9, 10, 16, 17, 27, 32, 50, 64, 81, 100};

    std::mt19937 gen(12345);
    std::cout << "Rekurencyjne odwracanie macierzy, mnozenie: " << alg->name
              << " (liczby z (1e-8, 1.0))\n\n";
    std::cout << std::setw(6) << "n" << std::setw(18) << "operacje (+-*/)" << std::setw(18) << "model"
              << std::setw(14) << "czas [ms]" << std::setw(16) << "|A*A^-1 - I|" << "\n";
    for (int n : sizes) {
        if (n < 1) continue;
        Matrix A = randomMatrix(n, gen);
        g_flops = 0;
        auto t0 = std::chrono::steady_clock::now();
        Matrix Ainv = inverse(A, alg->mul);
        auto t1 = std::chrono::steady_clock::now();
        const unsigned long long flops = g_flops;
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << std::setw(6) << n << std::setw(18) << flops << std::setw(18) << model(n)
                  << std::setw(14) << std::fixed << std::setprecision(3) << ms
                  << std::setw(16) << std::scientific << std::setprecision(2) << inverseError(A, Ainv)
                  << std::defaultfloat << "\n";
    }
    return 0;
}
