// LISTA METOD MNOZENIA (wybor przez --alg w inverse.cpp i lu.cpp)
//
// Kazda metoda to plik mul_<nazwa>.hpp z dwiema funkcjami:
//   Matrix multiply<Nazwa>(const Matrix& A, const Matrix& B);   // (p x q) * (q x r)
//   unsigned long long flops<Nazwa>(int p, int q, int r);       // model liczby operacji
// mul_strassen.hpp jest dolaczany automatycznie, jesli istnieje.

#pragma once

#include <string>
#include <vector>

#include "mul_binet.hpp"
#include "mul_laderman.hpp"
#if __has_include("mul_strassen.hpp")
#include "mul_strassen.hpp"
#define HAVE_STRASSEN 1
#endif

struct Algorithm {
    std::string name;
    MulFn mul;
    FlopsFn flops;
};

inline const std::vector<Algorithm>& algorithms() {
    static const std::vector<Algorithm> list = {
        {"binet", multiplyBinet, flopsBinet},
#ifdef HAVE_STRASSEN
        {"strassen", multiplyStrassen, flopsStrassen},
#endif
        {"laderman", multiplyLaderman, flopsLaderman},
    };
    return list;
}

inline const Algorithm* findAlgorithm(const std::string& name) {
    for (const Algorithm& a : algorithms())
        if (a.name == name) return &a;
    return nullptr;
}
