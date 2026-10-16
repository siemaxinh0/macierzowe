// ============================================================================
// ZADANIE 3: Rekurencyjne mnozenie macierzy metoda Ladermana
// ============================================================================
//
// IDEA METODY
// -----------
// Strassen dzieli macierz na 2x2 bloki i mnozy je 7 razy zamiast 8.
// Laderman (1976) dzieli macierz n x n na 3x3 bloki o rozmiarze n/3:
//
//        | A11 A12 A13 |          | B11 B12 B13 |          | C11 C12 C13 |
//    A = | A21 A22 A23 |      B = | B21 B22 B23 |      C = | C21 C22 C23 |
//        | A31 A32 A33 |          | B31 B32 B33 |          | C31 C32 C33 |
//
// Klasycznie (jak u Binet'a) potrzeba 27 mnozen blokow. Laderman wystarcza 23:
//
//    M1  = (A11 + A12 + A13 - A21 - A22 - A32 - A33) * B22
//    M2  = (A11 - A21) * (B22 - B12)
//    M3  =  A22 * (B12 + B21 + B33 - B11 - B22 - B23 - B31)
//    M4  = (A21 + A22 - A11) * (B11 + B22 - B12)
//    M5  = (A21 + A22) * (B12 - B11)
//    M6  =  A11 * B11
//    M7  = (A31 + A32 - A11) * (B11 + B23 - B13)
//    M8  = (A31 - A11) * (B13 - B23)
//    M9  = (A31 + A32) * (B13 - B11)
//    M10 = (A11 + A12 + A13 - A22 - A23 - A31 - A32) * B23
//    M11 =  A32 * (B13 + B21 + B32 - B11 - B22 - B23 - B31)
//    M12 = (A32 + A33 - A13) * (B22 + B31 - B32)
//    M13 = (A13 - A33) * (B22 - B32)
//    M14 =  A13 * B31
//    M15 = (A32 + A33) * (B32 - B31)
//    M16 = (A22 + A23 - A13) * (B23 + B31 - B33)
//    M17 = (A13 - A23) * (B23 - B33)
//    M18 = (A22 + A23) * (B33 - B31)
//    M19 =  A12 * B21        M20 = A23 * B32        M21 = A21 * B13
//    M22 =  A31 * B12        M23 = A33 * B33
//
//    C11 = M6 + M14 + M19
//    C12 = M1 + M4 + M5 + M6 + M12 + M14 + M15
//    C13 = M6 + M7 + M9 + M10 + M14 + M16 + M18
//    C21 = M2 + M3 + M4 + M6 + M14 + M16 + M17
//    C22 = M2 + M4 + M5 + M6 + M20
//    C23 = M14 + M16 + M17 + M18 + M21
//    C31 = M6 + M7 + M8 + M11 + M12 + M13 + M14
//    C32 = M12 + M13 + M14 + M15 + M22
//    C33 = M6 + M7 + M8 + M9 + M23
//
// (W kazdej sumie pierwszy skladnik jest dodatni, wiec nie ma osobnej negacji.)
// Na jednym poziomie: 23 mnozenia blokow + 56 dodawan/odejmowan na wejsciu
// + 42 przy skladaniu wyniku = 98 operacji na blokach.
// Kazde z 23 mnozen wykonujemy REKURENCYJNIE ta sama metoda.
// Zlozonosc: T(n) = 23*T(n/3) + O(n^2)  =>  O(n^(log3 23)) ~ O(n^2.854).
//
// DOWOLNY ROZMIAR n - TECHNIKA "OBIERANIA" (dynamic peeling)
// ----------------------------------------------------------
// Podzial na 3x3 rowne bloki wymaga n podzielnego przez 3. Gdy r = n mod 3
// jest rozne od 0, "odcinamy" r ostatnich wierszy i r ostatnich kolumn
// (tak jak w strassen.cpp, tylko tam zawsze r = 1):
//
//        | A11  a12 |       A11 - macierz m x m, m = n - r  (podzielne przez 3!)
//    A = |          |       a12 - pasek m x r
//        | a21  a22 |       a21 - pasek r x m
//                           a22 - macierz r x r  (r = 1 lub 2)
//
// (analogicznie B). Wtedy:
//
//    C11 = A11*B11 + a12*b21      <- A11*B11 liczymy Ladermanem (rekurencja!),
//    C12 = A11*b12 + a12*b22         a pozostale, "cienkie" iloczyny zwyklym
//    C21 = a21*B11 + a22*b21         mnozeniem (kosztuja tylko O(n^2)).
//    C22 = a21*b12 + a22*b22
//
// Dzieki temu nie dopelniamy macierzy zerami, wiec licznik operacji jest uczciwy.
//
// PRZYPADEK BAZOWY: n <= LADERMAN_CUTOFF = 2. Macierzy 1x1 i 2x2 nie da sie
// podzielic na 3x3 niepuste bloki, wiec mnozymy je klasycznie (dla n = 1 to
// po prostu a*b, dla n = 2: 8 mnozen + 4 dodawania).
//
// LICZENIE OPERACJI
// -----------------
// Zliczamy operacje zmiennoprzecinkowe + - * /. Licznik g_flops zwiekszamy w:
//   * add()/combine()  - po jednej operacji na element za kazdy dodany
//                        lub odjety skladnik,
//   * multiplyNaive()  - mnozenia i dodawania klasycznego algorytmu
//                        (przypadek bazowy i "cienkie" iloczyny przy obieraniu).
// Kopiowanie blokow, generowanie macierzy i weryfikacja NIE sa liczone.
//
// KOMPILACJA I URUCHOMIENIE
// -------------------------
//   g++ -O2 -std=c++17 -o laderman.exe laderman.cpp
//   laderman.exe              -> test dla domyslnej listy rozmiarow
//   laderman.exe 5 16 100     -> test dla podanych rozmiarow n
//
// Tryby uzywane do sprawozdania (wynik w formacie CSV na stdout):
//   laderman.exe --flops nMin nMax              -> liczba operacji ze wzoru (bez mnozenia)
//   laderman.exe --time nMin nMax krok powt.    -> zmierzony czas i liczba operacji
//   laderman.exe --example                      -> przyklad 4x4 (magic(4) * magic(4)'/7)
// ============================================================================

#include <algorithm>         // std::max
#include <chrono>            // pomiar czasu
#include <cmath>             // std::fabs
#include <cstdlib>           // std::atoi
#include <functional>        // std::reference_wrapper
#include <initializer_list>  // listy skladnikow w combine()
#include <iomanip>           // formatowanie tabeli
#include <iostream>
#include <map>               // memoizacja w modelu liczby operacji
#include <random>            // generator liczb losowych
#include <string>
#include <vector>

// Dla macierzy o rozmiarze <= LADERMAN_CUTOFF przestajemy rekurencyjnie dzielic
// i mnozymy klasycznie. Musi byc >= 2 (2x2 nie da sie podzielic na 3x3 bloki).
static const int LADERMAN_CUTOFF = 2;

// Licznik operacji zmiennoprzecinkowych (+, -, *, /).
static unsigned long long g_flops = 0;

// ----------------------------------------------------------------------------
// Prosta klasa macierzy prostokatnej, elementy w jednym wektorze wierszami:
// element (i, j) lezy pod indeksem i*cols + j.
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Macierz n x n z liczbami losowymi z przedzialu [1e-8, 1.0), rozklad rownomierny.
// ----------------------------------------------------------------------------
Matrix randomMatrix(int n, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(0.00000001, 1.0);
    Matrix m(n, n);
    for (double& x : m.data) x = dist(gen);
    return m;
}

// Wycina blok o lewym gornym rogu (r0, c0) i rozmiarze h x w (tylko kopiowanie).
Matrix subMatrix(const Matrix& M, int r0, int c0, int h, int w) {
    Matrix R(h, w);
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) R(i, j) = M(r0 + i, c0 + j);
    return R;
}

// Wkleja blok S do macierzy M w pozycji (r0, c0) (odwrotnosc subMatrix).
void placeBlock(Matrix& M, const Matrix& S, int r0, int c0) {
    for (int i = 0; i < S.rows; ++i)
        for (int j = 0; j < S.cols; ++j) M(r0 + i, c0 + j) = S(i, j);
}

// Dodawanie macierzy o tych samych wymiarach: 1 operacja na element.
Matrix add(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, A.cols);
    for (size_t k = 0; k < C.data.size(); ++k) C.data[k] = A.data[k] + B.data[k];
    g_flops += C.data.size();
    return C;
}

// ----------------------------------------------------------------------------
// Kombinacja liniowa blokow: plus[0] + plus[1] + ... - minus[0] - minus[1] - ...
// We wzorach Ladermana sumy maja do 7 skladnikow, wiec zamiast zagniezdzac
// add(sub(add(...))) podajemy listy blokow dodawanych i odejmowanych.
// Liczba operacji: (liczba skladnikow - 1) na kazdy element wyniku - dokladnie
// tyle, ile przy liczeniu wzoru "recznie" od lewej do prawej.
// Lista plus musi byc niepusta (pierwszy skladnik jest zawsze dodatni).
// ----------------------------------------------------------------------------
using MatrixList = std::initializer_list<std::reference_wrapper<const Matrix>>;

Matrix combine(MatrixList plus, MatrixList minus = {}) {
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

// ----------------------------------------------------------------------------
// Klasyczne mnozenie macierzy prostokatnych (p x q) * (q x r).
// Uzywane w przypadku bazowym oraz do "cienkich" iloczynow przy obieraniu.
// Liczba operacji: p*r*q mnozen i p*r*(q-1) dodawan.
// ----------------------------------------------------------------------------
Matrix multiplyNaive(const Matrix& A, const Matrix& B) {
    const int p = A.rows, q = A.cols, r = B.cols;
    Matrix C(p, r);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < r; ++j) {
            double sum = A(i, 0) * B(0, j);
            for (int k = 1; k < q; ++k) sum += A(i, k) * B(k, j);
            C(i, j) = sum;
        }
    }
    g_flops += static_cast<unsigned long long>(p) * r * q;        // mnozenia
    g_flops += static_cast<unsigned long long>(p) * r * (q - 1);  // dodawania
    return C;
}

// ----------------------------------------------------------------------------
// SERCE ALGORYTMU: rekurencyjne mnozenie metoda Ladermana.
// A i B musza byc kwadratowe o tym samym rozmiarze n x n.
// ----------------------------------------------------------------------------
Matrix multiplyLaderman(const Matrix& A, const Matrix& B) {
    const int n = A.rows;

    // PRZYPADEK BAZOWY: macierz 1x1 lub 2x2 - mnozymy klasycznie.
    if (n <= LADERMAN_CUTOFF) return multiplyNaive(A, B);

    // ------------------------------------------------------------------
    // n NIEPODZIELNE PRZEZ 3: obieranie r ostatnich wierszy i kolumn.
    // ------------------------------------------------------------------
    const int r = n % 3;
    if (r != 0) {
        const int m = n - r;  // rozmiar "rdzenia" - podzielny przez 3 (i >= 3)

        // Rozbicie A na: rdzen A11 (m x m), pasek a12 (m x r),
        // pasek a21 (r x m) i mala macierz a22 (r x r).
        Matrix A11 = subMatrix(A, 0, 0, m, m);
        Matrix a12 = subMatrix(A, 0, m, m, r);
        Matrix a21 = subMatrix(A, m, 0, r, m);
        Matrix a22 = subMatrix(A, m, m, r, r);
        // To samo dla B.
        Matrix B11 = subMatrix(B, 0, 0, m, m);
        Matrix b12 = subMatrix(B, 0, m, m, r);
        Matrix b21 = subMatrix(B, m, 0, r, m);
        Matrix b22 = subMatrix(B, m, m, r, r);

        // Rdzen mnozymy rekurencyjnie Ladermanem; reszta to tanie, cienkie iloczyny.
        //   a12*b21 : (m x r)*(r x m) = macierz m x m
        //   A11*b12 : (m x m)*(m x r) = pasek m x r
        //   a12*b22 : (m x r)*(r x r) = pasek m x r
        //   a21*B11 : (r x m)*(m x m) = pasek r x m
        //   a22*b21 : (r x r)*(r x m) = pasek r x m
        //   a21*b12 : (r x m)*(m x r) = macierz r x r
        //   a22*b22 : (r x r)*(r x r) = macierz r x r
        Matrix C11 = add(multiplyLaderman(A11, B11), multiplyNaive(a12, b21));
        Matrix C12 = add(multiplyNaive(A11, b12), multiplyNaive(a12, b22));
        Matrix C21 = add(multiplyNaive(a21, B11), multiplyNaive(a22, b21));
        Matrix C22 = add(multiplyNaive(a21, b12), multiplyNaive(a22, b22));

        // Skladamy wynik n x n z czterech czesci.
        Matrix C(n, n);
        placeBlock(C, C11, 0, 0);
        placeBlock(C, C12, 0, m);
        placeBlock(C, C21, m, 0);
        placeBlock(C, C22, m, m);
        return C;
    }

    // ------------------------------------------------------------------
    // n PODZIELNE PRZEZ 3: wlasciwy algorytm Ladermana.
    // ------------------------------------------------------------------
    const int h = n / 3;  // rozmiar bloku

    // Podzial A i B na 9 blokow h x h.
    Matrix A11 = subMatrix(A, 0, 0, h, h), A12 = subMatrix(A, 0, h, h, h), A13 = subMatrix(A, 0, 2 * h, h, h);
    Matrix A21 = subMatrix(A, h, 0, h, h), A22 = subMatrix(A, h, h, h, h), A23 = subMatrix(A, h, 2 * h, h, h);
    Matrix A31 = subMatrix(A, 2 * h, 0, h, h), A32 = subMatrix(A, 2 * h, h, h, h), A33 = subMatrix(A, 2 * h, 2 * h, h, h);

    Matrix B11 = subMatrix(B, 0, 0, h, h), B12 = subMatrix(B, 0, h, h, h), B13 = subMatrix(B, 0, 2 * h, h, h);
    Matrix B21 = subMatrix(B, h, 0, h, h), B22 = subMatrix(B, h, h, h, h), B23 = subMatrix(B, h, 2 * h, h, h);
    Matrix B31 = subMatrix(B, 2 * h, 0, h, h), B32 = subMatrix(B, 2 * h, h, h, h), B33 = subMatrix(B, 2 * h, 2 * h, h, h);

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

    // Odzyskanie blokow wyniku z M1..M23 (wzory Ladermana - same dodawania).
    Matrix C11 = combine({M6, M14, M19});
    Matrix C12 = combine({M1, M4, M5, M6, M12, M14, M15});
    Matrix C13 = combine({M6, M7, M9, M10, M14, M16, M18});
    Matrix C21 = combine({M2, M3, M4, M6, M14, M16, M17});
    Matrix C22 = combine({M2, M4, M5, M6, M20});
    Matrix C23 = combine({M14, M16, M17, M18, M21});
    Matrix C31 = combine({M6, M7, M8, M11, M12, M13, M14});
    Matrix C32 = combine({M12, M13, M14, M15, M22});
    Matrix C33 = combine({M6, M7, M8, M9, M23});

    // Skladamy wynik n x n z dziewieciu blokow.
    Matrix C(n, n);
    placeBlock(C, C11, 0, 0);     placeBlock(C, C12, 0, h);     placeBlock(C, C13, 0, 2 * h);
    placeBlock(C, C21, h, 0);     placeBlock(C, C22, h, h);     placeBlock(C, C23, h, 2 * h);
    placeBlock(C, C31, 2 * h, 0); placeBlock(C, C32, 2 * h, h); placeBlock(C, C33, 2 * h, 2 * h);
    return C;
}

// ----------------------------------------------------------------------------
// Referencyjne mnozenie klasyczne - NIE zwieksza licznika g_flops.
// Sluzy tylko do sprawdzenia poprawnosci wyniku Ladermana.
// ----------------------------------------------------------------------------
Matrix multiplyReference(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k)
            for (int j = 0; j < B.cols; ++j) C(i, j) += A(i, k) * B(k, j);
    return C;
}

// Najwieksza roznica bezwzgledna miedzy odpowiadajacymi sobie elementami
// (nie bedzie dokladnie 0 przez bledy zaokraglen).
double maxDifference(const Matrix& A, const Matrix& B) {
    double d = 0.0;
    for (size_t k = 0; k < A.data.size(); ++k) d = std::max(d, std::fabs(A.data[k] - B.data[k]));
    return d;
}

// Liczba operacji klasycznego mnozenia (p x q) * (q x r): p*r*(2q - 1).
unsigned long long naiveCost(unsigned long long p, unsigned long long q, unsigned long long r) {
    return p * r * (2 * q - 1);
}

// ----------------------------------------------------------------------------
// MODEL LICZBY OPERACJI (dla raportu): ta sama rekurencja co multiplyLaderman,
// ale tylko sumuje koszty zamiast mnozyc macierze. Pozwala policzyc liczbe
// operacji dla duzych n bez wykonywania obliczen (wyniki identyczne z g_flops).
//   - przypadek bazowy: n*n*(2n-1) operacji (klasyczne mnozenie),
//   - n podzielne przez 3: 23*L(n/3) + 98*(n/3)^2  (56 dodawan wejsc + 42 skladania),
//   - w przeciwnym razie (r = n mod 3, m = n-r): L(m) + koszt "cienkich" iloczynow
//     i 4 dodawan blokow (lacznie n*n elementow).
// Dla n = 3^k wychodzi dokladnie L(n) = 8*23^k - 7*9^k.
// ----------------------------------------------------------------------------
unsigned long long flopsModel(int n, std::map<int, unsigned long long>& memo) {
    if (n <= LADERMAN_CUTOFF) return naiveCost(n, n, n);
    auto it = memo.find(n);
    if (it != memo.end()) return it->second;

    unsigned long long total;
    const unsigned long long r = n % 3;
    if (r != 0) {
        const unsigned long long m = n - r;
        total = flopsModel(static_cast<int>(m), memo)                 // rdzen A11*B11 (Laderman)
              + naiveCost(m, r, m) + m * m                             // C11: a12*b21, dodawanie
              + naiveCost(m, m, r) + naiveCost(m, r, r) + m * r        // C12
              + naiveCost(r, m, m) + naiveCost(r, r, m) + r * m        // C21
              + naiveCost(r, m, r) + naiveCost(r, r, r) + r * r;       // C22
    } else {
        const unsigned long long h = n / 3;
        total = 23 * flopsModel(n / 3, memo) + 98 * h * h;
    }
    memo[n] = total;
    return total;
}

int main(int argc, char** argv) {
    // TRYB RAPORTOWY 1:  laderman.exe --flops nMin nMax
    // Wypisuje CSV "n,flops" ze wzoru rekurencyjnego (bez mnozenia macierzy).
    if (argc == 4 && std::string(argv[1]) == "--flops") {
        std::map<int, unsigned long long> memo;
        std::cout << "n,flops\n";
        for (int n = std::atoi(argv[2]); n <= std::atoi(argv[3]); ++n)
            std::cout << n << "," << flopsModel(n, memo) << "\n";
        return 0;
    }

    // TRYB RAPORTOWY 2:  laderman.exe --time nMin nMax krok powtorzenia
    // Dla kazdego n wykonuje prawdziwe mnozenie, mierzy czas (minimum z kilku
    // powtorzen) i wypisuje CSV "n,flops_zmierzone,czas_ms".
    if (argc == 6 && std::string(argv[1]) == "--time") {
        const int nMin = std::atoi(argv[2]), nMax = std::atoi(argv[3]);
        const int step = std::atoi(argv[4]), reps = std::atoi(argv[5]);
        std::mt19937 gen(12345);
        std::cout << "n,flops,time_ms\n";
        for (int n = nMin; n <= nMax; n += step) {
            Matrix A = randomMatrix(n, gen);
            Matrix B = randomMatrix(n, gen);
            double best = 1e300;
            unsigned long long flops = 0;
            for (int k = 0; k < reps; ++k) {
                g_flops = 0;
                auto t0 = std::chrono::steady_clock::now();
                Matrix C = multiplyLaderman(A, B);
                auto t1 = std::chrono::steady_clock::now();
                flops = g_flops;
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::cout << n << "," << flops << "," << std::fixed << std::setprecision(4) << best << std::endl;
        }
        return 0;
    }

    // TRYB RAPORTOWY 3:  laderman.exe --example
    // Mnozy stala macierz A = magic(4) przez B = A'/7 i wypisuje wynik C w
    // formacie czytelnym dla Octave/MATLAB (load) - patrz skrypt compare.m.
    // Linie zaczynajace sie od '%' sa komentarzami.
    if (argc == 2 && std::string(argv[1]) == "--example") {
        const double magic4[4][4] = {{16, 2, 3, 13}, {5, 11, 10, 8}, {9, 7, 6, 12}, {4, 14, 15, 1}};
        Matrix A(4, 4), B(4, 4);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                A(i, j) = magic4[i][j];
                B(i, j) = magic4[j][i] / 7.0;  // transpozycja podzielona przez 7
            }
        g_flops = 0;
        Matrix C = multiplyLaderman(A, B);
        std::cout << "% Laderman 4x4, liczba operacji = " << g_flops << "\n";
        std::cout << std::setprecision(17);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) std::cout << C(i, j) << (j < 3 ? " " : "\n");
        }
        return 0;
    }

    // Rozmiary: z argumentow wiersza polecen albo domyslne (wszystkie reszty
    // z dzielenia przez 3, zeby sprawdzic obieranie 1 i 2 wierszy/kolumn).
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 27, 28, 29, 50, 81, 100};

    std::mt19937 gen(12345);  // stale ziarno -> powtarzalne wyniki

    std::cout << "Mnozenie macierzy metoda Ladermana (liczby z [1e-8, 1.0)), "
              << "cutoff = " << LADERMAN_CUTOFF << "\n\n";
    std::cout << std::setw(6) << "n" << std::setw(18) << "operacje (+-*/)"
              << std::setw(20) << "klasyczne 2n^3-n^2" << std::setw(14) << "czas [ms]"
              << std::setw(14) << "max |blad|" << "\n";

    for (int n : sizes) {
        if (n < 1) continue;
        Matrix A = randomMatrix(n, gen);
        Matrix B = randomMatrix(n, gen);

        g_flops = 0;  // zerujemy licznik tuz przed wlasciwym mnozeniem
        auto t0 = std::chrono::steady_clock::now();
        Matrix C = multiplyLaderman(A, B);
        auto t1 = std::chrono::steady_clock::now();
        const unsigned long long flops = g_flops;

        // Weryfikacja poprawnosci (nie liczona do flops).
        Matrix R = multiplyReference(A, B);
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        unsigned long long classic = 2ULL * n * n * n - 1ULL * n * n;

        std::cout << std::setw(6) << n << std::setw(18) << flops << std::setw(20) << classic
                  << std::setw(14) << std::fixed << std::setprecision(3) << ms
                  << std::setw(14) << std::scientific << std::setprecision(2) << maxDifference(C, R)
                  << "\n";
    }
    return 0;
}
