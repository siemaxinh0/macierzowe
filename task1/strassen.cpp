// ZADANIE 2: Rekurencyjne mnozenie macierzy metoda Strassena
//
// IDEA METODY
// Tak jak w metodzie Binet'a dzielimy macierze n x n na 4 bloki:
//
//        | A11 A12 |          | B11 B12 |          | C11 C12 |
//    A = |         |      B = |         |      C = |         |
//        | A21 A22 |          | B21 B22 |          | C21 C22 |
//
// Binet potrzebuje 8 mnozen blokow. Strassen zauwazyl, ze wystarczy 7, kosztem
// wiekszej liczby dodawan/odejmowan (ktore sa tansze - O(n^2)):
//
//    M1 = (A11 + A22) * (B11 + B22)
//    M2 = (A21 + A22) *  B11
//    M3 =  A11        * (B12 - B22)
//    M4 =  A22        * (B21 - B11)
//    M5 = (A11 + A12) *  B22
//    M6 = (A21 - A11) * (B11 + B12)
//    M7 = (A12 - A22) * (B21 + B22)
//
//    C11 = M1 + M4 - M5 + M7
//    C12 = M3 + M5
//    C21 = M2 + M4
//    C22 = M1 - M2 + M3 + M6
//
// Kazde z 7 mnozen M1..M7 wykonujemy REKURENCYJNIE ta sama metoda.
// Zlozonosc: T(n) = 7*T(n/2) + O(n^2)  =>  O(n^(log2 7)) ~ O(n^2.807).
//
// DOWOLNY ROZMIAR n - TECHNIKA "OBIERANIA" (dynamic peeling)
// Podzial na 4 rowne bloki wymaga parzystego n. Gdy n jest NIEPARZYSTE,
// "odcinamy" ostatni wiersz i ostatnia kolumne:
//
//        | A11  a12 |       A11 - macierz (n-1)x(n-1)   (parzysty rozmiar!)
//    A = |          |       a12 - kolumna  (n-1)x1
//        | a21  a22 |       a21 - wiersz   1x(n-1)
//                           a22 - pojedyncza liczba
//
// (analogicznie B). Wtedy:
//
//    C11 = A11*B11 + a12*b21      <- A11*B11 liczymy Strassenem (rekurencja!),
//    C12 = A11*b12 + a12*b22         a pozostale, "cienkie" iloczyny zwyklym
//    C21 = a21*B11 + a22*b21         mnozeniem (kosztuja tylko O(n^2)).
//    C22 = a21*b12 + a22*b22
//
// Dzieki temu nie dopelniamy macierzy zerami, wiec licznik operacji jest uczciwy.
//
// PRZYPADEK BAZOWY: n == 1 (mnozenie dwoch liczb). Stala STRASSEN_CUTOFF
// pozwala zakonczyc rekurencje wczesniej (przy 1 jest "czysty" Strassen, tak
// jak w poleceniu). W praktyce zwykle ustawia sie ~32-64, bo dla malych
// macierzy klasyczne mnozenie jest szybsze.
//
// LICZENIE OPERACJI
// Zliczamy operacje zmiennoprzecinkowe + - * /. Licznik g_flops zwiekszamy w:
//   * add()/sub()      - po jednej operacji na element wyniku,
//   * multiplyNaive()  - mnozenia i dodawania klasycznego algorytmu
//                        (przypadek bazowy i "cienkie" iloczyny przy obieraniu).
// Kopiowanie blokow, generowanie macierzy i weryfikacja NIE sa liczone.
//
// KOMPILACJA I URUCHOMIENIE
//   g++ -O2 -std=c++17 -o strassen.exe strassen.cpp
//   strassen.exe              -> test dla domyslnej listy rozmiarow
//   strassen.exe 5 16 100     -> test dla podanych rozmiarow n
//
// Tryby uzywane do sprawozdania (wynik w formacie CSV na stdout):
//   strassen.exe --flops nMin nMax              -> liczba operacji ze wzoru (bez mnozenia)
//   strassen.exe --time nMin nMax krok powt.    -> zmierzony czas i liczba operacji
//   strassen.exe --example                      -> przyklad 4x4 (magic(4) * magic(4)'/7)

#include <algorithm>   // std::max
#include <chrono>      // pomiar czasu
#include <cmath>       // std::fabs
#include <cstdlib>     // std::atoi
#include <iomanip>     // formatowanie tabeli
#include <iostream>
#include <map>         // memoizacja w modelu liczby operacji
#include <random>      // generator liczb losowych
#include <string>
#include <vector>

// Dla macierzy o rozmiarze <= STRASSEN_CUTOFF przestajemy rekurencyjnie dzielic
// i mnozymy klasycznie. Wartosc 1 = rekurencja az do pojedynczych liczb.
static const int STRASSEN_CUTOFF = 1;

// Licznik operacji zmiennoprzecinkowych (+, -, *, /).
static unsigned long long g_flops = 0;

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

// Macierz n x n z liczbami losowymi z przedzialu [1e-8, 1.0), rozklad rownomierny.
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

// Odejmowanie macierzy o tych samych wymiarach: 1 operacja na element.
Matrix sub(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, A.cols);
    for (size_t k = 0; k < C.data.size(); ++k) C.data[k] = A.data[k] - B.data[k];
    g_flops += C.data.size();
    return C;
}

// Klasyczne mnozenie macierzy prostokatnych (p x q) * (q x r).
// Uzywane w przypadku bazowym oraz do "cienkich" iloczynow przy obieraniu.
// Liczba operacji: p*r*q mnozen i p*r*(q-1) dodawan.
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

// SERCE ALGORYTMU: rekurencyjne mnozenie metoda Strassena.
// A i B musza byc kwadratowe o tym samym rozmiarze n x n.
Matrix multiplyStrassen(const Matrix& A, const Matrix& B) {
    const int n = A.rows;

    // PRZYPADEK BAZOWY: macierz mala (domyslnie 1x1) - mnozymy klasycznie.
    if (n <= STRASSEN_CUTOFF) return multiplyNaive(A, B);

    // n NIEPARZYSTE: obieranie ostatniego wiersza i kolumny (patrz opis na gorze).
    if (n % 2 == 1) {
        const int m = n - 1;  // rozmiar "rdzenia" - parzysty

        // Rozbicie A na: rdzen A11 (m x m), kolumne a12 (m x 1),
        // wiersz a21 (1 x m) i liczbe a22 (1 x 1).
        Matrix A11 = subMatrix(A, 0, 0, m, m);
        Matrix a12 = subMatrix(A, 0, m, m, 1);
        Matrix a21 = subMatrix(A, m, 0, 1, m);
        Matrix a22 = subMatrix(A, m, m, 1, 1);
        // To samo dla B.
        Matrix B11 = subMatrix(B, 0, 0, m, m);
        Matrix b12 = subMatrix(B, 0, m, m, 1);
        Matrix b21 = subMatrix(B, m, 0, 1, m);
        Matrix b22 = subMatrix(B, m, m, 1, 1);

        // Rdzen mnozymy rekurencyjnie Strassenem; reszta to tanie, cienkie iloczyny.
        //   a12*b21 : (m x 1)*(1 x m) = macierz m x m (iloczyn zewnetrzny)
        //   A11*b12 : (m x m)*(m x 1) = kolumna m x 1
        //   a12*b22 : (m x 1)*(1 x 1) = kolumna m x 1
        //   a21*B11 : (1 x m)*(m x m) = wiersz 1 x m
        //   a22*b21 : (1 x 1)*(1 x m) = wiersz 1 x m
        //   a21*b12 : (1 x m)*(m x 1) = liczba 1 x 1
        //   a22*b22 : (1 x 1)*(1 x 1) = liczba 1 x 1
        Matrix C11 = add(multiplyStrassen(A11, B11), multiplyNaive(a12, b21));
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

    // n PARZYSTE: wlasciwy algorytm Strassena.
    const int h = n / 2;  // rozmiar bloku

    // Podzial A i B na cwiartki h x h.
    Matrix A11 = subMatrix(A, 0, 0, h, h);
    Matrix A12 = subMatrix(A, 0, h, h, h);
    Matrix A21 = subMatrix(A, h, 0, h, h);
    Matrix A22 = subMatrix(A, h, h, h, h);

    Matrix B11 = subMatrix(B, 0, 0, h, h);
    Matrix B12 = subMatrix(B, 0, h, h, h);
    Matrix B21 = subMatrix(B, h, 0, h, h);
    Matrix B22 = subMatrix(B, h, h, h, h);

    // Siedem rekurencyjnych mnozen (zamiast osmiu u Binet'a).
    // Przed kazdym mnozeniem wykonujemy ewentualne dodawanie/odejmowanie blokow.
    Matrix M1 = multiplyStrassen(add(A11, A22), add(B11, B22));
    Matrix M2 = multiplyStrassen(add(A21, A22), B11);
    Matrix M3 = multiplyStrassen(A11, sub(B12, B22));
    Matrix M4 = multiplyStrassen(A22, sub(B21, B11));
    Matrix M5 = multiplyStrassen(add(A11, A12), B22);
    Matrix M6 = multiplyStrassen(sub(A21, A11), add(B11, B12));
    Matrix M7 = multiplyStrassen(sub(A12, A22), add(B21, B22));

    // Odzyskanie blokow wyniku z M1..M7 (wzory Strassena).
    Matrix C11 = add(sub(add(M1, M4), M5), M7);   // M1 + M4 - M5 + M7
    Matrix C12 = add(M3, M5);                     // M3 + M5
    Matrix C21 = add(M2, M4);                     // M2 + M4
    Matrix C22 = add(add(sub(M1, M2), M3), M6);   // M1 - M2 + M3 + M6

    // Skladamy wynik n x n z czterech blokow.
    Matrix C(n, n);
    placeBlock(C, C11, 0, 0);
    placeBlock(C, C12, 0, h);
    placeBlock(C, C21, h, 0);
    placeBlock(C, C22, h, h);
    return C;
}

// Referencyjne mnozenie klasyczne - NIE zwieksza licznika g_flops.
// Sluzy tylko do sprawdzenia poprawnosci wyniku Strassena.
Matrix multiplyReference(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k)
            for (int j = 0; j < B.cols; ++j) C(i, j) += A(i, k) * B(k, j);
    return C;
}

// Najwieksza roznica bezwzgledna miedzy odpowiadajacymi sobie elementami
// (nie bedzie dokladnie 0 przez bledy zaokraglen - Strassen jest nieco mniej
// stabilny numerycznie niz metoda klasyczna).
double maxDifference(const Matrix& A, const Matrix& B) {
    double d = 0.0;
    for (size_t k = 0; k < A.data.size(); ++k) d = std::max(d, std::fabs(A.data[k] - B.data[k]));
    return d;
}

// MODEL LICZBY OPERACJI (dla raportu): ta sama rekurencja co multiplyStrassen,
// ale tylko sumuje koszty zamiast mnozyc macierze. Pozwala policzyc liczbe
// operacji dla duzych n bez wykonywania obliczen (wyniki identyczne z g_flops).
//   - przypadek bazowy: n*n*(2n-1) operacji (klasyczne mnozenie),
//   - n parzyste: 7*S(n/2) + 18*(n/2)^2   (10 dodawan wejsc + 8 skladania wyniku),
//   - n nieparzyste (m = n-1): S(m) + koszt "cienkich" iloczynow i dodawan.
unsigned long long flopsModel(int n, std::map<int, unsigned long long>& memo) {
    if (n <= STRASSEN_CUTOFF) return 1ULL * n * n * (2ULL * n - 1);
    auto it = memo.find(n);
    if (it != memo.end()) return it->second;

    unsigned long long total;
    if (n % 2 == 1) {
        const unsigned long long m = n - 1;
        total = flopsModel(n - 1, memo)  // rdzen A11*B11 (Strassen)
              + m * m + m * m            // C11: iloczyn a12*b21 oraz dodawanie
              + m * (2 * m - 1) + m + m  // C12: A11*b12, a12*b22, dodawanie
              + m * (2 * m - 1) + m + m  // C21: a21*B11, a22*b21, dodawanie
              + (2 * m - 1) + 1 + 1;     // C22: a21*b12, a22*b22, dodawanie
    } else {
        const unsigned long long h = n / 2;
        total = 7 * flopsModel(n / 2, memo) + 18 * h * h;
    }
    memo[n] = total;
    return total;
}

int main(int argc, char** argv) {
    // TRYB RAPORTOWY 1:  strassen.exe --flops nMin nMax
    // Wypisuje CSV "n,flops" ze wzoru rekurencyjnego (bez mnozenia macierzy).
    if (argc == 4 && std::string(argv[1]) == "--flops") {
        std::map<int, unsigned long long> memo;
        std::cout << "n,flops\n";
        for (int n = std::atoi(argv[2]); n <= std::atoi(argv[3]); ++n)
            std::cout << n << "," << flopsModel(n, memo) << "\n";
        return 0;
    }

    // TRYB RAPORTOWY 2:  strassen.exe --time nMin nMax krok powtorzenia
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
                Matrix C = multiplyStrassen(A, B);
                auto t1 = std::chrono::steady_clock::now();
                flops = g_flops;
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::cout << n << "," << flops << "," << std::fixed << std::setprecision(4) << best << std::endl;
        }
        return 0;
    }

    // TRYB RAPORTOWY 3:  strassen.exe --example
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
        Matrix C = multiplyStrassen(A, B);
        std::cout << "% Strassen 4x4, liczba operacji = " << g_flops << "\n";
        std::cout << std::setprecision(17);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) std::cout << C(i, j) << (j < 3 ? " " : "\n");
        }
        return 0;
    }

    // Rozmiary: z argumentow wiersza polecen albo domyslne (z nieparzystymi).
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {1, 2, 3, 4, 5, 7, 8, 10, 16, 17, 32, 50, 64, 100};

    std::mt19937 gen(12345);  // stale ziarno -> powtarzalne wyniki

    std::cout << "Mnozenie macierzy metoda Strassena (liczby z [1e-8, 1.0)), "
              << "cutoff = " << STRASSEN_CUTOFF << "\n\n";
    std::cout << std::setw(6) << "n" << std::setw(18) << "operacje (+-*/)"
              << std::setw(20) << "klasyczne 2n^3-n^2" << std::setw(14) << "czas [ms]"
              << std::setw(14) << "max |blad|" << "\n";

    for (int n : sizes) {
        if (n < 1) continue;
        Matrix A = randomMatrix(n, gen);
        Matrix B = randomMatrix(n, gen);

        g_flops = 0;  // zerujemy licznik tuz przed wlasciwym mnozeniem
        auto t0 = std::chrono::steady_clock::now();
        Matrix C = multiplyStrassen(A, B);
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
