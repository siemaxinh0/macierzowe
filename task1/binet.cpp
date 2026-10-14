// ============================================================================
// ZADANIE 1: Rekurencyjne mnozenie macierzy metoda Binet'a
// ============================================================================
//
// IDEA METODY
// -----------
// Mnozymy dwie macierze kwadratowe A i B o rozmiarze n x n. Dzielimy kazda z nich
// na 4 bloki (cwiartki):
//
//        | A11 A12 |          | B11 B12 |          | C11 C12 |
//    A = |         |      B = |         |      C = |         |
//        | A21 A22 |          | B21 B22 |          | C21 C22 |
//
// Wtedy (tak samo jak przy mnozeniu macierzy 2x2 liczb):
//
//    C11 = A11*B11 + A12*B21
//    C12 = A11*B12 + A12*B22
//    C21 = A21*B11 + A22*B21
//    C22 = A21*B12 + A22*B22
//
// Daje to 8 mnozen mniejszych macierzy (kazde wykonujemy REKURENCYJNIE) oraz
// 4 dodawania macierzy. Rekurencja konczy sie, gdy macierze sa juz "najmniejsze"
// (1x1 -> zwykle mnozenie dwoch liczb).
//
// Zlozonosc: T(n) = 8*T(n/2) + O(n^2)  =>  O(n^3)  (nic nie zyskujemy wzgledem
// klasycznego algorytmu, ale to dobry punkt odniesienia dla metody Strassena).
//
// DOWOLNY ROZMIAR n (rowniez nieparzysty, nie tylko potegi dwojki)
// ----------------------------------------------------------------
// Gdy n jest nieparzyste, dzielimy "prawie" na pol: gorne/lewe bloki maja
// ceil(n/2) wierszy/kolumn, dolne/prawe floor(n/2). Bloki nie sa wiec rownej
// wielkosci, dlatego funkcja mnozenia przyjmuje macierze PROSTOKATNE
// (p x q) * (q x r). Wzory na C11..C22 pozostaja poprawne dla blokow
// o roznych rozmiarach - wystarczy, ze wymiary "pasuja" do siebie.
// Dzieki temu NIE dopelniamy macierzy zerami, wiec licznik operacji jest uczciwy.
//
// LICZENIE OPERACJI
// -----------------
// Zliczamy operacje zmiennoprzecinkowe: + - * /. Licznik g_flops zwiekszamy:
//   * w add()/sub()      - po jednej operacji na kazdy element wyniku,
//   * w multiplyNaive()  - mnozenia i dodawania w klasycznym algorytmie
//                          (uzywanym tylko w przypadku bazowym rekurencji).
// Funkcje pomocnicze (kopiowanie blokow, generowanie macierzy, weryfikacja
// wyniku) NIE sa liczone.
//
// KOMPILACJA I URUCHOMIENIE
// -------------------------
//   g++ -O2 -std=c++17 -o binet.exe binet.cpp
//   binet.exe              -> test dla domyslnej listy rozmiarow
//   binet.exe 5 16 100     -> test dla podanych rozmiarow n
//
// Tryby uzywane do sprawozdania (wynik w formacie CSV na stdout):
//   binet.exe --flops nMin nMax              -> liczba operacji ze wzoru (bez mnozenia)
//   binet.exe --time nMin nMax krok powt.    -> zmierzony czas i liczba operacji
//   binet.exe --example                      -> przyklad 4x4 (magic(4) * magic(4)'/7)
// ============================================================================

#include <algorithm>   // std::max
#include <chrono>      // pomiar czasu
#include <cmath>       // std::fabs
#include <cstdlib>     // std::atoi
#include <iomanip>     // formatowanie tabeli
#include <iostream>
#include <map>         // memoizacja w modelu liczby operacji
#include <random>      // generator liczb losowych
#include <string>
#include <tuple>
#include <vector>

// Licznik operacji zmiennoprzecinkowych (+, -, *, /). Globalny, zeby kazda
// funkcja mogla go zwiekszac bez przekazywania go jako argumentu.
static unsigned long long g_flops = 0;

// ----------------------------------------------------------------------------
// Prosta klasa macierzy prostokatnej. Elementy trzymamy w jednym wektorze
// "wierszami" (row-major): element (i, j) lezy pod indeksem i*cols + j.
// ----------------------------------------------------------------------------
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> data;

    Matrix() = default;
    // Konstruktor tworzy macierz r x c wypelniona zerami.
    Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0) {}

    // Dostep do elementu (i, j) - wersja do odczytu i do zapisu.
    double& operator()(int i, int j) { return data[static_cast<size_t>(i) * cols + j]; }
    double operator()(int i, int j) const { return data[static_cast<size_t>(i) * cols + j]; }
};

// ----------------------------------------------------------------------------
// Macierz n x n z liczbami losowymi z przedzialu [1e-8, 1.0).
// Rozklad rownomierny; generator Mersenne Twister przekazujemy przez referencje,
// zeby kolejne macierze dostawaly kolejne (rozne) liczby.
// ----------------------------------------------------------------------------
Matrix randomMatrix(int n, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(0.00000001, 1.0);
    Matrix m(n, n);
    for (double& x : m.data) x = dist(gen);
    return m;
}

// ----------------------------------------------------------------------------
// Wycina z macierzy M blok o lewym gornym rogu (r0, c0) i rozmiarze h x w.
// To zwykle kopiowanie - nie wykonuje operacji arytmetycznych, wiec nie liczymy.
// ----------------------------------------------------------------------------
Matrix subMatrix(const Matrix& M, int r0, int c0, int h, int w) {
    Matrix R(h, w);
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) R(i, j) = M(r0 + i, c0 + j);
    return R;
}

// ----------------------------------------------------------------------------
// Wkleja blok S do macierzy M w pozycji (r0, c0). Operacja odwrotna do subMatrix.
// Uzywamy jej do zlozenia macierzy wynikowej C z cwiartek C11, C12, C21, C22.
// ----------------------------------------------------------------------------
void placeBlock(Matrix& M, const Matrix& S, int r0, int c0) {
    for (int i = 0; i < S.rows; ++i)
        for (int j = 0; j < S.cols; ++j) M(r0 + i, c0 + j) = S(i, j);
}

// ----------------------------------------------------------------------------
// Dodawanie macierzy o tych samych wymiarach. Kazdy element wyniku to 1 dodawanie,
// wiec do licznika dodajemy rows*cols.
// ----------------------------------------------------------------------------
Matrix add(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, A.cols);
    for (size_t k = 0; k < C.data.size(); ++k) C.data[k] = A.data[k] + B.data[k];
    g_flops += C.data.size();
    return C;
}

// ----------------------------------------------------------------------------
// Klasyczne (trzy petle) mnozenie macierzy prostokatnych (p x q) * (q x r).
// W metodzie Binet'a uzywamy go TYLKO jako przypadku bazowego rekurencji
// (gdy ktorys z wymiarow wynosi 1).
//
// Liczba operacji: dla kazdego z p*r elementow wyniku mamy q mnozen
// i (q - 1) dodawan.
// ----------------------------------------------------------------------------
Matrix multiplyNaive(const Matrix& A, const Matrix& B) {
    const int p = A.rows, q = A.cols, r = B.cols;
    Matrix C(p, r);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < r; ++j) {
            double sum = A(i, 0) * B(0, j);          // pierwszy iloczyn (bez dodawania)
            for (int k = 1; k < q; ++k) sum += A(i, k) * B(k, j);  // reszta: * i +
            C(i, j) = sum;
        }
    }
    g_flops += static_cast<unsigned long long>(p) * r * q;        // mnozenia
    g_flops += static_cast<unsigned long long>(p) * r * (q - 1);  // dodawania
    return C;
}

// ----------------------------------------------------------------------------
// SERCE ALGORYTMU: rekurencyjne mnozenie metoda Binet'a.
// A ma wymiar p x q, B ma wymiar q x r, wynik ma wymiar p x r.
// ----------------------------------------------------------------------------
Matrix multiplyBinet(const Matrix& A, const Matrix& B) {
    const int p = A.rows;   // liczba wierszy A (i wyniku)
    const int q = A.cols;   // liczba kolumn A = liczba wierszy B
    const int r = B.cols;   // liczba kolumn B (i wyniku)

    // PRZYPADEK BAZOWY: jesli ktorys wymiar to 1, nie da sie juz podzielic
    // na dwie niepuste polowy. Dla macierzy 1x1 jest to po prostu a*b.
    if (p == 1 || q == 1 || r == 1) return multiplyNaive(A, B);

    // Punkty podzialu: gorna/lewa czesc dostaje ceil(x/2) elementow
    // (dla nieparzystego x jest o 1 wieksza), dolna/prawa - reszte.
    const int ph = (p + 1) / 2;
    const int qh = (q + 1) / 2;
    const int rh = (r + 1) / 2;

    // Podzial A na cwiartki: wiersze dzielimy w punkcie ph, kolumny w punkcie qh.
    Matrix A11 = subMatrix(A, 0,  0,  ph,     qh);
    Matrix A12 = subMatrix(A, 0,  qh, ph,     q - qh);
    Matrix A21 = subMatrix(A, ph, 0,  p - ph, qh);
    Matrix A22 = subMatrix(A, ph, qh, p - ph, q - qh);

    // Podzial B: wiersze dzielimy w punkcie qh (musi pasowac do kolumn A!),
    // kolumny w punkcie rh.
    Matrix B11 = subMatrix(B, 0,  0,  qh,     rh);
    Matrix B12 = subMatrix(B, 0,  rh, qh,     r - rh);
    Matrix B21 = subMatrix(B, qh, 0,  q - qh, rh);
    Matrix B22 = subMatrix(B, qh, rh, q - qh, r - rh);

    // 8 rekurencyjnych mnozen + 4 dodawania - dokladnie wg wzorow z gory pliku.
    Matrix C11 = add(multiplyBinet(A11, B11), multiplyBinet(A12, B21));
    Matrix C12 = add(multiplyBinet(A11, B12), multiplyBinet(A12, B22));
    Matrix C21 = add(multiplyBinet(A21, B11), multiplyBinet(A22, B21));
    Matrix C22 = add(multiplyBinet(A21, B12), multiplyBinet(A22, B22));

    // Skladamy wynik z czterech blokow w jedna macierz p x r.
    Matrix C(p, r);
    placeBlock(C, C11, 0,  0);
    placeBlock(C, C12, 0,  rh);
    placeBlock(C, C21, ph, 0);
    placeBlock(C, C22, ph, rh);
    return C;
}

// ----------------------------------------------------------------------------
// Referencyjne mnozenie klasyczne - NIE zwieksza licznika g_flops.
// Sluzy tylko do sprawdzenia, czy wynik Binet'a jest poprawny.
// ----------------------------------------------------------------------------
Matrix multiplyReference(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k)
            for (int j = 0; j < B.cols; ++j) C(i, j) += A(i, k) * B(k, j);
    return C;
}

// Najwieksza roznica bezwzgledna miedzy odpowiadajacymi sobie elementami.
// Nie bedzie dokladnie 0, bo kolejnosc dodawan jest inna (bledy zaokraglen).
double maxDifference(const Matrix& A, const Matrix& B) {
    double d = 0.0;
    for (size_t k = 0; k < A.data.size(); ++k) d = std::max(d, std::fabs(A.data[k] - B.data[k]));
    return d;
}

// ----------------------------------------------------------------------------
// MODEL LICZBY OPERACJI (dla raportu): dokladnie ta sama rekurencja co
// multiplyBinet, ale zamiast mnozyc macierze tylko SUMUJE koszty. Dzieki temu
// mozna policzyc liczbe operacji dla duzych n bez wykonywania obliczen.
// Wyniki sa identyczne z licznikiem g_flops (sprawdza to tryb --time).
//   - przypadek bazowy: p*r*q mnozen + p*r*(q-1) dodawan,
//   - krok rekurencji: 8 podproblemow + p*r dodawan (4 bloki lacznie p*r elementow).
// Wyniki podproblemow zapamietujemy w mapie (memoizacja) - rozmiary blokow
// przyjmuja na kazdym poziomie tylko 2 rozne wartosci, wiec stanow jest malo.
// ----------------------------------------------------------------------------
unsigned long long flopsModel(int p, int q, int r,
                              std::map<std::tuple<int, int, int>, unsigned long long>& memo) {
    if (p == 1 || q == 1 || r == 1)
        return 1ULL * p * r * q + 1ULL * p * r * (q - 1);
    auto key = std::make_tuple(p, q, r);
    auto it = memo.find(key);
    if (it != memo.end()) return it->second;

    const int ph = (p + 1) / 2, qh = (q + 1) / 2, rh = (r + 1) / 2;
    const int pl = p - ph, ql = q - qh, rl = r - rh;
    // Osiem iloczynow blokow: (ph x qh)(qh x rh), (ph x ql)(ql x rh), itd.
    unsigned long long total = 1ULL * p * r;  // 4 dodawania blokow = p*r operacji
    total += flopsModel(ph, qh, rh, memo) + flopsModel(ph, ql, rh, memo);  // C11
    total += flopsModel(ph, qh, rl, memo) + flopsModel(ph, ql, rl, memo);  // C12
    total += flopsModel(pl, qh, rh, memo) + flopsModel(pl, ql, rh, memo);  // C21
    total += flopsModel(pl, qh, rl, memo) + flopsModel(pl, ql, rl, memo);  // C22
    memo[key] = total;
    return total;
}

int main(int argc, char** argv) {
    // TRYB RAPORTOWY 1:  binet.exe --flops nMin nMax
    // Wypisuje CSV "n,flops" ze wzoru rekurencyjnego (bez mnozenia macierzy).
    if (argc == 4 && std::string(argv[1]) == "--flops") {
        std::map<std::tuple<int, int, int>, unsigned long long> memo;
        std::cout << "n,flops\n";
        for (int n = std::atoi(argv[2]); n <= std::atoi(argv[3]); ++n)
            std::cout << n << "," << flopsModel(n, n, n, memo) << "\n";
        return 0;
    }

    // TRYB RAPORTOWY 2:  binet.exe --time nMin nMax krok powtorzenia
    // Dla kazdego n wykonuje prawdziwe mnozenie, mierzy czas (minimum z kilku
    // powtorzen - ogranicza wplyw zaklocen systemu) i wypisuje CSV
    // "n,flops_zmierzone,czas_ms".
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
                Matrix C = multiplyBinet(A, B);
                auto t1 = std::chrono::steady_clock::now();
                flops = g_flops;
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::cout << n << "," << flops << "," << std::fixed << std::setprecision(4) << best << std::endl;
        }
        return 0;
    }

    // TRYB RAPORTOWY 3:  binet.exe --example
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
        Matrix C = multiplyBinet(A, B);
        std::cout << "% Binet 4x4, liczba operacji = " << g_flops << "\n";
        std::cout << std::setprecision(17);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) std::cout << C(i, j) << (j < 3 ? " " : "\n");
        }
        return 0;
    }

    // Rozmiary do przetestowania: z argumentow wiersza polecen albo domyslne
    // (celowo z nieparzystymi, zeby pokazac, ze algorytm dziala dla dowolnego n).
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {1, 2, 3, 4, 5, 7, 8, 10, 16, 17, 32, 50, 64, 100};

    std::mt19937 gen(12345);  // stale ziarno -> powtarzalne wyniki

    std::cout << "Mnozenie macierzy metoda Binet'a (liczby z [1e-8, 1.0))\n\n";
    std::cout << std::setw(6) << "n" << std::setw(18) << "operacje (+-*/)"
              << std::setw(20) << "klasyczne 2n^3-n^2" << std::setw(14) << "czas [ms]"
              << std::setw(14) << "max |blad|" << "\n";

    for (int n : sizes) {
        if (n < 1) continue;
        Matrix A = randomMatrix(n, gen);
        Matrix B = randomMatrix(n, gen);

        g_flops = 0;  // zerujemy licznik tuz przed wlasciwym mnozeniem
        auto t0 = std::chrono::steady_clock::now();
        Matrix C = multiplyBinet(A, B);
        auto t1 = std::chrono::steady_clock::now();
        const unsigned long long flops = g_flops;  // zapamietujemy zanim cos innego go zmieni

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
