// BITMAPA W FORMACIE PPM (P6) I ZAMIANA NA MACIERZE R, G, B
//
// PPM to najprostszy format bitmapy: naglowek tekstowy "P6 szerokosc wysokosc 255"
// i dalej bajty R G B kolejnych pikseli, wierszami.

#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "matrix.hpp"

struct Image {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;  // 3 bajty na piksel

    Image() = default;
    Image(int w, int h, uint8_t v = 255) : width(w), height(h), rgb(static_cast<size_t>(w) * h * 3, v) {}
    uint8_t& at(int row, int col, int ch) { return rgb[(static_cast<size_t>(row) * width + col) * 3 + ch]; }
    uint8_t at(int row, int col, int ch) const { return rgb[(static_cast<size_t>(row) * width + col) * 3 + ch]; }
};

inline Image readPPM(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("nie mozna otworzyc " + path);
    auto token = [&]() {
        std::string t;
        while (t.empty()) {
            in >> t;
            if (!t.empty() && t[0] == '#') {  // komentarz do konca linii
                std::getline(in, t);
                t.clear();
            }
        }
        return t;
    };
    if (token() != "P6") throw std::runtime_error(path + ": oczekiwano PPM P6");
    const int w = std::stoi(token()), h = std::stoi(token());
    if (std::stoi(token()) != 255) throw std::runtime_error(path + ": oczekiwano maxval 255");
    in.get();  // jeden bialy znak po naglowku
    Image img(w, h);
    in.read(reinterpret_cast<char*>(img.rgb.data()), static_cast<std::streamsize>(img.rgb.size()));
    return img;
}

inline void writePPM(const std::string& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.rgb.data()), static_cast<std::streamsize>(img.rgb.size()));
}

// Kanal ch (0 = R, 1 = G, 2 = B) jako macierz wartosci z [0, 255].
inline Matrix channel(const Image& img, int ch) {
    Matrix M(img.height, img.width);
    for (int i = 0; i < img.height; ++i)
        for (int j = 0; j < img.width; ++j) M(i, j) = img.at(i, j, ch);
    return M;
}

inline uint8_t toByte(double x) {
    return static_cast<uint8_t>(std::lround(std::min(255.0, std::max(0.0, x))));
}

// Bitmapa z trzech macierzy (wartosci przyciete do [0, 255]).
inline Image fromChannels(const Matrix& R, const Matrix& G, const Matrix& B) {
    Image img(R.cols, R.rows);
    for (int i = 0; i < R.rows; ++i)
        for (int j = 0; j < R.cols; ++j) {
            img.at(i, j, 0) = toByte(R(i, j));
            img.at(i, j, 1) = toByte(G(i, j));
            img.at(i, j, 2) = toByte(B(i, j));
        }
    return img;
}

// Jeden kanal w jego kolorze (np. R -> odcienie czerwieni, pozostale kanaly = 0).
inline Image singleChannel(const Matrix& M, int ch) {
    Image img(M.cols, M.rows, 0);
    for (int i = 0; i < M.rows; ++i)
        for (int j = 0; j < M.cols; ++j) img.at(i, j, ch) = toByte(M(i, j));
    return img;
}
