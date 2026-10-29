g++ -O2 -std=c++17 -o inverse.exe inverse.cpp
g++ -O2 -std=c++17 -o test_mul.exe test_mul.cpp
if (Test-Path lu.cpp) { g++ -O2 -std=c++17 -o lu.exe lu.cpp }
