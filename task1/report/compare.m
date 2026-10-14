% compare.m - porownanie wynikow programow C++ z mnozeniem w Octave / MATLAB.
%
% Uzycie (z folderu task1/report):
%   1) wygeneruj wyniki z programow C++:
%        ..\binet.exe --example    > data\example_binet.txt
%        ..\strassen.exe --example > data\example_strassen.txt
%   2) uruchom ten skrypt w Octave lub MATLAB:  compare
%
% Przyklad: A = magic(4), B = A'/7 (te same macierze sa zaszyte w programach C++).

A = magic(4);
B = A' / 7;
Cref = A * B;                       % wynik referencyjny (wbudowane mnozenie)

Cbinet    = load('data/example_binet.txt');     % linie z '%' sa komentarzami
Cstrassen = load('data/example_strassen.txt');

format long
disp('Wynik referencyjny A*B (Octave/MATLAB):'); disp(Cref)
fprintf('max |Binet    - A*B| = %.3e\n', max(abs(Cbinet(:)    - Cref(:))));
fprintf('max |Strassen - A*B| = %.3e\n', max(abs(Cstrassen(:) - Cref(:))));
fprintf('max |Binet - Strassen| = %.3e\n', max(abs(Cbinet(:) - Cstrassen(:))));
