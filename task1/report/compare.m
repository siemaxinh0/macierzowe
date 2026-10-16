% compare.m - porownanie wynikow programow C++ z mnozeniem w Octave / MATLAB.
%
% Uzycie (z folderu task1/report):
%   1) wygeneruj wyniki z programow C++:
%        ..\binet.exe --example    > data\example_binet.txt
%        ..\strassen.exe --example > data\example_strassen.txt
%        ..\laderman.exe --example > data\example_laderman.txt   (opcjonalnie)
%   2) uruchom ten skrypt w Octave lub MATLAB:  compare
%      Inny katalog z danymi: ustaw dataDir przed wywolaniem, np.
%        dataDir = 'data_mateusz'; compare
%
% Przyklad: A = magic(4), B = A'/7 (te same macierze sa zaszyte w programach C++).

if ~exist('dataDir', 'var'), dataDir = 'data'; end

A = magic(4);
B = A' / 7;
Cref = A * B;                       % wynik referencyjny (wbudowane mnozenie)

Cbinet    = load([dataDir '/example_binet.txt']);     % linie z '%' sa komentarzami
Cstrassen = load([dataDir '/example_strassen.txt']);

format long
disp('Wynik referencyjny A*B (Octave/MATLAB):'); disp(Cref)
fprintf('max |Binet    - A*B| = %.3e\n', max(abs(Cbinet(:)    - Cref(:))));
fprintf('max |Strassen - A*B| = %.3e\n', max(abs(Cstrassen(:) - Cref(:))));
fprintf('max |Binet - Strassen| = %.3e\n', max(abs(Cbinet(:) - Cstrassen(:))));

if exist([dataDir '/example_laderman.txt'], 'file')
    Claderman = load([dataDir '/example_laderman.txt']);
    fprintf('max |Laderman - A*B| = %.3e\n', max(abs(Claderman(:) - Cref(:))));
end
