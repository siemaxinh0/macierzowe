# Pelny przebieg zadania 3: kompilacja, bitmapa, wartosci osobliwe, 6 konfiguracji z zadania,
# najlepsza kompresja, przeszukanie parametrow i rysunki do sprawozdania.
# Uruchomienie (z folderu task3):  .\run_all.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
.\build.ps1

Set-Location report
New-Item -ItemType Directory -Force out, data, fig | Out-Null
python prepare_image.py
..\compress.exe sv data\bitmapa.ppm | Out-File -Encoding ascii data\singular_values.csv

# nazwa, r, delta (s1 = sigma_1, slast = sigma_{2^k} = sigma_512, s256 = sigma_{2^k/2}, liczba = prog bezwzgledny)
$configs = @(
    @("r1_s1", 1, "s1"), @("r1_slast", 1, "slast"), @("r1_shalf", 1, "s256"),
    @("r4_s1", 4, "s1"), @("r4_slast", 4, "slast"), @("r4_shalf", 4, "s256"),
    @("best", 1, "50")
)
foreach ($c in $configs) {
    ..\compress.exe run data\bitmapa.ppm $c[1] $c[2] "out\$($c[0])" | Out-File -Encoding ascii "data\$($c[0]).csv"
    Get-Content "data\$($c[0]).csv" | Select-Object -Last 1
}

"r,delta,ratio,psnr" | Out-File -Encoding ascii data\grid.csv
foreach ($r in 1, 2, 4, 8, 16) {
    foreach ($d in 25, 50, 100, 200, 400, 800) {
        $f = (..\compress.exe run data\bitmapa.ppm $r $d "out\grid" | Select-String "^RGB").ToString().Split(",")
        "$r,$d,$($f[6]),$($f[9] -replace 'psnr=', '')" | Out-File -Append -Encoding ascii data\grid.csv
    }
}

python make_figures.py
