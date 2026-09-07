@echo off
cd /d "%~dp0"

echo ============================================================
echo   ChainPDF - Portable Python and Tesseract Setup
echo ============================================================
echo.

:: 1. Setup Embedded Python
if exist "python\python.exe" (
    echo [OK] Embedded Python already installed in python\
    goto step_dependencies
)

echo [1/4] Downloading Python 3.11 Embeddable (Windows 64-bit)...
if not exist "python" mkdir "python"
curl -L -o python_embed.zip "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
if errorlevel 1 (
    echo [ERROR] Failed to download Python embeddable zip.
    pause
    exit /b 1
)

echo [1/4] Extracting Python to python\...
powershell -NoProfile -Command "Expand-Archive -Path 'python_embed.zip' -DestinationPath 'python' -Force"
if exist python_embed.zip del python_embed.zip

echo [1/4] Configuring python311._pth...
(
echo python311.zip
echo .
echo ..
echo Lib\site-packages
echo import site
) > python\python311._pth

:step_dependencies
:: 2. Install Python Dependencies directly into Lib\site-packages
echo.
echo [2/4] Verifying Python library wheels (pillow, numpy, pytesseract, pymupdf)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "
$targetDir = Join-Path $pwd 'python\Lib\site-packages'
if (!(Test-Path $targetDir)) { New-Item -ItemType Directory -Force -Path $targetDir | Out-Null }

$wheels = @(
    @{ Name = 'pillow'; Url = 'https://files.pythonhosted.org/packages/63/c6/4bad1b18d132a50b27e1365e1ab163616f7a5bb56d330f66f9d1d9d4f9d4/pillow-12.3.0-cp311-cp311-win_amd64.whl' },
    @{ Name = 'numpy'; Url = 'https://files.pythonhosted.org/packages/b9/c6/cd4298729826af9979c5f9ab02fcaa344b82621e7c49322cd2d210483d3f/numpy-2.2.3-cp311-cp311-win_amd64.whl' },
    @{ Name = 'packaging'; Url = 'https://files.pythonhosted.org/packages/63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl' },
    @{ Name = 'pytesseract'; Url = 'https://files.pythonhosted.org/packages/7a/33/8312d7ce74670c9d39a532b2c246a853861120486be9443eebf048043637/pytesseract-0.3.13-py3-none-any.whl' },
    @{ Name = 'pymupdf'; Url = 'https://files.pythonhosted.org/packages/4a/61/d563bbccba262f9dd6d2d35ccb72593648184d886188efb12d9ce8f34dd6/pymupdf-1.28.2-cp310-abi3-win_amd64.whl' }
)

$wheelsDir = Join-Path $pwd 'wheels'
if (!(Test-Path $wheelsDir)) { New-Item -ItemType Directory -Force -Path $wheelsDir | Out-Null }

foreach ($w in $wheels) {
    $whlName = [System.IO.Path]::GetFileName($w.Url)
    $whlPath = Join-Path $wheelsDir $whlName
    if (!(Test-Path $whlPath)) {
        Write-Host ('Downloading ' + $w.Name + '...')
        curl.exe -L -o $whlPath $w.Url
    }
    $zipPath = $whlPath + '.zip'
    if (!(Test-Path $zipPath)) { Copy-Item -Path $whlPath -Destination $zipPath }
    Expand-Archive -Path $zipPath -DestinationPath $targetDir -Force
    Remove-Item -Path $zipPath -Force
}
"
echo [OK] All Python dependencies installed.

:: 3. Setup Portable Tesseract OCR
echo.
if exist "tesseract\tesseract.exe" (
    echo [OK] Portable Tesseract OCR already installed in tesseract\
    goto step_tessdata
)

echo [3/4] Downloading Portable Tesseract OCR 5.4.0 (Windows 64-bit)...
curl -L -o tesseract_setup.exe "https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe"
if errorlevel 1 (
    echo [ERROR] Failed to download Tesseract setup.
    pause
    exit /b 1
)

echo [3/4] Extracting portable Tesseract to tesseract\...
if exist "C:\Program Files\7-Zip\7z.exe" (
    "C:\Program Files\7-Zip\7z.exe" x tesseract_setup.exe -otesseract -y >nul
) else (
    where 7z >nul 2>nul
    if errorlevel 1 (
        echo Fetching standalone extraction tool...
        curl -L -o 7za.zip "https://www.7-zip.org/a/7za920.zip"
        powershell -NoProfile -Command "Expand-Archive -Path '7za.zip' -DestinationPath '.' -Force"
        .\7za.exe x tesseract_setup.exe -otesseract -y >nul
        if exist 7za.exe del 7za.exe
        if exist 7za.dll del 7za.dll
        if exist 7za.zip del 7za.zip
        if exist 7-zip.chm del 7-zip.chm
        if exist readme.txt del readme.txt
    ) else (
        7z x tesseract_setup.exe -otesseract -y >nul
    )
)
if exist tesseract_setup.exe del tesseract_setup.exe
echo [OK] Tesseract installed in project folder.

:step_tessdata
:: 4. Extract Tessdata Language Models
echo.
echo [4/4] Verifying Tessdata language models...
if not exist "tessdata" mkdir "tessdata"
if exist "tessdata\eng.traineddata.gz" (
    if not exist "tessdata\eng.traineddata" (
        echo Extracting eng.traineddata from gzip...
        .\python\python.exe -c "import gzip, shutil; shutil.copyfileobj(gzip.open('tessdata/eng.traineddata.gz', 'rb'), open('tessdata/eng.traineddata', 'wb'))"
    )
)

:: 5. Verification Test
echo.
echo ============================================================
echo   Verifying Portable Engine Integrity
echo ============================================================
.\python\python.exe -c "import pytesseract, PIL, numpy, pymupdf; print('[SUCCESS] Python runtime and dependencies ready!')"
if exist "tesseract\tesseract.exe" (
    echo [SUCCESS] Tesseract binary verified at: %~dp0tesseract\tesseract.exe
)
echo.
echo Setup Complete! You can now launch the studio using launch.bat
echo ============================================================
