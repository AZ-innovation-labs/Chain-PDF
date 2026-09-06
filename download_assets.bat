@echo off
cd /d "%~dp0"
echo Downloading offline dependencies for Tesseract and PDF-Lib...

:: Create tessdata folder
if not exist "tessdata" mkdir "tessdata"

:: Download Libraries & WebAssembly Core
echo [1/6] Downloading pdf-lib.min.js...
curl -L -o pdf-lib.min.js "https://unpkg.com/pdf-lib@1.17.1/dist/pdf-lib.min.js"

echo [2/6] Downloading tesseract.min.js...
curl -L -o tesseract.min.js "https://unpkg.com/tesseract.js@5.1.1/dist/tesseract.min.js"

echo [3/6] Downloading worker.min.js...
curl -L -o worker.min.js "https://unpkg.com/tesseract.js@5.1.1/dist/worker.min.js"

echo [4/6] Downloading tesseract-core-simd-lstm.wasm.js...
curl -L -o tesseract-core-simd-lstm.wasm.js "https://unpkg.com/tesseract.js-core@5.1.1/tesseract-core-simd-lstm.wasm.js"

echo [5/6] Downloading tesseract-core-simd-lstm.wasm...
curl -L -o tesseract-core-simd-lstm.wasm "https://unpkg.com/tesseract.js-core@5.1.1/tesseract-core-simd-lstm.wasm"

:: Download English OCR model
echo [6/6] Downloading eng.traineddata.gz...
curl -L -o tessdata\eng.traineddata.gz "https://github.com/naptha/tessdata/raw/gh-pages/4.0.0_fast/eng.traineddata.gz"

echo.
echo All assets downloaded successfully. You can now use the tool completely offline.
pause