@echo off
cd /d "%~dp0"

set PORT=8000
set TARGET_FILE=chainpdf.html

if not exist "%TARGET_FILE%" (
    if exist "chainpdf\%TARGET_FILE%" (
        cd /d "%~dp0chainpdf"
    ) else (
        echo [ERROR] Could not find %TARGET_FILE%.
        pause
        exit /b
    )
)

:: Check if Python is available
where python >nul 2>nul
if %errorlevel% equ 0 (
    echo Starting local server on port %PORT%...
    start "" http://localhost:%PORT%/%TARGET_FILE%
    python -m http.server %PORT%
) else (
    echo Python not detected. Falling back to direct browser launch...
    start "" "%TARGET_FILE%"
)