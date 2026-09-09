@echo off
cd /d "%~dp0"

set PORT=8088
set PYTHON_EXE=python\python.exe

if exist "%PYTHON_EXE%" goto start_server

echo ============================================================
echo   Embedded Python runtime not detected.
echo   Launching automated setup to download portable engine...
echo ============================================================
echo.
call setup_portable_env.bat

if not exist "%PYTHON_EXE%" goto setup_failed
goto start_server

:setup_failed
echo [ERROR] Embedded Python could not be configured.
pause
exit /b 1

:start_server
echo ============================================================
echo   Starting ChainPDF Portable Python Server on port %PORT%...
echo ============================================================
start "" http://localhost:%PORT%/
"%PYTHON_EXE%" server.py