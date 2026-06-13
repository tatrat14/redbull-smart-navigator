@echo off
REM ============================================================
REM   Astana Smart Navigator - one-click launcher (Windows)
REM   Just double-click this file. On the first run it sets up
REM   everything; after that it launches instantly.
REM ============================================================
setlocal
cd /d "%~dp0"

REM --- Find a working Python (prefer the 'py' launcher) -------
set "PYEXE="
where py >nul 2>nul && set "PYEXE=py"
if not defined PYEXE (
  where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
  echo.
  echo [ERROR] Python 3.10+ was not found.
  echo Install it from https://www.python.org/downloads/ and tick
  echo "Add python.exe to PATH", then run this file again.
  echo.
  pause
  exit /b 1
)

REM --- Create the virtual environment on first run -----------
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PYEXE% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create the virtual environment.
    pause
    exit /b 1
  )
)

set "VPY=.venv\Scripts\python.exe"

REM --- Install dependencies once -----------------------------
if not exist ".venv\.deps_installed" (
  echo Installing dependencies. The first run downloads packages
  echo and can take a few minutes - please wait...
  "%VPY%" -m pip install --upgrade pip
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed. See messages above.
    pause
    exit /b 1
  )
  echo installed> ".venv\.deps_installed"
)

REM --- Launch the app (opens in your browser) ----------------
echo.
echo Starting Astana Smart Navigator...
echo It will open in your browser. Close this window or press Ctrl+C to stop.
echo.
"%VPY%" -m streamlit run app.py

pause
