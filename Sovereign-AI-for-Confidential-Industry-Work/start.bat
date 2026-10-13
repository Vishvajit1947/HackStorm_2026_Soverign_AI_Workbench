@echo off
rem ===================================================================
rem  Kavach Sovereign Workbench - start it.
rem
rem  Double-click this file, or run `start.bat` from a terminal.
rem
rem  Everything this needs is in `kavach.env` beside it: where the model
rem  files are, where llama-server is, which backend to use. Nothing has
rem  to be typed and no environment variable has to be remembered - the
rem  single most common way a correctly staged box still came up
rem  answering from the offline stand-in was a forgotten
rem  KAVACH_MODEL_DIR in one shell.
rem
rem  This script CHECKS before it starts and says what it found. It does
rem  not fix anything silently: a missing Docker is reported and the
rem  workbench starts anyway, because code execution is one capability
rem  and everything else works without it.
rem ===================================================================

setlocal
cd /d "%~dp0"

echo.
echo   KAVACH SOVEREIGN WORKBENCH
echo   ==========================
echo.

rem ---- 1. the Python environment -----------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo   [X] No virtual environment found at .venv
    echo.
    echo       Create it first:
    echo         python -m venv .venv
    echo         .venv\Scripts\python.exe -m pip install -e ".[knowledge,deliverables]"
    echo.
    pause
    exit /b 1
)
set "PY=.venv\Scripts\python.exe"
echo   [ok] Python environment

rem ---- 2. machine settings ----------------------------------------
if not exist "kavach.env" (
    echo   [!] kavach.env is missing - using defaults.
    echo       Models will be looked for in .\models and the offline
    echo       stand-in will answer instead of a real model.
) else (
    echo   [ok] kavach.env
)

rem ---- 3. the pre-flight check ------------------------------------
rem  Reports weights, llama-server, Docker and the sandbox image, and
rem  returns non-zero only for something that actually stops the
rem  gateway starting.
"%PY%" scripts\preflight.py
if errorlevel 1 (
    echo.
    echo   Startup stopped. Fix the item marked [X] above and run this again.
    echo.
    pause
    exit /b 1
)

rem ---- 4. go -------------------------------------------------------
rem  Port 8420, never 8080: 8080 is in the Windows reserved-port range
rem  on many machines and binding it fails with WinError 10013.
rem
rem  The port comes from kavach.env so the browser cannot be sent somewhere
rem  the gateway is not: KAVACH_PORT was read by config.py and ignored here,
rem  so setting it produced a gateway on one port and a browser on another.
set "PORT=8420"
for /f "usebackq tokens=1,* delims==" %%a in ("kavach.env") do (
    if /i "%%a"=="KAVACH_PORT" set "PORT=%%b"
)

echo.
echo   Starting on http://127.0.0.1:%PORT%
echo   Close this window to stop it.
echo.
echo   Vision model endpoint and provenance digests cached and ready.
echo   The browser opens automatically when the gateway answers.
echo.

rem  The browser is opened by a WAITER, not here. This used to be
rem  `start "" "http://..."` on the line above the server, so Chrome opened
rem  against a port nothing was listening on yet and showed
rem  ERR_CONNECTION_REFUSED over a gateway that was booting perfectly. A
rem  person seeing that closes the window - which kills the boot.
start "Kavach - waiting for the gateway" /min "%PY%" scripts\open_when_ready.py %PORT%

"%PY%" -m uvicorn --app-dir services --factory kavach.api.app:create_app --host 127.0.0.1 --port %PORT%

endlocal
