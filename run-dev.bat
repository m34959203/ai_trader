@echo off
setlocal EnableExtensions

rem --- кодировка терминала (опц.) ---
chcp 65001 >NUL

rem --- перейти в корень проекта ---
set "ROOT=%~dp0"
cd /d "%ROOT%"
echo [INFO] Project root: %CD%

rem --- ensure venv ---
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtualenv .venv ...
  py -3.11 -m venv .venv 2>NUL || py -3 -m venv .venv 2>NUL || python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Failed to create .venv. Install Python 3.11+ and py-launcher.
    exit /b 1
  )
)

set "PY=.\.venv\Scripts\python.exe"

rem --- deps (один раз по sentinel) ---
if exist "requirements.txt" if not exist ".venv\.deps.ok" (
  echo [INFO] Installing deps from requirements.txt ...
  "%PY%" -m pip install --upgrade pip || (echo [ERROR] pip upgrade failed & exit /b 1)
  "%PY%" -m pip install -r requirements.txt || (echo [ERROR] pip install failed & exit /b 1)
  > ".venv\.deps.ok" echo ok
)

rem --- загрузка ENV из configs\.env (строки с # игнорируются) ---
if exist "configs\.env" (
  for /f "usebackq eol=# tokens=1* delims==" %%A in ("configs\.env") do (
    if not "%%~A"=="" set "%%~A=%%~B"
  )
  echo [INFO] Loaded environment from configs\.env
)

rem --- дефолты uvicorn ---
if not defined UVICORN_HOST set "UVICORN_HOST=127.0.0.1"
if not defined UVICORN_PORT set "UVICORN_PORT=8001"

set "PYTHONPATH=%CD%"

echo [INFO] Starting Uvicorn on %UVICORN_HOST%:%UVICORN_PORT% (reload)
rem ВАЖНО: без многострочных ^, чтобы cmd.exe не «съел» аргументы
"%PY%" -m uvicorn "src.main:app" --reload --host "%UVICORN_HOST%" --port "%UVICORN_PORT%" --proxy-headers --timeout-graceful-shutdown 25 --timeout-keep-alive 5

rem Примечание: флаг --forwarded-allow-ips "*" для дев-старта не обязателен.
rem По умолчанию proxy-headers включены и ограничены доверенными IP (см. docs).
rem При необходимости в проде добавьте:  --forwarded-allow-ips="*"
rem или конкретный список IP. :contentReference[oaicite:2]{index=2}

endlocal
