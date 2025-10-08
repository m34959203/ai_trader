@echo off
setlocal enableextensions

rem ──────────────────────────────────────────────────────────────
rem 1) Переключаемся в каталог скрипта и включаем UTF-8
rem ──────────────────────────────────────────────────────────────
pushd "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"

rem ──────────────────────────────────────────────────────────────
rem 2) Определяем Python: venv приоритетнее системного
rem ──────────────────────────────────────────────────────────────
set "PRJ=%CD%\"
set "VENV_PY=%PRJ%.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  set "PYEXE=%VENV_PY%"
) else (
  set "PYEXE=python"
)

rem ──────────────────────────────────────────────────────────────
rem 3) Чистим залипающий state перед тестами
rem ──────────────────────────────────────────────────────────────
set "STATE_FILE=%PRJ%data\state\daily_limits.json"
if exist "%STATE_FILE%" (
  echo [TEST] Removing stale state: "%STATE_FILE%"
  del /f /q "%STATE_FILE%"
) else (
  echo [TEST] No stale state to remove.
)

rem ──────────────────────────────────────────────────────────────
rem 4) Запускаем pytest (с пробросом любых доп. аргументов)
rem ──────────────────────────────────────────────────────────────
echo [INFO] Project root: %PRJ%
echo [INFO] Using Python: %PYEXE%
echo [INFO] Running tests: %PYEXE% -m pytest -q %*
"%PYEXE%" -m pytest -q %*
set "CODE=%ERRORLEVEL%"

rem ──────────────────────────────────────────────────────────────
rem 5) Возврат в исходную папку и корректный код выхода
rem ──────────────────────────────────────────────────────────────
popd
exit /b %CODE%
