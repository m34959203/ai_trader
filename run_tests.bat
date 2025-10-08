@echo off
setlocal

rem === Путь к проекту ===
set "PRJ=%~dp0"

rem === ВКЛЮЧАЕМ ТЕСТОВЫЙ РЕЖИМ ДЛЯ API ===
rem В этом режиме /prices/store делает purge по (source, symbol, timeframe)
set "APP_ENV=test"

rem === Удаляем «залипающее» состояние лимитов, чтобы не блокировались сделки в тестах ===
set "STATE_FILE=%PRJ%data\state\daily_limits.json"
if exist "%STATE_FILE%" (
  echo [TEST] Removing stale state: "%STATE_FILE%"
  del /F /Q "%STATE_FILE%"
)

rem === Выбираем Python из локального venv, если есть ===
set "PYTHON=%PRJ%.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  rem fallback на системный python
  set "PYTHON=python"
)

echo [INFO] Project root: %CD%\
echo [INFO] Using Python: %PYTHON%
echo [INFO] Running tests: %PYTHON% -m pytest -q

"%PYTHON%" -m pytest -q
endlocal
