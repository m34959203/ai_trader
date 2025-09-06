@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ── консоль в UTF-8 ───────────────────────────────────────────────────────────
chcp 65001 >NUL
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

:: ── корень проекта ────────────────────────────────────────────────────────────
pushd "%~dp0"

echo [INFO] Project root: %CD%

:: ── поиск python ──────────────────────────────────────────────────────────────
where python >NUL 2>&1
if errorlevel 1 (
  echo [ERROR] Python не найден в PATH.
  echo        Установите Python 3.11+ и добавьте в PATH.
  exit /b 1
)

:: ── виртуальное окружение ─────────────────────────────────────────────────────
if not exist ".venv" (
  echo [INFO] Создаю .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Не удалось создать .venv
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Не удалось активировать .venv
  exit /b 1
)

:: ── обновление pip + зависимости ─────────────────────────────────────────────
python -m pip install --upgrade pip >NUL
if exist requirements-dev.txt (
  echo [INFO] Устанавливаю зависимости из requirements-dev.txt ...
  python -m pip install -r requirements-dev.txt
) else if exist requirements.txt (
  echo [INFO] Устанавливаю зависимости из requirements.txt ...
  python -m pip install -r requirements.txt
) else (
  echo [WARN] Файл требований не найден. Пропускаю установку зависимостей.
)

:: ── переменные окружения для тестов ──────────────────────────────────────────
:: Чтобы модульный код видел локальные пакеты
set "PYTHONPATH=%CD%;%PYTHONPATH%"

:: Тише логи, быстрее тесты
set "LOG_LEVEL=WARNING"

:: Симулятор по умолчанию — достаточно для большинства тестов
set "SIM_EQUITY_USDT=100000"
set "SIM_BTCUSDT_PRICE=60000"

:: SQLite/прочие флаги (безопасно оставить по умолчанию)
set "SQLITE_WAL=1"
set "SQLITE_SYNC_NORMAL=1"
set "SQLITE_FOREIGN_KEYS=1"

:: Отключить автозагрузку swagger (ускоряет старты приложений в тестах)
set "DISABLE_DOCS=1"

:: ── запуск pytest ─────────────────────────────────────────────────────────────
echo.
echo [INFO] Запускаю pytest %* ...
python -m pytest %*
set "PYTEST_EXIT=%ERRORLEVEL%"

:: ── финал ────────────────────────────────────────────────────────────────────
if "%PYTEST_EXIT%"=="0" (
  echo.
  echo [OK] Все тесты пройдены.
) else (
  echo.
  echo [FAIL] Тесты завершились с кодом %PYTEST_EXIT%.
)

popd
exit /b %PYTEST_EXIT%
