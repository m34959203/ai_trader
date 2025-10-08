@echo off
setlocal
cd /d C:\Users\Дмитрий\Desktop\ai_trader

rem 0) опционально: удалить старую .venv
if exist .venv (
  echo Removing old .venv ...
  rmdir /s /q .venv
)

rem 1) создать venv
py -3.11 -m venv .venv

rem 2) обновить pip внутри venv
.\.venv\Scripts\python -m pip install --upgrade pip

rem 3) установить зависимости проекта
.\.venv\Scripts\python -m pip install -r requirements.txt

echo.
echo === VENV READY ===
.\.venv\Scripts\python -V
.\.venv\Scripts\python -c "import sys; print('venv:', sys.prefix)"
endlocal
