@echo off
setlocal
cd /d %~dp0
set PYTHONPATH=%CD%
set UVICORN_HOST=127.0.0.1
set UVICORN_PORT=8001
echo PYTHONPATH=%PYTHONPATH%
uvicorn src.main:app --reload --host %UVICORN_HOST% --port %UVICORN_PORT%
endlocal