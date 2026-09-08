@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto launch

echo Preparing the dashboard for first use...
set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%PYTHON_EXE%" goto create_environment

where python >nul 2>&1
if errorlevel 1 goto python_missing
set "PYTHON_EXE=python"

:create_environment
"%PYTHON_EXE%" -m venv .venv
if errorlevel 1 goto setup_failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto setup_failed

:launch
echo Opening KSEI Ownership Dashboard...
".venv\Scripts\python.exe" -m streamlit run app.py
goto end

:python_missing
echo Python was not found. Open this project through Codex or install Python, then try again.
pause
goto end

:setup_failed
echo The one-time dashboard setup did not finish. Check your internet connection and try again.
pause

:end
endlocal
