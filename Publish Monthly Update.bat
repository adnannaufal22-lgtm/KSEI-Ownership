@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto validate

echo Preparing the dashboard tools for first use...
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

:validate
echo Validating the monthly BEI files...
".venv\Scripts\python.exe" validate_data.py
if errorlevel 1 goto validation_failed

git add -- BEI_Data
git diff --cached --quiet
if not errorlevel 1 goto no_changes

git commit -m "Update monthly BEI ownership data"
if errorlevel 1 goto publish_failed
git push
if errorlevel 1 goto publish_failed

echo.
echo Monthly data published. The hosted dashboard will refresh automatically.
pause
goto end

:no_changes
echo.
echo No new or changed files were found in BEI_Data.
pause
goto end

:python_missing
echo Python was not found. Open this project through Codex or install Python, then try again.
pause
goto end

:setup_failed
echo The one-time setup did not finish. Check your internet connection and try again.
pause
goto end

:validation_failed
echo.
echo The BEI data was not published because validation failed.
pause
goto end

:publish_failed
echo.
echo GitHub publishing did not finish. Sign in to GitHub if prompted, then run this file again.
pause

:end
endlocal

