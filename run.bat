@echo off
cd /d "%~dp0"
title Video Crop Tool
color 0A

echo ==============================
echo   Video Crop Automation
echo ==============================
echo.

:: Detect python3 first, fallback to python
echo Checking Python executable...
python3 --version >nul 2>&1
if errorlevel 1 (
    set PY=python
) else (
    set PY=python3
)

echo Using %PY%

:: Create venv if missing
if not exist venv (
    echo Creating virtual environment...
    %PY% -m venv venv
)

:: Activate venv
call venv\Scripts\activate

:: Upgrade pip and install dependencies
echo Installing required packages (opencv-python, numpy, tqdm, moviepy)...
%PY% -m pip install --upgrade pip
%PY% -m pip install opencv-python numpy tqdm moviepy==1.0.3

:: Ask user for folder path
set /p FOLDER_PATH=Enter path to folder containing videos (leave empty for current folder):

:: Run script with argument if provided
if "%FOLDER_PATH%"=="" (
    echo Running video crop script on current folder...
    %PY% main.py
) else (
    echo Running video crop script on: %FOLDER_PATH%
    %PY% main.py "%FOLDER_PATH%"
)

echo.
echo ==============================
echo     Task finished!
echo ==============================
pause
