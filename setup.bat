@echo off

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

if exist requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found!
)

if not exist data (
    echo Creating data folder...
    mkdir data
) else (
    echo data folder already exists.
)

pause
