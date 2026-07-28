@echo off

echo Creating virtual environment...
python -m venv .venv

echo Activating...
call .venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Done!
pause