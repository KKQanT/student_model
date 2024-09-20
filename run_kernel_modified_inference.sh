@echo off

:: Check if the virtual environment exists
if not exist venv (
    echo Virtual environment not found. Please create it first.
    pause
    exit /b
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Run the Python script
echo Running Kernel_Modified_cost_sensitive_SVM_balanced_inference.py...
python Kernel_Modified_cost_sensitive_SVM_balanced_inference.py

pause
