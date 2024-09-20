@echo off

if not exist venv (
    echo Virtual environment not found. Please create it first.
    pause
    exit /b
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Running Kernel_Modified_cost_sensitive_SVM_balanced_inference.py...
python Kernel_Modified_cost_sensitive_SVM_balanced_inference.py

pause