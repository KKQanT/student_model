@echo off

if not exist venv (
    echo Virtual environment not found. Please create it first.
    pause
    exit /b
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Running FSVM_class_center_rieman_diff_dmax_linear_train_inference.py...
python FSVM_class_center_rieman_diff_dmax_linear_train_inference.py

pause