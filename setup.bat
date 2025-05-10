@echo off
title WITS CrewAI Setup Script
color 0A

echo ################################################################################
echo #                                                                              #
echo #                      WITS CrewAI System Setup Script                         #
echo #                                                                              #
echo ################################################################################
echo.
echo This script will help you set up the WITS CrewAI environment.
echo Please ensure you have Python 3.8+ installed and added to your PATH.
echo.
pause
echo.

REM --- Check for Python ---
echo --- 1. Checking for Python ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python does not seem to be installed or is not in your PATH.
    echo Please install Python (3.8 or newer) from python.org and ensure it's added to your PATH.
    pause
    exit /b 1
)
python --version
echo Python found.
echo.

REM --- Check for Pip ---
echo --- 2. Checking for Pip ---
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Pip does not seem to be installed or is not in your PATH.
    echo Pip usually comes with Python. Please check your Python installation.
    echo You might need to run: python -m ensurepip --upgrade
    pause
    exit /b 1
)
pip --version
echo Pip found.
echo.

REM --- Create Virtual Environment ---
echo --- 3. Setting up Virtual Environment ---
if exist venv (
    echo Virtual environment 'venv' already exists. Skipping creation.
) else (
    echo Creating virtual environment 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)
echo.

REM --- Install Dependencies ---
echo --- 4. Installing Dependencies from requirements.txt ---
echo This might take a while...
echo.
echo IMPORTANT: Your requirements.txt includes 'faiss-gpu'. 
echo If you encounter errors during its installation, you may need to:
echo   a) Ensure you have a compatible NVIDIA GPU and the correct CUDA Toolkit installed.
echo   b) Have C++ Build Tools (e.g., from Visual Studio) installed.
echo   c) Or, if you don't need/have GPU support for FAISS, you can edit
echo      'requirements.txt', change 'faiss-gpu' to 'faiss-cpu', and re-run this script.
echo.

call .\venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies from requirements.txt.
    echo Please check the error messages above. You might need to resolve issues manually.
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

REM --- Create Necessary Directories ---
echo --- 5. Creating Output Directories ---
if not exist "output" mkdir "output"
if not exist "output\visualizations" mkdir "output\visualizations"
if not exist "output\documents" mkdir "output\documents"
if not exist "output\research_notes" mkdir "output\research_notes"
if not exist "output\plans" mkdir "output\plans"
echo Output directories checked/created.
echo.

REM --- Ollama Reminder ---
echo --- 6. Ollama Reminder ---
echo IMPORTANT: WITS CrewAI requires Ollama to be installed and running.
echo Please ensure Ollama is installed from ollama.com.
echo.
echo After installing Ollama, you need to pull the models specified in your 'config.yaml'.
echo For example, if your config uses 'llama2' and 'codellama:7b', run:
echo   ollama pull llama2
echo   ollama pull codellama:7b
echo (And any other models your 'config.yaml' specifies for different agents).
echo Make sure Ollama is running before starting the WITS CrewAI application.
echo.
pause
echo.

REM --- Config.yaml Reminder ---
echo --- 7. Configuration File Reminder ---
echo Please ensure you have a 'config.yaml' file in the project root.
echo You can copy 'config.example.yaml' (if provided) to 'config.yaml' 
echo and customize it for your setup, especially the model names and API keys if any.
echo The current setup requires you to manually create/verify 'config.yaml'.
echo.
pause
echo.

REM --- Setup Complete ---
echo ################################################################################
echo #                                                                              #
echo #                       Setup Process Completed!                               #
echo #                                                                              #
echo ################################################################################
echo.
echo To run the WITS CrewAI application:
echo 1. Activate the virtual environment in your terminal:
echo    .\venv\Scripts\activate
echo 2. Then run the application (CLI or Web):
echo    For CLI mode: python main.py
echo    For Web UI mode: python main.py --web  (or python app.py if you run it directly)
echo.
echo Remember to have Ollama running with the required models pulled.
echo.
pause
exit /b 0
