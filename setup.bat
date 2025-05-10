@echo off
REM WITS CrewAI Setup Script

echo Checking Python version...
for /f "delims=" %%i in ('python -c "import sys; print(sys.version_info[0])"') do set PY_MAJOR=%%i
for /f "delims=" %%i in ('python -c "import sys; print(sys.version_info[1])"') do set PY_MINOR=%%i
if %PY_MAJOR% LSS 3 (
    echo ERROR: Python 3.10 or higher is required.
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo ERROR: Python 3.10 or higher is required.
    exit /b 1
)
echo Python version OK: %PY_MAJOR%.%PY_MINOR%

REM Check for Ollama
where ollama >nul 2>&1
if %errorlevel% NEQ 0 (
    echo WARNING: Ollama not found. Please install Ollama from https://ollama.com/download/windows
) else (
    echo Ollama found.
)

REM Install Python dependencies
echo Installing Python dependencies...
pip install crewai
pip install ollama
pip install openai-whisper
pip install pyyaml

REM Check FFmpeg for Whisper
where ffmpeg >nul 2>&1
if %errorlevel% NEQ 0 (
    echo WARNING: FFmpeg not found. Whisper transcription requires FFmpeg.
) else (
    echo FFmpeg found.
)

echo Setup complete. You can now run the WITS CrewAI system.
