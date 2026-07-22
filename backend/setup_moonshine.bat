@echo off
echo Creating Moonshine isolated virtual environment...
cd /d "%~dp0"

if exist moonshine_env (
    echo moonshine_env already exists. Recreating...
    rmdir /s /q moonshine_env
)

python -m venv moonshine_env
call moonshine_env\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing Moonshine dependencies...
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=5.0.0" librosa soundfile numpy huggingface-hub safetensors

echo.
echo Moonshine venv setup complete!
echo To test: moonshine_env\Scripts\python.exe app\moonshine_worker.py --test
pause
