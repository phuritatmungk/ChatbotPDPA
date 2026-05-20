@echo off
set PATH=%~dp0poppler\poppler-26.02.0\Library\bin;%PATH%
echo 🚀 Starting document ingestion with Source Citation metadata
echo.

REM Check if PDF files exist in knowledge folder
if not exist "knowledge\*.pdf" (
    echo ❌ No PDF files found in knowledge folder
    echo 💡 Please add PDF files to knowledge folder first
    pause
    exit /b 1
)

echo 📚 Found PDF files in knowledge folder
echo.

REM Activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found!
    echo Please create it first: python -m venv .venv
    pause
    exit /b 1
)

REM Use new ingest system with metadata
echo 🔄 Ingesting documents with Source Citation metadata...
python ingest_uploader.py --path knowledge --with-metadata

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Ingestion completed successfully!
    echo 📚 Documents with Source Citation metadata uploaded to Qdrant
    echo.
    echo 💡 You can now use the RAG system with Source Citation
    echo 🚀 Run: streamlit run app_llama3.2.py
) else (
    echo.
    echo ❌ Error occurred during ingestion
    echo 💡 Please check Qdrant configuration and API keys
)

echo.
pause
