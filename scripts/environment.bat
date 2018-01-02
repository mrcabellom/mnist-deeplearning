@echo off
call conda-env create --file ../anaconda_env.yml
if %errorlevel% neq 0 ( 
    echo "Hubo un error creando el entorno virtual" 
    exit /b %ERRORLEVEL%
    ) else (
        activate cntk22dotnetmalaga
    )
