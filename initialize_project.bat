@echo off
chcp 65001 >nul 2>&1  :: 解决中文显示问题（不影响英文输出）

echo ==============================================
echo          AI Music Program Environment Setup
echo ==============================================
echo.

:: 1. 检查目标文件夹是否存在，不存在则新建
set "TARGET_DIR=D:\Program -AI music"
if not exist "%TARGET_DIR%" (
    echo Error: Target folder does not exist - %TARGET_DIR%
    echo Creating folder automatically...
    mkdir "%TARGET_DIR%" || (
        echo Error: Failed to create folder. Please check permissions.
        pause
        exit /b 1
    )
    echo Folder created successfully: %TARGET_DIR%
) else (
    echo Target folder exists: %TARGET_DIR%
)

:: 2. 进入目标文件夹并检查所需文件（requirements.txt）是否存在
cd /d "%TARGET_DIR%" || (
    echo Error: Failed to access folder - %TARGET_DIR%. Please copy the codes into %TARGET_DIR%.
    pause
    exit /b 1
)

set "REQUIREMENTS=requirements.txt"
if not exist "%REQUIREMENTS%" (
    echo Error: Dependency file not found - %REQUIREMENTS%
    echo Please place the file in the folder and try again: %TARGET_DIR%
    pause
    exit /b 1
) else (
    echo Dependency file exists: %REQUIREMENTS%
)

:: 3. 检查Python 3.11.9是否已安装
echo.
echo Checking for Python 3.11.9...

:: 先尝试获取Python版本
python --version >nul 2>&1

if %errorlevel% equ 0 (
    :: Python已安装，再检查是否为3.11.9
    python --version | findstr /i "3.11.9" >nul
    if %errorlevel% equ 0 (
        echo Python 3.11.9 is already installed.
    ) else (
        echo Python version is not 3.11.9. Proceeding with installation...
        goto :install_python
    )
) else (
    echo Python is not installed. Proceeding with installation...
    goto :install_python
)

goto :python_checked


:install_python
echo Python 3.11.9 not detected. Starting download and installation...
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
set "PYTHON_EXE=python-3.11.9-amd64.exe"

curl -o "%TARGET_DIR%\%PYTHON_EXE%" "%PYTHON_URL%" || (
    echo Error: Failed to download Python installer. Please manually download and place it in %TARGET_DIR%.
    pause
    exit /b 1
)

echo Installing Python 3.11.9...
"%TARGET_DIR%\%PYTHON_EXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 || (
    echo Error: Python installation failed. Please run %PYTHON_EXE% manually.
    pause
    exit /b 1
)

del "%TARGET_DIR%\%PYTHON_EXE%"
echo Python 3.11.9 installation completed.

:python_checked
echo Python check completed.


:: 4. 创建并激活虚拟环境

echo 刷新PATH并检查Python版本...
set "PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311;%PATH%"  :: 手动添加3.11.9路径

if not exist "env" (
    echo Creating virtual environment...
    python -m venv env || (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call env\Scripts\activate.bat || (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 5. 安装依赖库
pip install -r "%REQUIREMENTS%" || (
    echo Error: Failed to install dependencies.
    pause
    exit /b 1
)

:: 6. 启动程序
echo.
echo All environment setup completed. Starting the program...
start "" "env\Scripts\uvicorn.exe" app:app --host 0.0.0.0 --reload

pause