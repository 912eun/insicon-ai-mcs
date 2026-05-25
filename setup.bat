@echo off
REM ai-mcs 개발 환경 세팅 스크립트 (Windows)

echo === ai-mcs 환경 세팅 시작 ===

REM 1. submodule 초기화 (PySCFabSim)
echo [1/3] PySCFabSim submodule 초기화...
git submodule update --init --recursive

if not exist "external\PySCFabSim\requirements.txt" (
    echo ERROR: external\PySCFabSim submodule이 비어있습니다.
    echo   먼저 아래 명령어로 submodule을 등록해주세요:
    echo   git submodule add https://github.com/prosysscience/PySCFabSim-release external/PySCFabSim
    exit /b 1
)

REM 2. conda 환경 생성
echo [2/3] conda 환경 생성 (Python 3.9.13)...
conda env list | findstr "ai-mcs" >nul 2>&1
if %errorlevel% == 0 (
    echo   기존 ai-mcs 환경 업데이트...
    conda env update -f environment.yml --prune
) else (
    echo   ai-mcs 환경 새로 생성...
    conda env create -f environment.yml
)

REM 3. 완료 안내
echo [3/3] 완료!
echo.
echo 환경 활성화:
echo   conda activate ai-mcs
echo.
echo PySCFabSim 실행 테스트:
echo   cd external\PySCFabSim ^&^& python main.py
