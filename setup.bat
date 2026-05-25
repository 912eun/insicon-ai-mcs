@echo off
REM ai-mcs 개발 환경 세팅 스크립트 (Windows x86_64)
REM macOS/Linux는 setup.sh 참고

echo === ai-mcs 환경 세팅 시작 ===

REM 1. submodule 초기화 (PySCFabSim)
echo [1/4] PySCFabSim submodule 초기화...
git submodule update --init --recursive

if not exist "external\PySCFabSim\requirements.txt" (
    echo ERROR: external\PySCFabSim submodule이 비어있습니다.
    echo   먼저 아래 명령어로 submodule을 등록해주세요:
    echo   git submodule add https://github.com/prosysscience/PySCFabSim-release external/PySCFabSim
    exit /b 1
)

REM 2. conda 환경 생성
echo [2/4] conda 환경 생성 (Python 3.9.13)...
conda env list | findstr /B "ai-mcs " >nul 2>&1
if %errorlevel% == 0 (
    echo   기존 ai-mcs 환경 삭제...
    conda env remove -n ai-mcs -y
)
conda create -n ai-mcs python=3.9.13 pip -y

REM 3. pip 다운그레이드 후 gym==0.19.0 설치
REM gym==0.19.0은 opencv-python>=3. 스펙이 불완전해 pip 24.1+가 거부함
REM pip 21.3.1을 끝까지 유지 (복원 시 gym 캐시 메타데이터 검증 실패)
echo [3/4] gym==0.19.0 설치...
conda run -n ai-mcs pip install "pip==21.3.1" "setuptools<65" "wheel==0.37.0" --quiet
conda run -n ai-mcs pip install gym==0.19.0 --no-build-isolation --quiet

REM 4. 나머지 패키지 설치 (Windows x86_64는 PyPI wheel 사용)
REM numpy==1.21.4 명시: plotly 최신버전이 narwhals를 끌어들여 numpy2를 설치할 수 있음
echo [4/4] 패키지 설치...
conda run -n ai-mcs pip install ^
  "numpy==1.21.4" ^
  absl-py==1.0.0 ^
  cachetools==4.2.4 ^
  certifi==2021.10.8 ^
  charset-normalizer==2.0.9 ^
  click==8.0.3 ^
  cloudpickle==1.6.0 ^
  configparser==5.2.0 ^
  cycler==0.11.0 ^
  docker-pycreds==0.4.0 ^
  fonttools==4.28.4 ^
  gitdb==4.0.9 ^
  GitPython==3.1.24 ^
  google-auth==2.3.3 ^
  google-auth-oauthlib==0.4.6 ^
  idna==3.3 ^
  importlib-metadata==4.9.0 ^
  kiwisolver==1.3.2 ^
  Markdown==3.3.6 ^
  matplotlib==3.5.1 ^
  oauthlib==3.1.1 ^
  opencv-python==4.5.4.60 ^
  packaging==21.3 ^
  pandas==1.3.5 ^
  pathtools==0.1.2 ^
  Pillow==8.4.0 ^
  promise==2.3 ^
  protobuf==3.19.1 ^
  psutil==5.8.0 ^
  pyasn1==0.4.8 ^
  pyasn1-modules==0.2.8 ^
  pyparsing==3.0.6 ^
  python-dateutil==2.8.2 ^
  pytz==2021.3 ^
  PyYAML==6.0 ^
  requests==2.26.0 ^
  requests-oauthlib==1.3.0 ^
  rsa==4.8 ^
  sentry-sdk==1.5.1 ^
  shortuuid==1.0.8 ^
  six==1.16.0 ^
  smmap==5.0.0 ^
  stable-baselines3==1.3.0 ^
  tensorboard==2.7.0 ^
  termcolor==1.1.0 ^
  torch==1.10.1 ^
  typing-extensions==4.0.1 ^
  urllib3==1.26.7 ^
  wandb==0.12.9 ^
  Werkzeug==2.0.2 ^
  yaspin==2.1.0 ^
  zipp==3.6.0 ^
  "pyinstrument~=4.1.1" ^
  rware ^
  pettingzoo ^
  xgboost ^
  "plotly<6"

echo.
echo === 완료! ===
echo.
echo 환경 활성화:
echo   conda activate ai-mcs
echo.
echo PySCFabSim 실행 테스트:
echo   cd external\PySCFabSim
echo   mkdir greedy
echo   python main.py --dataset SMT2020_HVLM --days 7 --dispatcher fifo --seed 0
