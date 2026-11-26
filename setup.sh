#!/bin/bash
# ai-mcs 개발 환경 세팅 스크립트 (macOS / Linux)
# Windows는 setup.bat 참고

set -e

echo "=== ai-mcs 환경 세팅 시작 ==="

# 1. submodule 초기화 (PySCFabSim)
echo "[1/5] PySCFabSim submodule 초기화..."
git submodule update --init --recursive

if [ ! -d "external/PySCFabSim" ] || [ -z "$(ls -A external/PySCFabSim)" ]; then
  echo "ERROR: external/PySCFabSim submodule이 비어있습니다."
  echo "  git submodule add https://github.com/prosysscience/PySCFabSim-release external/PySCFabSim"
  exit 1
fi

# 2. conda 환경 생성 (Python만)
echo "[2/5] conda 환경 생성 (Python 3.9.13)..."
if conda env list | grep -q "^ai-mcs "; then
  echo "  기존 ai-mcs 환경 발견 — 삭제 후 재생성..."
  conda env remove -n ai-mcs -y
fi
conda create -n ai-mcs python=3.9.13 pip -y

# 3. 컴파일 필요 패키지 → conda-forge (ARM64 포함 네이티브 바이너리)
# numpy/pandas/torch/opencv 등은 구버전 PyPI wheel이 ARM64를 지원하지 않아 conda 사용
echo "[3/5] conda 바이너리 패키지 설치 (numpy, pandas, torch, opencv 등)..."
conda install -n ai-mcs -c pytorch -c conda-forge -y \
  "numpy>=1.21,<1.22" \
  "pandas>=1.3,<1.4" \
  "matplotlib>=3.5,<3.6" \
  "pytorch::pytorch>=1.10" \
  "opencv"
# pillow는 conda-forge 단독 설치: pkgs/main 버전은 libtiff.5.dylib 참조하나
# conda-forge libtiff 4.5.0은 libtiff.6.dylib 제공 → 채널 혼용 시 dlopen 실패
conda install -n ai-mcs -c conda-forge pillow -y

# 4. pip/setuptools/wheel 다운그레이드 후 gym==0.19.0 설치
# gym==0.19.0은 opencv-python>=3. 스펙이 불완전해서 신버전 pip/wheel이 거부함
echo "[4/5] gym==0.19.0 설치..."
# pip 21.3.1로 다운그레이드: gym의 불완전한 메타데이터(opencv-python>=3.)를 허용
# pip 24.1+는 이 메타데이터를 캐시 및 설치 단계 모두에서 거부하므로 복원하지 않음
conda run -n ai-mcs pip install "pip==21.3.1" "setuptools<65" "wheel==0.37.0" --quiet
conda run -n ai-mcs pip install gym==0.19.0 --no-build-isolation --quiet

# 5. 나머지 순수 Python 패키지 설치
echo "[5/5] pip 패키지 설치..."
# pip 21.3.1로 설치: gym의 buggy 메타데이터를 무시하고 이미 설치된 패키지를 존중
# plotly<6: plotly 6.x → narwhals → numpy>=2.0 요구로 conda numpy 1.x 충돌
conda run -n ai-mcs pip install \
  "numpy<2" \
  absl-py==1.0.0 \
  cachetools==4.2.4 \
  certifi==2021.10.8 \
  charset-normalizer==2.0.9 \
  click==8.0.3 \
  cloudpickle==1.6.0 \
  configparser==5.2.0 \
  docker-pycreds==0.4.0 \
  gitdb==4.0.9 \
  GitPython==3.1.24 \
  google-auth==2.3.3 \
  google-auth-oauthlib==0.4.6 \
  idna==3.3 \
  importlib-metadata==4.9.0 \
  Markdown==3.3.6 \
  oauthlib==3.1.1 \
  packaging==21.3 \
  pathtools==0.1.2 \
  promise==2.3 \
  protobuf==3.19.1 \
  psutil==5.8.0 \
  pyasn1==0.4.8 \
  pyasn1-modules==0.2.8 \
  pyparsing==3.0.6 \
  python-dateutil==2.8.2 \
  pytz==2021.3 \
  PyYAML==6.0 \
  requests==2.26.0 \
  requests-oauthlib==1.3.0 \
  rsa==4.8 \
  sentry-sdk==1.5.1 \
  shortuuid==1.0.8 \
  six==1.16.0 \
  smmap==5.0.0 \
  stable-baselines3==1.3.0 \
  tensorboard==2.7.0 \
  termcolor==1.1.0 \
  typing-extensions==4.0.1 \
  urllib3==1.26.7 \
  wandb==0.12.9 \
  Werkzeug==2.0.2 \
  yaspin==2.1.0 \
  zipp==3.6.0 \
  "pyinstrument~=4.1.1" \
  rware \
  pettingzoo \
  xgboost \
  "plotly<6"

echo ""
echo "=== 완료! ==="
echo ""
echo "환경 활성화:"
echo "  conda activate ai-mcs"
echo ""
echo "PySCFabSim 실행 테스트:"
echo "  cd external/PySCFabSim && python main.py"
