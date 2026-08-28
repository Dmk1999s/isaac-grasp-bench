#!/usr/bin/env bash
# Phase 0 재현 스크립트 — isaac-grasp-bench
# 대상: AWS EC2 g5.2xlarge (A10G 24GB), Ubuntu 22.04, Deep Learning AMI
# 멱등(idempotent)하게 작성. 여러 번 실행해도 안전해야 한다.
set -euo pipefail

log() { echo "[setup] $*"; }

# ─────────────────────────────────────────────────────────────
# Step 1. 영속 데이터 볼륨 (/dev/nvme1n1, 100GB EBS) → /data
#   실행 완료: 2026-08-28
#   주의: 디바이스명(nvme1n1)은 재부팅/인스턴스 재생성 시 바뀔 수 있다.
#         새 인스턴스에서는 아래 DATA_DEV를 lsblk로 확인 후 지정할 것.
# ─────────────────────────────────────────────────────────────
DATA_DEV="${DATA_DEV:-/dev/nvme1n1}"
DATA_MNT="/data"

setup_data_volume() {
  if mountpoint -q "$DATA_MNT"; then
    log "$DATA_MNT 이미 마운트됨 — 건너뜀"
    return 0
  fi

  # 안전장치: 기존 파일시스템이 있으면 포맷하지 않는다
  local existing
  existing="$(sudo blkid -s TYPE -o value "$DATA_DEV" 2>/dev/null || true)"
  if [ -n "$existing" ]; then
    log "$DATA_DEV 에 이미 '$existing' 파일시스템 존재 — 포맷 생략, 마운트만 수행"
  else
    log "$DATA_DEV 포맷 (ext4, label=data, reserve 1%)"
    sudo mkfs.ext4 -L data -m 1 "$DATA_DEV"
  fi

  local uuid
  uuid="$(sudo blkid -s UUID -o value "$DATA_DEV")"
  log "UUID=$uuid"

  sudo mkdir -p "$DATA_MNT"
  # fstab 은 반드시 UUID 로. nofail: 볼륨 미부착 시 부팅 실패 방지
  if ! grep -q "$uuid" /etc/fstab; then
    echo "UUID=$uuid  $DATA_MNT  ext4  defaults,noatime,nofail  0  2" | sudo tee -a /etc/fstab
  fi
  sudo systemctl daemon-reload
  sudo mount -a
  sudo chown -R "$(id -u):$(id -g)" "$DATA_MNT"
  df -hT "$DATA_MNT"
}


# ─────────────────────────────────────────────────────────────
# Step 2. Python 3.11 (Isaac Sim 5.X 요구사항)
#   Ubuntu 22.04 universe 의 python3.11 은 3.11.0~rc1 (릴리스 후보)라 쓰면 안 된다.
#   deadsnakes PPA 에서 안정판(3.11.15)을 받는다. 시스템 기본 python3(3.10)는 건드리지 않는다.
# ─────────────────────────────────────────────────────────────
VENV="${VENV:-$HOME/env_isaaclab}"

# ─────────────────────────────────────────────────────────────
# Step 2a. 시스템 라이브러리
#   Deep Learning AMI 에는 GUI 계열 라이브러리가 빠져 있다.
#   특히 libGLU.so.1 이 없으면 Isaac Sim 이 죽지 않고 머티리얼 생성 단계에서
#   CPU 를 태우며 무한정 매달린다(에러 로그 한 줄만 남고 진행이 멈춘다).
#   증상: rtx.neuraylib.plugin "Failed to open ... libneuray.so: libGLU.so.1"
# ─────────────────────────────────────────────────────────────
setup_system_deps() {
  log "Isaac Sim 헤드리스용 시스템 라이브러리 설치"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 libxss1 \
    libsm6 libice6 libxt6 libfontconfig1 libxkbcommon-x11-0 libglib2.0-0
}

setup_python311() {
  if command -v python3.11 >/dev/null 2>&1 && ! python3.11 --version | grep -q "rc"; then
    log "python3.11 이미 설치됨: $(python3.11 --version)"
  else
    log "deadsnakes PPA 추가 + python3.11 설치"
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3.11-venv python3.11-dev
  fi

  if [ ! -x "$VENV/bin/python" ]; then
    log "venv 생성: $VENV"
    python3.11 -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --upgrade pip
  log "venv python: $(python --version)"
}

# ─────────────────────────────────────────────────────────────
# Step 3. 캐시 리다이렉트 (인스턴스 스토어)
#   stop 하면 소실되므로 start 할 때마다 재실행해야 한다.
# ─────────────────────────────────────────────────────────────
setup_caches() {
  "$(dirname "${BASH_SOURCE[0]}")/setup_cache.sh"
  # shellcheck disable=SC1091
  source "$HOME/.isaac_cache_env"
}

# ─────────────────────────────────────────────────────────────
# Step 4. Isaac Sim 5.1.0 (pip) + PyTorch
#   버전 조합은 Isaac Lab v2.3.2 공식 pip 설치 문서 기준:
#     - Isaac Sim 5.X  -> Python 3.11
#     - Linux x86_64   -> torch 2.7.0 / torchvision 0.22.0 (cu128)
#   설치 순서도 문서와 동일하게 isaacsim -> torch (-U) 순서를 지킨다.
# ─────────────────────────────────────────────────────────────
ISAACSIM_VERSION="5.1.0"

setup_isaacsim() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  if python -c "import isaacsim" 2>/dev/null; then
    log "isaacsim 이미 설치됨 — 건너뜀"
  else
    log "isaacsim[all,extscache]==$ISAACSIM_VERSION 설치 (수 GB 다운로드)"
    pip install "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com
  fi
  log "CUDA 지원 PyTorch 설치 (cu128)"
  pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
}

# ─────────────────────────────────────────────────────────────
# Step 5. Isaac Lab v2.3.2 (소스)
#   v2.3.X <-> Isaac Sim 4.5/5.0/5.1 호환. 3.0.0-beta 는 의도적으로 피한다.
# ─────────────────────────────────────────────────────────────
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
ISAACLAB_TAG="v2.3.2"

setup_isaaclab() {
  if [ ! -d "$ISAACLAB_DIR" ]; then
    log "Isaac Lab $ISAACLAB_TAG 클론"
    git clone --branch "$ISAACLAB_TAG" --depth 1 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  # EULA: 헤드리스에서는 stdin 이 없어 대화형 프롬프트가 즉시 EOF 로 죽는다.
  export OMNI_KIT_ACCEPT_EULA=YES

  # flatdict==4.0.1 은 휠이 없어 sdist 를 빌드하는데, pip 빌드격리가 최신
  # setuptools(pkg_resources 제거됨)를 쓰는 바람에 빌드가 깨진다.
  # 그 결과 isaaclab.sh --install 이 exit 0 을 반환하면서도 핵심 isaaclab
  # 패키지만 조용히 빠진다. 먼저 빌드격리 없이 설치해 이 경로를 막는다.
  log "flatdict==4.0.1 선설치 (빌드격리 없이)"
  pip install --no-build-isolation "flatdict==4.0.1"

  log "Isaac Lab 확장 설치 (editable)"
  "$ISAACLAB_DIR/isaaclab.sh" --install

  # isaacsim 이 명시 고정하는 버전들. 위 설치 과정에서 밀려 올라가므로 되돌린다.
  log "isaacsim 명시 핀 정렬"
  pip install "packaging==23.0" "click==8.1.7" "psutil==5.9.8" \
              "typing_extensions==4.12.2"
  pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

  # 검증: 조용한 실패를 여기서 잡는다.
  if ! pip show isaaclab >/dev/null 2>&1; then
    log "ERROR: 핵심 isaaclab 패키지가 설치되지 않았다."
    return 1
  fi
  log "isaaclab $(pip show isaaclab | awk '/^Version:/{print $2}') 설치 확인"
}

# ─────────────────────────────────────────────────────────────
# Step 6. 헤드리스 스모크 테스트 — GPU 가 실제로 도는지 확인
#   Isaac-Lift-Cube-Franka-v0 를 아주 짧게(기본 5 iteration) 학습시킨다.
#   목적은 성능 측정이 아니라 "헤드리스로 시뮬레이터가 뜨고 GPU 가 돈다"의 확인이다.
#   여기서 나오는 보상/성공률 숫자는 베이스라인이 아니다. 절대 지표로 인용하지 마라.
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Step 5a. 학습 산출물을 /data 로
#   rsl_rl 은 기본적으로 $ISAACLAB_DIR/logs 에 체크포인트를 쌓는다.
#   거기는 루트 볼륨이라 원칙(체크포인트·로그는 /data)에 어긋난다.
# ─────────────────────────────────────────────────────────────
setup_log_redirect() {
  local runs="$DATA_MNT/isaac-grasp-bench/runs"
  mkdir -p "$runs"
  if [ -L "$ISAACLAB_DIR/logs" ]; then
    log "$ISAACLAB_DIR/logs 이미 링크됨 — 건너뜀"
    return 0
  fi
  if [ -d "$ISAACLAB_DIR/logs" ]; then
    cp -a "$ISAACLAB_DIR/logs/." "$runs/" 2>/dev/null || true
    rm -rf "$ISAACLAB_DIR/logs"
  fi
  ln -s "$runs" "$ISAACLAB_DIR/logs"
  log "$ISAACLAB_DIR/logs -> $runs"
}

SMOKE_TASK="${SMOKE_TASK:-Isaac-Lift-Cube-Franka-v0}"
SMOKE_ENVS="${SMOKE_ENVS:-32}"
SMOKE_ITERS="${SMOKE_ITERS:-5}"

smoke_test() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  export OMNI_KIT_ACCEPT_EULA=YES
  local logf="$DATA_MNT/isaac-grasp-bench/logs/smoke_train.log"
  log "헤드리스 스모크 학습: $SMOKE_TASK (envs=$SMOKE_ENVS, iters=$SMOKE_ITERS)"
  if python "$ISAACLAB_DIR/scripts/reinforcement_learning/rsl_rl/train.py" \
       --task "$SMOKE_TASK" --headless \
       --num_envs "$SMOKE_ENVS" --max_iterations "$SMOKE_ITERS" > "$logf" 2>&1; then
    log "스모크 테스트 통과. 로그: $logf"
  else
    log "스모크 테스트 실패(exit=$?). 로그 확인: $logf"
    return 1
  fi
}

main() {
  setup_data_volume
  mkdir -p "$DATA_MNT/isaac-grasp-bench"/{scripts,logs,checkpoints,eval}
  setup_python311
  setup_system_deps
  setup_caches
  setup_isaacsim
  setup_isaaclab
  setup_log_redirect
  smoke_test
  log "Step 1~6 완료."
}

main "$@"
