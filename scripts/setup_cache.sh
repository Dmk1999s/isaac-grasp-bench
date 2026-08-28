#!/usr/bin/env bash
# Omniverse / pip 캐시를 인스턴스 스토어(/opt/dlami/nvme)로 리다이렉트.
#
# ⚠ /opt/dlami/nvme 는 인스턴스 stop 시 전부 소실된다.
#   따라서 이 스크립트는 "재구축 스크립트"다. 인스턴스를 start 할 때마다 실행하라.
#   소실되는 것은 캐시뿐이며, 재실행하면 Isaac Sim 이 알아서 다시 채운다.
set -euo pipefail

SCRATCH="/opt/dlami/nvme"
CACHE_ROOT="$SCRATCH/ovcache"

if ! mountpoint -q "$SCRATCH"; then
  echo "[cache] ERROR: $SCRATCH 가 마운트되지 않았다. 인스턴스 스토어 확인 필요." >&2
  exit 1
fi

mkdir -p "$CACHE_ROOT"/{ov-cache,ov-data,ov-logs,nvidia-omniverse,pip,kit-cache}

# ~/.cache/ov, ~/.local/share/ov 등을 심볼릭 링크로 대체.
# 기존에 실디렉터리가 있으면 옮기지 않고 그대로 두고 경고만 한다(데이터 유실 방지).
link_dir() {  # link_dir <target-under-CACHE_ROOT> <link-path>
  local target="$CACHE_ROOT/$1" link="$2"
  mkdir -p "$target" "$(dirname "$link")"
  if [ -L "$link" ]; then
    ln -sfn "$target" "$link"
  elif [ -e "$link" ]; then
    echo "[cache] SKIP: $link 이 실제 디렉터리로 존재한다. 수동 확인 필요." >&2
    return 0
  else
    ln -s "$target" "$link"
  fi
  echo "[cache] $link -> $target"
}

link_dir ov-cache          "$HOME/.cache/ov"
link_dir kit-cache         "$HOME/.cache/Kit"
link_dir ov-data           "$HOME/.local/share/ov"
link_dir nvidia-omniverse  "$HOME/.nvidia-omniverse"
link_dir pip               "$HOME/.cache/pip"

# Kit 이 참조하는 환경변수도 같이 고정해 둔다.
ENVFILE="$HOME/.isaac_cache_env"
cat > "$ENVFILE" <<EOF
export OMNI_KIT_CACHE_PATH="$CACHE_ROOT/kit-cache"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
# 헤드리스 필수: 없으면 import isaacsim 이 대화형 EULA 프롬프트에서
# "Unable to bootstrap inner kit kernel: EOF when reading a line" 로 죽는다.
export OMNI_KIT_ACCEPT_EULA=YES
EOF
echo "[cache] 환경변수 파일: $ENVFILE  (source 해서 사용)"
echo "[cache] 완료. 루트 볼륨 사용량:"
df -h / "$SCRATCH" | sed 's/^/[cache]   /'
