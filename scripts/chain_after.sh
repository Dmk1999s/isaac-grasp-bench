#!/usr/bin/env bash
# 앞 작업이 끝나면 뒤 작업을 자동으로 잇는다 — GPU 유휴 시간을 없앤다.
#
# 왜: 학습이 12:37 에 끝난 걸 48분 뒤에야 알아차려 GPU 를 놀린 적이 있다.
#     사람이 확인해서 다음을 띄우는 구조면 그 공백이 반드시 생긴다.
#
# 사용: chain_after.sh <기다릴_프로세스_패턴> <다음에_실행할_명령...>
set -u
WAIT_PAT="${1:?기다릴 프로세스 패턴}"; shift
echo "[chain] '$WAIT_PAT' 종료 대기 시작 $(date +%H:%M)"
while pgrep -f "$WAIT_PAT" >/dev/null 2>&1; do sleep 20; done
echo "[chain] '$WAIT_PAT' 종료 확인 $(date +%H:%M) — 다음 작업 시작: $*"
exec "$@"
