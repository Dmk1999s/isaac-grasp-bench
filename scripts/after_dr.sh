#!/usr/bin/env bash
# DR 학습이 끝나면 평가와 분석까지 자동으로 이어서 한다.
#
# 왜: 사람이 확인해서 다음을 띄우는 구조면 그 사이에 GPU 유휴가 생긴다.
#     실제로 학습이 12:37 에 끝난 걸 14:27 에 알아차려 GPU 를 두 시간 놀린 적이 있다.
#
# ⚠ 기다릴 때 pgrep 패턴을 쓰지 않는다. `pgrep -f <패턴>` 은 명령줄 전체를 보므로
#   패턴이 감시 프로세스 자신의 명령줄에도 들어 있어 **자기 자신을 매칭**한다.
#   그 버그로 감시 루프 두 개가 영원히 안 끝났다. PID 로 기다리면 원리적으로 불가능하다.
#
# 사용: after_dr.sh <기다릴_PID>
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh

WAIT_PID="${1:?기다릴 PID}"
echo "[after] PID $WAIT_PID 종료 대기 시작 $(date +%H:%M)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[after] 학습 종료 확인 $(date +%H:%M)"

echo "[after] === DR 평가 시작 $(date +%H:%M) ==="
bash scripts/run_dr_eval.sh 2>&1 | sed 's/^/[eval] /'

echo "[after] === 분석 $(date +%H:%M) ==="
"$PY" scripts/analyze_dr.py 2>&1 | sed 's/^/[분석] /'
"$PY" scripts/make_figures.py 2>&1 | sed 's/^/[그림] /'
"$PY" scripts/check_criteria.py 2>&1 | sed 's/^/[기준] /'

echo "[after] 전체 완료 $(date +%H:%M)"
echo "[after] ⚠ 인스턴스는 자동으로 끄지 않는다 — 콘솔에서 Stop 하라."
echo "[after]   (shutdown -h 는 계정 설정에 따라 terminate 가 될 수 있어 쓰지 않는다)"
