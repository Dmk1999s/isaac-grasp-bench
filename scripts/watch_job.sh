#!/usr/bin/env bash
# 장시간 작업 감시기 — 진척 / 완료 / 비정상 종료 / 교착을 모두 이벤트로 내보낸다.
#
# 왜 필요한가:
#   setsid 로 분리해 띄우면 SIGTERM 전파는 막지만 완료 알림을 잃는다.
#   실제로 학습이 12:37 에 끝난 걸 48분 뒤에야 알아차려 GPU 를 놀렸다.
#   반대로 '프로세스가 살아있다'만 보면 종료 핸들러 교착(17분간 학습 0)을 못 잡는다.
#   그래서 세 가지를 다 본다: 로그 진척, 프로세스 생존, GPU 사용률.
#
# 사용: watch_job.sh <로그파일> <프로세스패턴> <완료마커> [교착초]
#
# stdout 한 줄 = 알림 하나. 종료하면 감시가 끝난다.
set -u
LOG="${1:?로그파일}"
PAT="${2:?프로세스 패턴}"
DONE_MARK="${3:?완료 마커}"
STALL=${4:-600}

# 우리가 알림으로 올릴 줄: 진척 표시, 완료, 그리고 실패 신호 전부.
# ⚠ 성공 신호만 걸면 크래시났을 때 '조용함'과 구분이 안 된다.
FILTER='^\[|전체 완료|ERROR|Error|Traceback|FAILED|Killed|OOM|CUDA|assert'

prev_n=0
last_change=$(date +%s)
stall_reported=0

while true; do
  n=$(grep -cE "$FILTER" "$LOG" 2>/dev/null || echo 0)
  if [ "$n" -gt "$prev_n" ]; then
    grep -E "$FILTER" "$LOG" 2>/dev/null | tail -n $(( n - prev_n ))
    prev_n=$n
    last_change=$(date +%s)
    stall_reported=0
  fi

  if grep -q "$DONE_MARK" "$LOG" 2>/dev/null; then
    echo "완료: $DONE_MARK 확인됨 ($(date +%H:%M))"
    exit 0
  fi

  if ! pgrep -f "$PAT" >/dev/null 2>&1; then
    echo "비정상종료: '$PAT' 프로세스가 사라졌는데 완료 마커가 없다 ($(date +%H:%M))"
    exit 1
  fi

  now=$(date +%s); idle=$(( now - last_change ))
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -dc '0-9')
  gpu=${gpu:-0}
  if [ "$idle" -gt "$STALL" ] && [ "$gpu" -lt 5 ] && [ "$stall_reported" -eq 0 ]; then
    echo "교착의심: ${idle}초간 로그 진척 없음, GPU ${gpu}% — py-spy 로 스택 확인 필요"
    stall_reported=1
    last_change=$now   # 같은 교착으로 반복 알림하지 않는다
  fi

  sleep 30
done
