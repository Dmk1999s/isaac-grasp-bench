#!/usr/bin/env bash
# 여러 시드의 RL 정책을 같은 물체 격자에서 평가한다.
#
# 왜: 시드 하나로 잰 실패 지도는 '그 시드가 유독 나빴을' 가능성을 배제하지 못한다.
# 학습 편차를 측정해야 "RL 이 이 조건에서 무너진다"를 주장할 수 있다(원칙 3).
#
# 멱등하다 — 결과 CSV 가 이미 있으면 건너뛴다.
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh   # EULA·PY 등 실행 환경 고정
SEEDS=${SEEDS:-"1 2 3 4"}
CELLS=${CELLS:-"base:0.8,0.8,0.8 elong15:1.2,0.8,0.8 elong20:1.6,0.8,0.8 wide12:0.8,1.2,0.8 wide16:0.8,1.6,0.8 small:0.5,0.5,0.5"}
REPEATS=${REPEATS:-5}
ENVS=${ENVS:-32}

for CELL in $CELLS; do
  LABEL="${CELL%%:*}"; SCALE="${CELL#*:}"
  for S in $SEEDS; do
    OUT="eval/gridseed_${LABEL}_rl_s${S}.csv"
    [ -f "$OUT" ] && { echo "[$LABEL|seed$S] 이미 있음 — 건너뜀"; continue; }
    CKPT="checkpoints/final/policy_ikrel_seed${S}.pt"
    [ -f "$CKPT" ] || { echo "[$LABEL|seed$S] ERROR: 체크포인트 없음"; continue; }
    R=$(timeout 1800 "$PY" -u scripts/run_eval.py --controller rl --headless \
        --object_scale "$SCALE" --num_envs $ENVS --repeats $REPEATS \
        --checkpoint "$CKPT" --out "$OUT" 2>&1 | grep -E "^\[rl\] 성공률" | head -1)
    echo "[$LABEL|seed$S] $R"
  done
done
echo "전체 완료 $(date +%H:%M)"
