#!/usr/bin/env bash
# 랜덤화 강도별 정책을 '폭 스윕'과 같은 축에서 평가한다.
#
# Phase 3 의 질문: 도메인 랜덤화가 RL 의 실패 구간(폭 70.9mm 이상)을 회복시키는가?
# 회복 여부를 보려면 실패가 관측된 그 축 위에서 재야 한다.
# 폭 스윕(run_width_sweep.sh)과 같은 스케일·같은 반복수·같은 env 수를 쓴다 —
# 다르면 '회복'이 아니라 '측정 조건 차이'가 섞인다(원칙 1).
#
# ⚠ 커버리지 축소 기록(원칙 5): 폭 6단계 중 5단계만 쓴다.
#    86.6mm 를 뺐다 — 78.8mm 와 94.5mm 사이라 곡선 모양보다 비용이 크다.
#    남긴 52.5mm 는 대조군이다: 랜덤화가 쉬운 조건을 해치지 않는지 본다.
#    3강도 x 3시드 x 5폭 = 45회 (약 40분).
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh   # EULA·PY 등 실행 환경 고정

REPEATS=${REPEATS:-5}
ENVS=${ENVS:-32}
STRENGTHS=${STRENGTHS:-"000 050 100"}
SEEDS=${SEEDS:-"1 2 3"}
# run_width_sweep.sh 와 같은 값이어야 비교가 성립한다.
SCALES=${SCALES:-"1.0 1.2 1.35 1.5 1.8"}

for ST in $STRENGTHS; do
  for S in $SEEDS; do
    CKPT="checkpoints/final/policy_dr${ST}_seed${S}.pt"
    if [ ! -f "$CKPT" ]; then
      echo "[dr$ST|seed$S] 체크포인트 없음 — 건너뜀 (학습이 아직 안 끝났다)"
      continue
    fi
    for W in $SCALES; do
      TAG=$(echo "$W" | tr -d '.')
      OUT="eval/dr${ST}_w${TAG}_s${S}.csv"
      [ -f "$OUT" ] && { echo "[dr$ST|w$W|s$S] 이미 있음"; continue; }
      R=$(timeout 1800 "$PY" -u scripts/run_eval.py --controller rl --headless \
          --object_scale "0.8,${W},0.8" --num_envs $ENVS --repeats $REPEATS \
          --checkpoint "$CKPT" --out "$OUT" 2>&1 | grep -E "^\[rl\] 성공률" | head -1)
      echo "[dr$ST|w$W|s$S] $R"
    done
  done
done
echo "전체 완료 $(date +%H:%M)"
