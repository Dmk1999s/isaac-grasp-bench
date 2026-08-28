#!/usr/bin/env bash
# 닫힘축 폭을 촘촘히 훑어 '실패가 시작되는 지점'을 관측에서 찾는다.
#
# 왜: 6셀 격자는 성기다. wide12(63mm) 에서 RL 81%, wide16(84mm) 에서 46% 였는데
#     그 사이 어디서 꺾이는지 모른다. 임계값을 내가 정하지 않고
#     관측에서 도출하려면(원칙 4) 더 촘촘한 표본이 필요하다.
#
# 폭 = 52.5mm x scale. Franka 개방 한계는 80mm.
# 멱등하다 — 결과 CSV 가 있으면 건너뛴다.
set -u
cd "$(dirname "$0")/.."
REPEATS=${REPEATS:-5}
ENVS=${ENVS:-32}
SEEDS=${SEEDS:-"1 2 3 4"}
# y 스케일 -> 폭(mm): 1.0=52.5 1.2=63 1.35=70.9 1.5=78.8 1.65=86.6 1.8=94.5
SCALES=${SCALES:-"1.0 1.2 1.35 1.5 1.65 1.8"}

for W in $SCALES; do
  TAG=$(echo "$W" | tr -d '.')
  SCALE="0.8,${W},0.8"

  OUT="eval/width_w${TAG}_pca.csv"
  if [ -f "$OUT" ]; then echo "[w$W|pca] 이미 있음"; else
    R=$(timeout 1800 python -u scripts/run_eval.py --controller pca --headless \
        --object_scale "$SCALE" --num_envs $ENVS --repeats $REPEATS \
        --out "$OUT" 2>&1 | grep -E "^\[pca\] 성공률" | head -1)
    echo "[w$W|pca] $R"
  fi

  for S in $SEEDS; do
    OUT="eval/width_w${TAG}_rl_s${S}.csv"
    [ -f "$OUT" ] && { echo "[w$W|seed$S] 이미 있음"; continue; }
    CKPT="checkpoints/final/policy_ikrel_seed${S}.pt"
    [ -f "$CKPT" ] || { echo "[w$W|seed$S] ERROR: 체크포인트 없음"; continue; }
    R=$(timeout 1800 python -u scripts/run_eval.py --controller rl --headless \
        --object_scale "$SCALE" --num_envs $ENVS --repeats $REPEATS \
        --checkpoint "$CKPT" --out "$OUT" 2>&1 | grep -E "^\[rl\] 성공률" | head -1)
    echo "[w$W|seed$S] $R"
  done
done
echo "전체 완료 $(date +%H:%M)"
