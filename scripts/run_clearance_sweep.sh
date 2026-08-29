#!/usr/bin/env bash
# 바닥 클리어런스를 훑어 '값을 어떻게 골라야 하는지'를 관측에서 정한다.
#
# 왜: 크기 스윕에서 26.2mm 큐브가 clr=30 에서 1.9%, clr=10 에서 100.0% 였다.
#     그런데 "10 으로 하니 100% 나왔다"는 이 프로젝트가 경계하는 방식이다 —
#     Phase 1 에서 grasp_depth 를 고를 때 "성공률을 보고 고른 값이 아니다"라고
#     명시했다. 같은 기준을 여기에도 적용해야 한다.
#
# FLOOR_CLEARANCE_MM 은 손끝이 테이블을 긁지 않게 하는 가드다. 그래서 물리적 조건은
#     clr < 물체 높이   (아니면 손끝이 물체 윗면 위에 선다)
# 이고, Phase 1 의 `depth > clearance` 와 같은 형태다.
# 이 스윕은 그 조건이 성공률에서 실제로 어디서 꺾이는지를 보여주기 위한 것이지,
# 성공률이 제일 높은 값을 고르기 위한 것이 아니다.
#
# 물체는 26.2mm(scale 0.5) 고정 — 판정이 뒤집힌 바로 그 셀이다.
# 멱등하다 — 결과 CSV 가 있으면 건너뛴다. clr=10, 30 은 크기 스윕에서 이미 쟀다.
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh
REPEATS=${REPEATS:-5}
ENVS=${ENVS:-32}
SCALE=${SCALE:-"0.5,0.5,0.5"}     # 26.2mm 큐브
CLEARANCES=${CLEARANCES:-"5 10 15 20 25 30"}

for C in $CLEARANCES; do
  OUT="eval/size_s05_pca_clr${C}.csv"
  if [ -f "$OUT" ]; then echo "[clr$C] 이미 있음"; continue; fi
  R=$(timeout 1800 "$PY" -u scripts/run_eval.py --controller pca --headless \
      --object_scale "$SCALE" --num_envs $ENVS --repeats $REPEATS \
      --floor_clearance_mm "$C" --out "$OUT" 2>&1 | grep -E "^\[pca\] 성공률" | head -1)
  echo "[clr$C] $R"
done
echo "클리어런스 스윕 완료 $(date +%H:%M)"
