#!/usr/bin/env bash
# 물체 격자 실패 지도 — 두 제어 방식을 같은 격자 위에서 훑는다.
#
# 기본 Lift-Cube 조건에서는 양쪽 다 성공률 100% 로 천장에 붙어 변별력이 없었다.
# 그래서 물체 형상을 바꿔가며 어디서 갈라지는지 본다.
#
# 축:
#   종횡비 — x 를 늘려 주축을 뚜렷하게 만든다. PCA 가 유리해질 조건.
#   폭     — y(닫힘축)를 늘려 그리퍼 한계(80mm)에 접근시킨다.
#   크기   — 전체를 줄여 작은 물체를 만든다.
#
# ⚠ 커버리지 축소 기록(원칙 5): 셀당 5회 반복 x 32 env = 160 에피소드.
#    기본 조건 측정의 10회 반복보다 적다. 격자 탐색용이다.
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh   # EULA·PY 등 실행 환경 고정
CKPT=checkpoints/final/policy_ikrel_seed1.pt
REPEATS=${REPEATS:-5}
ENVS=${ENVS:-32}
OUT=eval/grid_summary.txt
: > $OUT

run_cell() {  # run_cell <label> <scale>
  local label="$1" scale="$2"
  for C in pca rl; do
    local extra=""
    [ "$C" = "rl" ] && extra="--checkpoint $CKPT"
    local line
    line=$(timeout 1800 python -u scripts/run_eval.py --controller $C --headless \
        --object_scale "$scale" --num_envs $ENVS --repeats $REPEATS \
        $extra --out "eval/grid_${label}_${C}.csv" 2>&1 \
        | grep -E "^\[$C\] 성공률|^\[$C\] 사이클|^\[물체\]|경고" | tr '\n' ' ')
    echo "[$label|$scale|$C] $line" | tee -a $OUT
  done
}

run_cell base      "0.8,0.8,0.8"    # 42 x 42 x 42  (기준)
run_cell elong15   "1.2,0.8,0.8"    # 63 x 42 x 42  (주축 뚜렷)
run_cell elong20   "1.6,0.8,0.8"    # 84 x 42 x 42
run_cell wide12    "0.8,1.2,0.8"    # 42 x 63 x 42  (닫힘축 넓음)
run_cell wide16    "0.8,1.6,0.8"    # 42 x 84 x 42  (그리퍼 한계 80mm 초과)
run_cell small     "0.5,0.5,0.5"    # 26 x 26 x 26  (작은 물체)

echo "완료. 요약: $OUT"
