#!/usr/bin/env bash
# PCA 파지 깊이(grasp_depth_mm) 훑기 — 값을 성공률로 고르지 않기 위한 근거 자료.
#
# 원본 실기 상수는 '보이는 윗면 중심에서 5mm 아래'(depth 15 - clearance 10)에 손끝을 둔다.
# 둥근 부위는 가장 넓은 단면(적도)보다 위를 쥐면 수박씨처럼 튕겨 나간다.
# 기하 예측: 엎드린 몸통의 보이는 점 중심 z≈36mm, 적도 z≈22mm
#            → 손끝이 적도에 닿으려면 depth ≥ 10 + 14 = 24mm.
# 전이가 이 근처에서 일어나면 '값'이 아니라 '부등식'(손끝 ≤ 적도)이 근거가 된다.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
OUT=${OUT:-/data/isaac-grasp-bench/eval/duck/depth_sweep}
mkdir -p "$OUT"
for D in 15 18 21 24 27 30 35; do  # 본실험 28 은 pca.csv
  $PY scripts/run_duck_eval.py --controller pca --num_envs 32 --repeats 2 \
      --grasp_depth_mm "$D" --out "$OUT/depth_${D}.csv" < /dev/null \
      2>&1 | grep -aE "^\[pca\]" | sed "s/^/[depth=$D] /"
done
