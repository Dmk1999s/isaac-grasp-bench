#!/usr/bin/env bash
# 가림 실험 (2026-10-20) — 인식만 가리는 반평면 판. PCA·PPO v2 가 같은 판을 받는다.
#   가림 0% 결과는 eval/duck_aligned 를 그대로 쓴다 (같은 배치·같은 코드 경로).
#   QUEUE=a|b 로 나눠 두 개를 동시에 돌린다 (GPU 여유).
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh
OUT=${OUT:-/data/isaac-grasp-bench/eval/duck_occlusion}
mkdir -p "$OUT"
COMMON="--num_envs 64 --repeats 5 --base_seed 1000"
run() {
  local name=$1; shift
  if [ -f "$OUT/$name.csv" ]; then echo "[$name] 이미 있음"; return; fi
  echo "[$name] 시작 $(date +%H:%M)"
  "$PY" scripts/run_duck_eval.py $COMMON "$@" --out "$OUT/$name.csv" \
     > "/data/isaac-grasp-bench/logs/occ_$name.log" 2>&1 < /dev/null
  grep -aE "^\[(pca|rl)" "/data/isaac-grasp-bench/logs/occ_$name.log" | sed "s/^/  /"
}
LEVELS="25 50 75"
case "${QUEUE:-a}" in
  a)
    for L in $LEVELS; do
      F=$(awk "BEGIN{print $L/100}")
      run pca_d28_c5_occ$L  --controller pca --grasp_depth_mm 28 --floor_clearance_mm 5 --occlusion $F
      run pca_d15_c30_occ$L --controller pca --grasp_depth_mm 15 --floor_clearance_mm 30 --occlusion $F
    done
    SEEDS="1 2 3" ;;
  b) SEEDS="4 5 6" ;;
esac
for S in $SEEDS; do
  for L in $LEVELS; do
    F=$(awk "BEGIN{print $L/100}")
    run rl_v2_seed${S}_occ$L --controller rl --task_version v2 \
        --checkpoint checkpoints/duck_v2/policy_duck_v2_seed$S.pt --occlusion $F
  done
done
echo "전체 완료 $(date +%H:%M)"
