#!/usr/bin/env bash
# 조건 맞추기 2차 실험 (2026-10-20)
#   A. 튜닝 대칭: PCA 도 상자 실험 설정 그대로 (깊이 15, 바닥 여유 30 / 보정값 10)
#   B. 기존 본실험 설정 (깊이 28, 바닥 여유 5) — 새 점군 생성기·배치 기록으로 재실행
#   C. 관측·보상 대칭: PPO v2 (PCA 와 같은 보이는 점군만 관측, 손가락 접촉 보상)
#   모든 실행이 에피소드별 스폰·안착 자세를 기록한다 → 배치 동일성을 검증한다.
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh
OUT=${OUT:-/data/isaac-grasp-bench/eval/duck_aligned}
mkdir -p "$OUT"
COMMON="--num_envs 64 --repeats 5 --base_seed 1000"
run() {
  local name=$1; shift
  if [ -f "$OUT/$name.csv" ]; then echo "[$name] 이미 있음"; return; fi
  echo "[$name] 시작 $(date +%H:%M)"
  "$PY" scripts/run_duck_eval.py $COMMON "$@" --out "$OUT/$name.csv" \
     > "/data/isaac-grasp-bench/logs/aligned_$name.log" 2>&1 < /dev/null
  grep -aE "^\[(pca|rl)" "/data/isaac-grasp-bench/logs/aligned_$name.log" | sed "s/^/  /"
}
run pca_d28_c5        --controller pca --grasp_depth_mm 28 --floor_clearance_mm 5
run pca_d28_c5_taskv2 --controller pca --grasp_depth_mm 28 --floor_clearance_mm 5 --task_version v2
run pca_d15_c30       --controller pca --grasp_depth_mm 15 --floor_clearance_mm 30
run pca_d15_c10       --controller pca --grasp_depth_mm 15 --floor_clearance_mm 10
run ctrl_wrong_part   --controller pca --grasp_depth_mm 28 --floor_clearance_mm 5 --wrong_part
run ctrl_offset       --controller pca --grasp_depth_mm 28 --floor_clearance_mm 5 --grasp_offset_m 0.2
for S in 1 2 3; do run rl_v0_seed$S --controller rl --checkpoint checkpoints/duck/policy_duck_seed$S.pt; done
while pgrep -f "train_duck_seeds.sh" >/dev/null; do sleep 60; done
for S in ${V2_SEEDS:-1 2 3 4 5 6}; do
  CK=checkpoints/duck_v2/policy_duck_v2_seed$S.pt
  [ -f "$CK" ] && run rl_v2_seed$S --controller rl --task_version v2 --checkpoint "$CK"
done
echo "전체 완료 $(date +%H:%M)"
