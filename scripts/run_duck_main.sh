#!/usr/bin/env bash
# 오리 본실험 일괄 실행 — 깊이 훑기와 PPO 학습이 끝나기를 기다렸다가 평가한다.
#   PCA 본실험 / 강체 비교 / 대조군 2종 → (학습 완료 후) PPO 시드별
# 모든 실행이 같은 평가 시드(1000~1004), 같은 env 수, 같은 성공 판정을 쓴다.
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh
OUT=/data/isaac-grasp-bench/eval/duck
# 깊이 28mm 의 근거 (성공률로 고르지 않았다):
#   형상 계산 — 손끝이 부위의 가장 넓은 단면(적도)에 닿는 깊이 = 10 + (보이는 점 중심 z - 부위 중심 z)
#   엎드림 머리 22.2 / 엎드림 몸통 25.1 / 옆 머리 25.3 / 옆 몸통 27.0 mm (yaw 50개 중앙값, 최대 27.3)
#   → 머리·몸통 모든 자세를 만족하는 최소 정수 28. 다리는 바닥 클리어런스가 지배해 이 기준 밖이다.
COMMON="--num_envs 64 --repeats 5 --base_seed 1000 --grasp_depth_mm 28"
run() {  # run <name> <args...>
  local name=$1; shift
  if [ -f "$OUT/$name.csv" ]; then echo "[$name] 이미 있음"; return; fi
  echo "[$name] 시작 $(date +%H:%M)"
  "$PY" scripts/run_duck_eval.py $COMMON "$@" --out "$OUT/$name.csv" \
     > "/data/isaac-grasp-bench/logs/eval_$name.log" 2>&1 < /dev/null
  grep -aE "^\[(pca|rl)" "/data/isaac-grasp-bench/logs/eval_$name.log" | sed "s/^/  /"
}
run pca             --controller pca
run pca_rigid       --controller pca --usd assets/duck/duck_rigid.usda
run ctrl_wrong_part --controller pca --wrong_part
run ctrl_offset     --controller pca --grasp_offset_m 0.2
while pgrep -f "train_duck_seeds.sh" >/dev/null; do sleep 60; done
for S in 1 2 3; do
  CK=checkpoints/duck/policy_duck_seed$S.pt
  [ -f "$CK" ] && run rl_seed$S --controller rl --checkpoint "$CK"
done
echo "전체 완료 $(date +%H:%M)"
