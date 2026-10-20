#!/usr/bin/env bash
# 오리 '지시 부위' 태스크 PPO 를 여러 시드로 학습한다 (학습 편차를 재기 위해, 원칙 3).
# 하이퍼파라미터는 큐브와 동일 (lift_duck_env.DuckPPORunnerCfg — 실험 이름만 다르다).
# 멱등: 최종 체크포인트가 있는 시드는 건너뛴다.
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh

# v2(관측·보상을 PCA 와 맞춘 태스크): TASK=IsaacGraspBench-Lift-Duck-Franka-IK-Rel-v2 EXP=franka_lift_duck_v2 TAG=duck_v2
TASK=${TASK:-IsaacGraspBench-Lift-Duck-Franka-IK-Rel-v0}
EXP=${EXP:-franka_lift_duck}
TAG=${TAG:-duck}
ENVS=${ENVS:-4096}
ITERS=${ITERS:-1500}
SEEDS=${SEEDS:-"1 2 3"}
mkdir -p checkpoints/$TAG

for S in $SEEDS; do
  OUT="checkpoints/$TAG/policy_${TAG}_seed${S}.pt"
  if [ -f "$OUT" ]; then echo "[seed $S] 이미 있음 — 건너뜀"; continue; fi
  echo "[seed $S] 학습 시작 $(date +%H:%M)"
  "$PY" -u scripts/train_rl.py --task "$TASK" --headless \
    --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$S" \
    --run_name "ppo_${TAG}_seed$S" > "/data/isaac-grasp-bench/logs/train_${TAG}_seed$S.log" 2>&1 < /dev/null
  D=$(ls -td logs/rsl_rl/$EXP/*ppo_${TAG}_seed"${S}" 2>/dev/null | head -1)
  if [ -z "${D:-}" ]; then echo "[seed $S] ERROR: run 디렉터리를 찾지 못했다"; continue; fi
  F=$(ls "$D"/*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  if [ -z "${F:-}" ]; then echo "[seed $S] ERROR: 체크포인트 없음"; continue; fi
  cp "$D/model_$F.pt" "$OUT"
  echo "[seed $S] 완료 $(date +%H:%M) -> $OUT (model_$F.pt, run=$D)"
done
echo "전체 완료 $(date +%H:%M)"
