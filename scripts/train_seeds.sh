#!/usr/bin/env bash
# 여러 시드로 PPO 를 학습해 '학습 편차'를 측정할 수 있게 한다.
#
# 왜 필요한가: 시드 하나로 학습한 정책의 평가 편차는 '같은 정책을 여러 번 잰'
# 편차일 뿐이다. 다른 시드로 학습했을 때 얼마나 달라지는지를 모르면
# "RL 이 이 조건에서 무너진다"는 주장이 시드 노이즈와 구분되지 않는다(원칙 3).
#
# 멱등하다 — 최종 체크포인트가 이미 있는 시드는 건너뛴다.
# 중단되어도 다시 실행하면 남은 것부터 이어서 한다.
set -u
cd "$(dirname "$0")/.."

TASK=IsaacGraspBench-Lift-Cube-Franka-IK-Rel-v0
ENVS=${ENVS:-4096}
ITERS=${ITERS:-1500}
SEEDS=${SEEDS:-"2 3 4"}

for S in $SEEDS; do
  OUT="checkpoints/final/policy_ikrel_seed${S}.pt"
  if [ -f "$OUT" ]; then
    echo "[seed $S] 이미 있음 — 건너뜀"
    continue
  fi

  # 남아있는 같은 시드 프로세스를 정리하고 새로 시작한다.
  #
  # ⚠ 처음엔 '살아있으면 끝날 때까지 기다린다'로 짰다가 크게 데였다.
  #   래퍼 셸이 죽으면 SIGTERM 이 python 으로 전파되는데, Isaac Sim 은
  #   종료 핸들러에서 멈춰 CPU 만 태운다. py-spy 로 뜬 스택:
  #     _abort_signal_handle_callback -> simulation_app.close
  #       -> render -> torch.cuda.set_device
  #   프로세스는 '살아있지만' 학습은 0 이다 (체크포인트 0개, tfevents 88바이트).
  #   그 상태로 17분을 기다렸다. '살아있음'을 '진행 중'으로 보면 안 된다.
  #   게다가 일반 pkill 로는 안 죽어 sudo kill -9 가 필요했다.
  if pgrep -f "train_rl.py.*--seed ${S}$" >/dev/null 2>&1; then
    echo "[seed $S] 남은 프로세스 정리"
    sudo pkill -9 -f "train_rl.py.*--seed ${S}\$" 2>/dev/null || true
    sleep 5
  fi

  echo "[seed $S] 학습 시작 $(date +%H:%M)"
  python -u scripts/train_rl.py --task "$TASK" --headless \
    --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$S" \
    --run_name "ppo_ikrel_seed$S" 2>&1 | grep -E "Mean reward:" | tail -1

  D=$(ls -td runs/rsl_rl/franka_lift/*ppo_ikrel_seed"${S}" 2>/dev/null | head -1)
  if [ -z "${D:-}" ]; then echo "[seed $S] ERROR: run 디렉터리를 찾지 못했다"; continue; fi
  F=$(ls "$D"/*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  if [ -z "${F:-}" ]; then echo "[seed $S] ERROR: 체크포인트가 없다 — 학습이 실제로 돌았는지 확인하라"; continue; fi
  cp "$D/model_$F.pt" "$OUT"
  echo "[seed $S] 완료 $(date +%H:%M) -> $OUT (model_$F.pt)"
done
echo "전체 완료 $(date +%H:%M)"
