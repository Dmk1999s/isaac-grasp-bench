#!/usr/bin/env bash
# Phase 3 — 도메인 랜덤화 강도별 학습.
#
# ⚠ 강도당 시드를 여러 개 돌린다. 이유:
#   격자 평가에서 같은 하이퍼파라미터·같은 환경인데 시드만 바꿔
#   wide16 성공률이 38.8% ~ 79.4% (편차 40.6%p) 로 벌어졌다.
#   강도당 시드 1개면 그려지는 곡선은 '강도의 효과'가 아니라 '시드 운'이다.
#   강도 간 차이가 시드 편차보다 커야 무언가를 주장할 수 있다(원칙 3).
#
# 규모를 줄인 기록(원칙 5):
#   강도 5단계 x 시드 3개 = 15회(약 7.5시간)는 과하다.
#   강도를 0 / 0.5 / 1.0 세 단계로 줄였다. 곡선의 모양이 아니라
#   '기울기가 있는가'를 먼저 본다. 필요하면 중간 강도를 나중에 채운다.
set -u
cd "$(dirname "$0")/.."
ITERS=${ITERS:-1500}
ENVS=${ENVS:-4096}
STRENGTHS=${STRENGTHS:-"000 050 100"}
SEEDS=${SEEDS:-"1 2 3"}

for ST in $STRENGTHS; do
  TASK="IsaacGraspBench-Lift-Cube-Franka-IK-Rel-DR${ST}-v0"
  for S in $SEEDS; do
    OUT="checkpoints/final/policy_dr${ST}_seed${S}.pt"
    [ -f "$OUT" ] && { echo "[dr$ST|seed$S] 이미 있음 — 건너뜀"; continue; }
    if pgrep -f "train_rl.py.*DR${ST}.*--seed ${S}$" >/dev/null 2>&1; then
      echo "[dr$ST|seed$S] 남은 프로세스 정리"
      sudo pkill -9 -f "train_rl.py.*DR${ST}.*--seed ${S}\$" 2>/dev/null || true
      sleep 5
    fi
    echo "[dr$ST|seed$S] 학습 시작 $(date +%H:%M)"
    python -u scripts/train_rl.py --task "$TASK" --headless \
      --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$S" \
      --run_name "ppo_dr${ST}_seed$S" 2>&1 | grep -E "Mean reward:" | tail -1
    D=$(ls -td runs/rsl_rl/franka_lift/*ppo_dr${ST}_seed${S} 2>/dev/null | head -1)
    [ -z "${D:-}" ] && { echo "[dr$ST|seed$S] ERROR: run 디렉터리 없음"; continue; }
    F=$(ls "$D"/*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
    [ -z "${F:-}" ] && { echo "[dr$ST|seed$S] ERROR: 체크포인트 없음"; continue; }
    cp "$D/model_$F.pt" "$OUT"
    echo "[dr$ST|seed$S] 완료 $(date +%H:%M) -> $OUT"
  done
done
echo "전체 완료 $(date +%H:%M)"
