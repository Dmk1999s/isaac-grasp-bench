#!/usr/bin/env bash
# 물체 자세(yaw)를 훑는다 — 계획에 있었으나 비어 있던 축.
#
# 왜: 기본 태스크의 reset_object_position 은 pose_range 에 x/y/z 만 있고
#     회전이 없다(lift_env_cfg.py:127). 그래서 지금까지의 모든 측정이
#     **축 정렬된 상자**로만 이뤄졌다 — PCA 에 가장 유리한 입력이다
#     (장축이 곧 월드 축이라 추정이 틀릴 여지가 없다).
#     README 의 계획은 "물체 자세·크기·재질 격자"인데 크기만 훑은 상태였다.
#
# 물체는 길쭉한 84x42x42 로 고정한다. 정육면체는 yaw 를 돌려도 대칭이라
# 아무 일도 일어나지 않는다 — 변별력이 없다.
# 직육면체는 180도 주기이고 90도에서 축이 뒤바뀌므로 0~90 도면 전 구간이다.
#
# RL 은 축 정렬 물체로만 학습했다. yaw 는 학습 분포 밖이다.
# 멱등하다 — 결과 CSV 가 있으면 건너뛴다.
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh
REPEATS=${REPEATS:-5}
ENVS=${ENVS:-32}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8"}
SCALE=${SCALE:-"1.6,0.8,0.8"}      # 84 x 42 x 42
YAWS=${YAWS:-"0 15 30 45 60 75 90"}

for Y in $YAWS; do
  OUT="eval/yaw_y${Y}_pca.csv"
  if [ -f "$OUT" ]; then echo "[yaw$Y|pca] 이미 있음"; else
    R=$(timeout 1800 "$PY" -u scripts/run_eval.py --controller pca --headless \
        --object_scale "$SCALE" --object_yaw_deg "$Y" --num_envs $ENVS --repeats $REPEATS \
        --out "$OUT" 2>&1 | grep -E "^\[pca\] 성공률|경고" | head -2)
    echo "[yaw$Y|pca] $R"
  fi

  for S in $SEEDS; do
    OUT="eval/yaw_y${Y}_rl_s${S}.csv"
    [ -f "$OUT" ] && { echo "[yaw$Y|seed$S] 이미 있음"; continue; }
    CKPT="checkpoints/final/policy_ikrel_seed${S}.pt"
    [ -f "$CKPT" ] || { echo "[yaw$Y|seed$S] ERROR: 체크포인트 없음"; continue; }
    R=$(timeout 1800 "$PY" -u scripts/run_eval.py --controller rl --headless \
        --object_scale "$SCALE" --object_yaw_deg "$Y" --num_envs $ENVS --repeats $REPEATS \
        --checkpoint "$CKPT" --out "$OUT" 2>&1 | grep -E "^\[rl\] 성공률" | head -1)
    echo "[yaw$Y|seed$S] $R"
  done
done
echo "전체 완료 $(date +%H:%M)"
