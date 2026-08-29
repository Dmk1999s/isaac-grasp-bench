#!/usr/bin/env bash
# 물체 크기를 촘촘히 훑어 'PCA 의 실패 시작 크기'를 관측에서 찾는다.
#
# 왜: 지금까지의 측정은 한쪽으로만 촘촘하다.
#     닫힘축 폭(RL 이 무너지는 축)은 폭 스윕 30회 + DR 45회 = 75회를 썼는데,
#     크기(PCA 가 무너지는 축)는 격자 셀 하나(26mm)뿐이다.
#     check_criteria.py 의 '관측에서 임계값 도출'(원칙 4)이 RL 에만 적용됐다.
#
# 왜 PCA 를 두 팔로 나누나:
#     FLOOR_CLEARANCE_MM=30 이 손끝 목표 z 를 30mm 아래로 못 내려가게 클램프한다.
#     26mm 큐브는 윗면이 26.2mm 라 손끝이 물체 위에 서고, 그리퍼가 허공에서 닫힌다.
#     (Isaac Sim 없이 확인: 손끝 30.0mm vs 윗면 26.2mm, 바닥가드 상승분 8.7mm)
#     즉 한 팔만 재면 '알고리즘'이 아니라 '내가 고른 상수'를 재게 된다 — 원칙 1 위반.
#     clr 은 크기별로 바꾸지 않고 10mm 고정이다. 크기마다 바꾸면 그것도 변수가 된다.
#
# 앵커: scale 0.5(26.2mm)와 0.8(42.0mm)은 기존 격자 셀과 같은 조건이다.
#       기존 측정(PCA 1.9%/100%, RL 95.3%/100%)과 어긋나면 스윕을 믿으면 안 된다.
#
# 멱등하다 — 결과 CSV 가 있으면 건너뛴다.
set -u
cd "$(dirname "$0")/.."
. scripts/env.sh   # EULA·PY 등 실행 환경 고정
REPEATS=${REPEATS:-5}
ENVS=${ENVS:-32}
SEEDS=${SEEDS:-"1 2 3 4"}
CLR_LOW=${CLR_LOW:-10}
# scale -> 한 변(mm): 0.5=26.2 0.55=28.9 0.6=31.5 0.65=34.1 0.7=36.8 0.8=42.0
# 0.5~0.55 는 손끝이 윗면 위(예측: 실패), 0.6~0.65 는 전이 구간,
# 0.7 부터 바닥가드가 아예 안 걸린다(예측: 정상).
SCALES=${SCALES:-"0.5 0.55 0.6 0.65 0.7 0.8"}

for W in $SCALES; do
  TAG=$(echo "$W" | tr -d '.')
  SCALE="${W},${W},${W}"

  # A팔: 지금까지 측정한 조건 그대로 (clr=30)
  OUT="eval/size_s${TAG}_pca_clr30.csv"
  if [ -f "$OUT" ]; then echo "[s$W|pca clr30] 이미 있음"; else
    R=$(timeout 1800 "$PY" -u scripts/run_eval.py --controller pca --headless \
        --object_scale "$SCALE" --num_envs $ENVS --repeats $REPEATS \
        --floor_clearance_mm 30 --out "$OUT" 2>&1 | grep -E "^\[pca\] 성공률" | head -1)
    echo "[s$W|pca clr30] $R"
  fi

  # B팔: 바닥가드를 물리적 조건이 요구하는 만큼만 (clr=10)
  OUT="eval/size_s${TAG}_pca_clr${CLR_LOW}.csv"
  if [ -f "$OUT" ]; then echo "[s$W|pca clr$CLR_LOW] 이미 있음"; else
    R=$(timeout 1800 "$PY" -u scripts/run_eval.py --controller pca --headless \
        --object_scale "$SCALE" --num_envs $ENVS --repeats $REPEATS \
        --floor_clearance_mm $CLR_LOW --out "$OUT" 2>&1 | grep -E "^\[pca\] 성공률" | head -1)
    echo "[s$W|pca clr$CLR_LOW] $R"
  fi

  # C팔: RL (체크포인트 재사용 — 재학습 없음). clr 은 휴리스틱 전용이라 무관하다.
  for S in $SEEDS; do
    OUT="eval/size_s${TAG}_rl_s${S}.csv"
    [ -f "$OUT" ] && { echo "[s$W|seed$S] 이미 있음"; continue; }
    CKPT="checkpoints/final/policy_ikrel_seed${S}.pt"
    [ -f "$CKPT" ] || { echo "[s$W|seed$S] ERROR: 체크포인트 없음"; continue; }
    R=$(timeout 1800 "$PY" -u scripts/run_eval.py --controller rl --headless \
        --object_scale "$SCALE" --num_envs $ENVS --repeats $REPEATS \
        --checkpoint "$CKPT" --out "$OUT" 2>&1 | grep -E "^\[rl\] 성공률" | head -1)
    echo "[s$W|seed$S] $R"
  done
done
echo "전체 완료 $(date +%H:%M)"
