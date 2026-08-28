"""시드별 격자 결과를 모아 '반복 편차'와 '학습 편차'를 나란히 놓는다.

원칙 3(편차보다 작은 차이는 주장하지 않는다)을 시드 축으로 확장한 것이다.

  반복 편차 — 같은 정책을 여러 번 잰 편차. 셀 안에서 repeat 별 성공률의 표준편차.
  학습 편차 — 시드만 바꿔 다시 학습한 정책들 사이의 편차. 시드별 성공률의 표준편차.

앞의 실패 지도는 seed 1 하나로 그렸다. 학습 편차가 셀 간 차이보다 크면
그 셀의 결론은 '이 시드의 성질'이지 '학습 방식의 성질'이 아니다.

Isaac Sim 없이 돈다 — CSV 만 읽는다.
"""
import csv
import glob
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL_DESC = {
    "base": "42x42x42 (기준)",
    "wide12": "42x63x42 (닫힘축 넓음)",
    "wide16": "42x84x42 (한계 초과)",
    "small": "26x26x26 (작음)",
}


def rates_by_repeat(path):
    """repeat 별 성공률 리스트를 낸다. 이게 '반복 편차'의 재료다."""
    per_repeat = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            per_repeat[row["repeat"]].append(int(row["success"]))
    return [sum(v) / len(v) for v in per_repeat.values() if v]


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def main():
    # eval/gs_<cell>_seed<N>.csv 를 모은다.
    found = defaultdict(dict)  # cell -> seed -> rates
    for p in sorted(glob.glob(str(ROOT / "eval" / "gs_*_seed*.csv"))):
        m = re.match(r"gs_(.+)_seed(\d+)\.csv", Path(p).name)
        if not m:
            continue
        cell, seed = m.group(1), int(m.group(2))
        r = rates_by_repeat(p)
        if r:
            found[cell][seed] = r

    if not found:
        print("eval/gs_*_seed*.csv 가 없다 — scripts/run_grid_seeds.sh 를 먼저 돌려라.")
        return

    out_rows = []
    order = [c for c in CELL_DESC if c in found] + [c for c in found if c not in CELL_DESC]
    print(f"{'셀':<26} {'시드별 성공률':<34} {'평균':>7} {'학습편차':>9} {'반복편차':>9}")
    print("-" * 90)
    for cell in order:
        seeds = sorted(found[cell])
        means = [sum(found[cell][s]) / len(found[cell][s]) for s in seeds]
        # 반복 편차는 시드마다 재서 평균낸다 — 시드 하나의 우연에 기대지 않는다.
        rep_sd = st.mean([sd(found[cell][s]) for s in seeds])
        per_seed = "  ".join(f"s{s}:{m*100:5.1f}" for s, m in zip(seeds, means))
        label = f"{cell} {CELL_DESC.get(cell, '')}"
        print(f"{label:<26} {per_seed:<34} {st.mean(means)*100:6.1f}% "
              f"{sd(means)*100:8.1f}%p {rep_sd*100:8.1f}%p")
        out_rows.append({
            "cell": cell,
            "n_seeds": len(seeds),
            "seeds": " ".join(map(str, seeds)),
            "mean_success": round(st.mean(means), 4),
            "seed_sd": round(sd(means), 4),
            "repeat_sd": round(rep_sd, 4),
        })

    dst = ROOT / "eval" / "seed_variance.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n기록: {dst}")

    # 해석을 사람이 읽을 수 있게 한 줄로 못박는다.
    print("\n판정 (원칙 3):")
    for r in out_rows:
        if r["n_seeds"] < 2:
            print(f"  {r['cell']}: 시드 {r['n_seeds']}개 — 학습 편차를 잴 수 없다. 판단 보류.")
        elif r["seed_sd"] > r["repeat_sd"] * 2:
            print(f"  {r['cell']}: 학습 편차({r['seed_sd']*100:.1f}%p)가 반복 편차"
                  f"({r['repeat_sd']*100:.1f}%p)보다 크다 — 시드 하나로는 결론을 못 낸다.")
        else:
            print(f"  {r['cell']}: 학습 편차({r['seed_sd']*100:.1f}%p)가 반복 편차 수준 "
                  f"— 시드에 안정적이다.")


if __name__ == "__main__":
    main()
