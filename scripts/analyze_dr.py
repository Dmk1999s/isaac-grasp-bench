"""랜덤화 강도가 실패 구간을 회복시켰는지 판정한다.

Phase 3 의 질문은 "DR 을 하면 좋아지는가"가 아니라
**"강도에 따른 차이가 시드 편차보다 큰가"** 다(원칙 3).

강도당 시드 3개를 학습했으므로, 각 폭에서
  - 강도 간 차이  (신호로 주장하고 싶은 것)
  - 시드 간 편차  (같은 강도인데도 갈리는 폭 = 노이즈)
를 나란히 놓고, 차이가 편차보다 작으면 '판단 보류'로 적는다.

기준선은 랜덤화 없이 학습한 기존 정책(policy_ikrel_seed*)이 아니라
**DR000(강도 0)** 이다. DR000 은 같은 코드 경로를 타므로
'랜덤화 효과'에 '코드 경로 차이'가 섞이지 않는다.

Isaac Sim 없이 돈다 — CSV 만 읽는다.
"""
import csv
import glob
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_CUBE_MM = 52.5


def rates_by_repeat(path):
    per = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            per[row["repeat"]].append(int(row["success"]))
    return [sum(v) / len(v) for v in per.values() if v]


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def main():
    # eval/dr<ST>_w<TAG>_s<SEED>.csv
    data = defaultdict(lambda: defaultdict(dict))  # width -> strength -> seed -> rate
    for p in sorted(glob.glob(str(ROOT / "eval" / "dr*_w*_s*.csv"))):
        m = re.match(r"dr(\d+)_w(\d+)_s(\d+)\.csv", Path(p).name)
        if not m:
            continue
        stg, wtag, seed = m.group(1), m.group(2), int(m.group(3))
        r = rates_by_repeat(p)
        if not r:
            continue
        # 파일명 태그 "135" -> 스케일 1.35. 소수점을 지운 값이라 되살린다.
        scale = float(wtag[0] + "." + wtag[1:]) if len(wtag) > 1 else float(wtag)
        data[round(BASE_CUBE_MM * scale, 1)][stg][seed] = sum(r) / len(r)

    if not data:
        print("eval/dr*_w*_s*.csv 가 없다 — scripts/run_dr_eval.sh 를 먼저 돌려라.")
        return

    strengths = sorted({s for w in data.values() for s in w})
    print(f"{'폭(mm)':>8}  " + "  ".join(f"{'DR'+s:>18}" for s in strengths)
          + "   판정 (기준=DR000)")
    print("-" * (10 + 20 * len(strengths) + 28))

    rows = []
    for width in sorted(data):
        cells, means, sds = [], {}, {}
        for stg in strengths:
            vals = list(data[width].get(stg, {}).values())
            if vals:
                means[stg] = st.mean(vals)
                sds[stg] = sd(vals)
                cells.append(f"{means[stg]*100:6.1f}% ±{sds[stg]*100:4.1f}")
            else:
                cells.append(f"{'—':>13}")

        verdict = "데이터 부족"
        if "000" in means and len(means) > 1:
            best = max((s for s in means if s != "000"), key=lambda s: means[s])
            diff = means[best] - means["000"]
            # 노이즈는 비교하는 두 강도의 시드 편차 중 큰 쪽 — 보수적으로 잡는다.
            noise = max(sds.get("000", 0.0), sds.get(best, 0.0))
            if abs(diff) > noise and noise > 0:
                verdict = (f"DR{best} {'우세' if diff > 0 else '열세'} "
                           f"({diff*100:+.1f}%p vs 편차 {noise*100:.1f}%p)")
            else:
                verdict = f"판단 보류 ({diff*100:+.1f}%p ≤ 편차 {noise*100:.1f}%p)"
            rows.append({"width_mm": width, "best": best,
                         "diff": round(diff, 4), "noise": round(noise, 4),
                         "verdict": verdict})
        print(f"{width:>8.1f}  " + "  ".join(f"{c:>18}" for c in cells) + f"   {verdict}")

    if rows:
        dst = ROOT / "eval" / "dr_summary.csv"
        with open(dst, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n기록: {dst}")

    print("\n> 강도 간 차이가 시드 편차보다 작으면 회복을 주장하지 않는다(원칙 3).")
    print("> 기준선은 기본 태스크가 아니라 DR000 이다 — 같은 코드 경로여야")
    print("> '랜덤화 효과'에 '코드 경로 차이'가 섞이지 않는다(원칙 1).")


if __name__ == "__main__":
    main()
