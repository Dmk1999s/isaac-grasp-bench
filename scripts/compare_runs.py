"""두 제어 방식을 물체 격자 위에서 비교하고 판정까지 적는다.

설계 출처: awsim-autoware-bench/src/autoware_bench/scripts/compare_runs.py
그쪽에서 가져온 핵심은 `ab_table` 의 판정 규칙이다 —
**차이가 반복 편차보다 작으면 '판단 보류'로 적는다.** 같은 조건에서도 그만큼
흔들리므로 노이즈와 구분되지 않기 때문이다(실험 설계 원칙 3).

이 프로젝트에 맞게 바꾼 것:
  - 편차의 출처가 둘이다. PCA 는 '반복 편차', RL 은 '시드 간 편차'.
    RL 은 학습 자체가 시드에 따라 달라지므로, 같은 정책을 여러 번 잰 편차만
    쓰면 노이즈를 과소평가한다.
  - 그래서 판정에 쓰는 노이즈는 **둘 중 큰 쪽**이다. 보수적인 선택이다.

Isaac Sim 없이 돈다 (CSV 만 읽는다).
"""
import argparse
import csv
import glob
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

CELLS = [
    ("base",    "42×42×42",  "기준"),
    ("elong15", "63×42×42",  "길쭉"),
    ("elong20", "84×42×42",  "더 길쭉"),
    ("wide12",  "42×63×42",  "닫힘축 넓음"),
    ("wide16",  "42×84×42",  "그리퍼 한계 초과"),
    ("small",   "26×26×26",  "작음"),
]


def rates_by_repeat(path):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["repeat"], []).append(int(r["success"]))
    return np.array([np.mean(v) for v in by.values()]) * 100.0


def spread(a):
    """편차 척도 = 최대-최소. 표준편차보다 보수적이다(원본 설계와 동일)."""
    return float(a.max() - a.min()) if len(a) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/comparison.md")
    a = ap.parse_args()

    L = ["# 제어 방식 비교 — 물체 격자", "",
         "같은 태스크·같은 액션 인터페이스(IK-Rel)·같은 시드·같은 성공 기준.",
         "달라지는 변수는 제어 방식 하나뿐이다.", "",
         "| 물체 (mm) | 성격 | PCA | RL (시드 평균) | 차이 | 노이즈 | 판정 |",
         "|---|---|---:|---:|---:|---:|---|"]

    rows_done = 0
    for key, size, note in CELLS:
        p = ROOT / f"eval/grid_{key}_pca.csv"
        seeds = sorted(glob.glob(str(ROOT / f"eval/gridseed_{key}_rl_s*.csv")))
        if not p.exists() or not seeds:
            continue
        pca = rates_by_repeat(p)
        per_seed = np.array([rates_by_repeat(f).mean() for f in seeds])

        pca_m, rl_m = pca.mean(), per_seed.mean()
        diff = rl_m - pca_m
        # 노이즈: PCA 반복 편차와 RL 시드 간 편차 중 큰 쪽 (보수적)
        noise = max(spread(pca), spread(per_seed))

        if abs(diff) <= noise:
            verdict = "판단 보류 (편차 이내)"
        elif pca_m >= 99.9 and rl_m >= 99.9:
            verdict = "판단 보류 (양쪽 천장)"
        else:
            verdict = "**RL 우세**" if diff > 0 else "**PCA 우세**"

        L.append(f"| {size} | {note} | {pca_m:.1f}% | {rl_m:.1f}% "
                 f"| {diff:+.1f}%p | {noise:.1f}%p | {verdict} |")
        rows_done += 1

    if rows_done == 0:
        print("비교할 데이터가 아직 없다.")
        return

    L += ["",
          f"> RL 은 시드 {len(seeds)}개의 평균이다. 노이즈는 PCA 의 반복 편차와",
          "> RL 의 **시드 간** 편차 중 큰 쪽을 썼다 — 학습 자체가 시드에 따라 달라지므로",
          "> 같은 정책을 여러 번 잰 편차만 쓰면 노이즈를 과소평가한다.",
          "",
          "> 차이가 노이즈보다 작으면 개선/악화를 주장하지 않는다.",
          "> 양쪽이 모두 100% 에 붙은 셀도 마찬가지다 — 그 조건은 변별력이 없다는 뜻이다.",
          ""]

    # 시드별 상세
    L += ["## 시드별 상세 (RL)", "",
          "| 물체 (mm) | " + " | ".join(f"seed {pathlib.Path(f).stem.split('_s')[-1]}"
                                        for f in seeds) + " | 시드 간 편차 |",
          "|---" * (len(seeds) + 2) + "|"]
    for key, size, _ in CELLS:
        seeds_k = sorted(glob.glob(str(ROOT / f"eval/gridseed_{key}_rl_s*.csv")))
        if not seeds_k:
            continue
        vals = [rates_by_repeat(f).mean() for f in seeds_k]
        L.append(f"| {size} | " + " | ".join(f"{v:.1f}%" for v in vals)
                 + f" | {spread(np.array(vals)):.1f}%p |")
    L.append("")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n기록: {out}")


if __name__ == "__main__":
    main()
