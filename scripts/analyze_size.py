"""크기 축에서 두 제어 방식을 비교하고, 바닥 클리어런스 상수의 영향을 분리한다.

왜 이 스크립트가 따로 필요한가
------------------------------
`compare_runs.py` 의 격자는 PCA 를 한 조건(`floor_clearance_mm=30`)에서만 잰다.
그런데 26mm 셀에서 그 상수가 결과를 통째로 만든다 — 손끝 목표가 물체 윗면 위에
서서 그리퍼가 허공에서 닫힌다. 그러면 셀의 차이를 '제어 방식'으로 귀속할 수 없다
(실험 설계 원칙 1). 그래서 여기서는 PCA 를 두 팔로 잰다:

  A팔: clr=30            — 격자에서 쓰던 조건 그대로
  B팔: clr < 물체 높이   — 물리적 조건을 만족하는 값

판정 규칙과 편차 척도(최대-최소)는 `compare_runs.py` 와 같다. 다르게 쓰면
같은 저장소 안에서 판정이 서로 비교되지 않는다.

Isaac Sim 없이 돈다 (CSV 만 읽는다).
"""
import argparse
import csv
import glob
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_CUBE_MM = 52.5
SCALES = ["0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.8"]


def available_seeds(ev):
    """시드를 하드코딩하지 않는다 — 있는 만큼 쓴다.
    시드를 늘렸을 때 스크립트를 고치는 걸 잊으면 조용히 옛 결론이 유지된다."""
    import re
    ss = set()
    for f in glob.glob(str(ev / "size_s*_rl_s*.csv")):
        m = re.search(r"_rl_s(\d+)\.csv$", f)
        if m:
            ss.add(int(m.group(1)))
    return tuple(sorted(ss))

CLEARANCES = [5, 10, 15, 20, 25, 30]      # 클리어런스 스윕 (26.2mm 고정)
CLR_LOW = 10                              # B팔에 쓴 값


def rates_by_repeat(path):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["repeat"], []).append(int(r["success"]))
    return np.array([np.mean(v) for v in by.values()]) * 100.0


def spread(a):
    """편차 척도 = 최대-최소. compare_runs.py:44 와 같아야 한다."""
    return float(a.max() - a.min()) if len(a) > 1 else 0.0


def tag(scale):
    return scale.replace(".", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/size_sweep.md")
    args = ap.parse_args()
    ev = ROOT / "eval"
    SEEDS = available_seeds(ev)
    out = []

    out += ["# 크기 축 실패 지도 — 상수의 영향을 분리한다", "",
            "같은 태스크·같은 액션 인터페이스(IK-Rel)·같은 성공 기준.",
            f"크기당 PCA 2회 + RL 시드 {len(SEEDS)}개, 회당 5반복 x 32 env = 160 에피소드.", "",
            "PCA 를 두 조건에서 잰다 — `floor_clearance_mm` 이 작은 물체에서 결과를",
            "통째로 만들기 때문이다. 한 조건만 재면 '제어 방식'이 아니라 '내가 고른 상수'를",
            "재게 된다(원칙 1).", ""]

    hdr = ("| 크기 (mm) | PCA `clr=30` | PCA `clr=%d` | RL (시드 %d개) | 차이 | 노이즈 | 판정 |"
           % (CLR_LOW, len(SEEDS)))
    out += [hdr, "|---|---:|---:|---:|---:|---:|---|"]

    rows = []
    for s in SCALES:
        mm = float(s) * BASE_CUBE_MM
        t = tag(s)
        a = rates_by_repeat(ev / f"size_s{t}_pca_clr30.csv")
        b = rates_by_repeat(ev / f"size_s{t}_pca_clr{CLR_LOW}.csv")
        per_seed = np.array([rates_by_repeat(ev / f"size_s{t}_rl_s{k}.csv").mean()
                             for k in SEEDS])
        diff = b.mean() - per_seed.mean()
        noise = max(spread(b), spread(per_seed))
        verdict = ("판단 보류 (편차 이내)" if abs(diff) <= noise
                   else ("**PCA 우세**" if diff > 0 else "**RL 우세**"))
        rows.append((mm, a.mean(), b.mean(), per_seed, diff, noise, verdict))
        out.append(f"| {mm:.1f} | {a.mean():.1f}% | {b.mean():.1f}% | "
                   f"{per_seed.mean():.1f}% | {diff:+.1f}%p | {noise:.1f}%p | {verdict} |")

    out += ["",
            "> 노이즈는 PCA 반복 편차와 RL **시드 간** 편차 중 큰 쪽이다(최대-최소).",
            "> `compare_runs.py` 와 같은 규칙을 쓴다 — 다르게 쓰면 판정끼리 비교되지 않는다.",
            ""]

    # ── 분포 요약 ─────────────────────────────────────────────────
    # 평균은 이 데이터에서 어느 시드도 대표하지 않는다. 18.4mm 의 시드는
    # 0.0 / 15.6 / ... / 66.2 / 100.0% 로 갈라지는데 평균은 35.8% 다.
    # 그래서 중앙값·범위와 '의미 있게 낮은 시드 수'를 같이 낸다.
    #
    # 임계값 T 는 고르지 않고 도출한다(원칙 4): 학습 분포 크기(42.0mm)에서
    # 관측된 시드 간 범위다. 그 크기에서는 분포 밖 효과가 없으므로,
    # 거기서 시드가 흔들리는 폭보다 크게 낮으면 분포 밖 효과로 본다.
    base_seeds = np.array([rates_by_repeat(ev / f"size_s08_rl_s{k}.csv").mean()
                           for k in SEEDS])
    T = spread(base_seeds)

    out += ["## 분포 요약 — 평균이 대표하지 못하는 구간", "",
            f"임계값 T = **{T:.1f}%p**. 고른 값이 아니라 학습 분포 크기(42.0mm)에서",
            "관측된 시드 간 범위다 — 그 크기에서는 분포 밖 효과가 없다(원칙 4).", "",
            "| 크기 (mm) | RL 중앙값 | RL 최저 | RL 최고 | PCA 보다 T 이상 낮은 시드 |",
            "|---|---:|---:|---:|---:|"]
    for mm, _, b_m, per_seed, _, _, _ in rows:
        k = int((per_seed < b_m - T).sum())
        out.append(f"| {mm:.1f} | {np.median(per_seed):.1f}% | {per_seed.min():.1f}% "
                   f"| {per_seed.max():.1f}% | **{k} / {len(per_seed)}** |")
    out += ["",
            "> 크기와 빈도를 같이 센다. 부호검정은 천장 효과에 끌려 0.6%p 차이도",
            "> 유의에 가깝게 만들고, t분포 신뢰구간은 상한이 100% 를 넘는다",
            "> (성공률은 0~100 에 갇힌 값이다). 둘 다 판정 규칙으로 쓰지 않는다.",
            ""]

    out += ["## 시드별 상세 (RL)", "",
            "| 크기 (mm) | " + " | ".join(f"seed {k}" for k in SEEDS) + " | 시드 간 편차 |",
            "|---|" + "---|" * (len(SEEDS) + 1)]
    for mm, _, _, per_seed, _, _, _ in rows:
        out.append(f"| {mm:.1f} | " + " | ".join(f"{v:.1f}%" for v in per_seed)
                   + f" | {spread(per_seed):.1f}%p |")

    out += ["", "## 바닥 클리어런스 스윕 — 26.2mm 물체 고정", "",
            "값을 성공률로 고르지 않기 위해서다(원칙 4). 물체 높이는 26.25mm 다.", "",
            "| `floor_clearance_mm` | " + " | ".join(str(c) for c in CLEARANCES) + " |",
            "|---|" + "---:|" * len(CLEARANCES)]
    clr_row = []
    for c in CLEARANCES:
        p = ev / f"size_s05_pca_clr{c}.csv"
        clr_row.append(f"{rates_by_repeat(p).mean():.1f}%" if p.exists() else "—")
    out.append("| 성공률 | " + " | ".join(clr_row) + " |")
    out += ["",
            "5~25mm 가 모두 100% 다 — **`clr=10` 은 특별한 값이 아니다.**",
            "전이는 25 와 30 사이, 즉 물체 높이(26.25mm) 근처에서 일어난다.",
            "그래서 도출되는 조건은 값이 아니라 부등식이다: **`clr < 물체 높이`**.",
            "Phase 1 의 `depth > clearance` 와 같은 형태다.", ""]

    text = "\n".join(out) + "\n"
    (ROOT / args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
