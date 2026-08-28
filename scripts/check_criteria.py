"""관측치에서 '실패가 시작되는 지점'을 도출한다. 임계값을 미리 박지 않는다.

설계 출처: awsim-autoware-bench/scripts/check_criteria.py
그쪽은 criteria.yaml 로 합격/불합격을 판정하고 종료코드를 낸다.
여기서는 판정 기준 자체를 **관측에서 만든다**(실험 설계 원칙 4).

방법
----
1. 가장 쉬운 조건(폭이 가장 작은 셀)을 기준선으로 삼는다.
2. 그 조건의 **반복 편차**로 허용 밴드를 만든다: [평균 - 편차폭, 100]
   편차폭 = 최대-최소 (표준편차보다 보수적). 편차가 0 이면 최소 밴드 5%p 를 준다 —
   천장에 붙은 조건은 편차가 0 으로 나오는데, 그걸 '오차 없음'으로 읽으면
   1 에피소드만 실패해도 '실패 시작'으로 잘못 판정된다.
3. 폭을 키우며 처음으로 밴드 아래로 내려가는 지점이 **도출된 실패 시작 폭**이다.

종료코드: 두 방식 모두 실패 시작점을 찾았으면 0, 못 찾았으면 1(데이터 부족).
"""
import argparse
import csv
import glob
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_CUBE_MM = 52.5          # DexCube 한 변 (측정: scale 0.8 에서 42.0mm)
MIN_BAND_PP = 5.0            # 천장 조건의 편차 0 을 그대로 쓰지 않기 위한 최소 밴드


def rates_by_repeat(path):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["repeat"], []).append(int(r["success"]))
    return np.array([np.mean(v) for v in by.values()]) * 100.0


def collect(pattern, scale_from_name):
    """{폭mm: [반복별 성공률...]} 를 모은다."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "eval" / pattern))):
        w = scale_from_name(pathlib.Path(f).name)
        if w is None:
            continue
        out.setdefault(w, []).append(rates_by_repeat(f))
    return {w: np.concatenate(v) for w, v in sorted(out.items())}


def width_from_tag(name):
    m = re.match(r"width_w(\d+)_", name)
    if not m:
        return None
    tag = m.group(1)
    scale = float(tag[0] + "." + tag[1:]) if len(tag) > 1 else float(tag)
    return round(scale * BASE_CUBE_MM, 1)


def onset(series, label):
    """도출된 실패 시작 폭과 근거를 낸다."""
    widths = sorted(series)
    if len(widths) < 2:
        return None, f"{label}: 표본이 부족하다 (폭 {len(widths)}개)"

    base_w = widths[0]
    base = series[base_w]
    band = max(float(base.max() - base.min()), MIN_BAND_PP)
    floor = float(base.mean()) - band

    lines = [f"{label}",
             f"  기준선 = 폭 {base_w}mm: 평균 {base.mean():.1f}%, "
             f"반복 편차 {base.max()-base.min():.1f}%p -> 허용 밴드 하한 {floor:.1f}%"]
    hit = None
    for w in widths:
        m = float(series[w].mean())
        mark = ""
        if hit is None and m < floor:
            hit = w
            mark = "  <- 여기서 밴드 아래로 내려간다"
        lines.append(f"  폭 {w:6.1f}mm : {m:6.1f}%{mark}")
    if hit is None:
        lines.append(f"  -> 관측 범위({widths[0]}~{widths[-1]}mm) 안에서는 실패 시작점이 없다")
    else:
        lines.append(f"  -> 도출된 실패 시작 폭: **{hit}mm**")
    return hit, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    pca = collect("width_w*_pca.csv", width_from_tag)
    rl = collect("width_w*_rl_s*.csv", width_from_tag)
    if not pca and not rl:
        print("폭 스윕 결과가 없다 — scripts/run_width_sweep.sh 를 먼저 돌려라.")
        return 1

    print("실패 시작 지점 — 관측에서 도출 (임계값을 미리 정하지 않는다)\n")
    print(f"참고: Franka Hand 개방 한계는 80mm 다. "
          f"이 값은 '설계 사양'이지 '관측된 실패 시작점'이 아니다.\n")

    h1, t1 = onset(pca, "PCA 휴리스틱") if pca else (None, "PCA: 데이터 없음")
    h2, t2 = onset(rl, "PPO 정책 (시드 합산)") if rl else (None, "RL: 데이터 없음")
    print(t1); print(); print(t2)

    if h1 is not None and h2 is not None:
        print(f"\n두 방식의 실패 시작 폭: PCA {h1}mm vs RL {h2}mm")
        if h1 == h2:
            print("같은 지점에서 무너진다 — 물리적 한계(그리퍼 개방)가 지배한다는 뜻이다.")
        else:
            earlier = "RL" if h2 < h1 else "PCA"
            print(f"{earlier} 이 먼저 무너진다. 그 차이가 제어 방식의 차이다.")
    return 0 if (h1 is not None or h2 is not None) else 1


if __name__ == "__main__":
    sys.exit(main())
