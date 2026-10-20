"""가림 실험 정리 — 가림 0/25/50/75% 에서 PCA(시나리오 A·B) vs PPO v2.

  $PY scripts/analyze_duck_occlusion.py \
      --aligned /data/isaac-grasp-bench/eval/duck_aligned \
      --dir /data/isaac-grasp-bench/eval/duck_occlusion --fig docs/img/duck

가림 0% 는 2차(조건 맞춤) 결과를 그대로 쓴다 — 같은 배치·같은 코드 경로.
판정 규칙은 1·2차와 같다 (실험 전에 정해 둔 규칙, 바꾸지 않는다):
  |차이| > max(PPO 시드 범위, PCA 평가 시드 범위) 일 때만 우세, 아니면 판단 보류.
"""
import argparse
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--aligned", default="/data/isaac-grasp-bench/eval/duck_aligned")
ap.add_argument("--dir", default="/data/isaac-grasp-bench/eval/duck_occlusion")
ap.add_argument("--fig", default=str(ROOT / "docs/img/duck"))
args = ap.parse_args()
A, D, F = pathlib.Path(args.aligned), pathlib.Path(args.dir), pathlib.Path(args.fig)
F.mkdir(parents=True, exist_ok=True)

for f in glob.glob("/usr/share/fonts/truetype/nanum/NanumBarunGothic*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "NanumBarunGothic", "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 12,
})
C_PCA, C_PCA_A, C_RL = "#2a78d6", "#8fb8e8", "#eb6834"
TNAME = {"head": "머리", "body": "몸통", "leg": "다리"}
PNAME = {"prone": "엎드림", "side": "옆으로"}
CELLS = [(t, p) for t in ("head", "body", "leg") for p in ("prone", "side")]
LEVELS = [0, 25, 50, 75]
SEEDS = sorted(int(p.stem.split("seed")[1]) for p in A.glob("rl_v2_seed*.csv"))


def load(level, name):
    p = (A / f"{name}.csv") if level == 0 else (D / f"{name}_occ{level}.csv")
    return pd.read_csv(p) if p.exists() else None


def seed_rates(df):
    g = df.groupby(["target", "pose", "seed"]).success.mean()
    return {c: g.loc[c].values for c in CELLS}


def verdict(pca_rates, rl_means):
    pm, rm = pca_rates.mean(), rl_means.mean()
    noise = max(np.ptp(rl_means) if len(rl_means) > 1 else np.inf, np.ptp(pca_rates))
    d = pm - rm
    if len(rl_means) < 2:
        v = "판단 보류 (시드 부족)"
    elif abs(d) > noise:
        v = "PCA 우세" if d > 0 else "PPO 우세"
    else:
        v = "판단 보류"
    return pm, rm, d, noise, v


rows, out, cats = [], ["# 가림 실험 — 가림 0/25/50/75% (인식만)\n"], []
out.append("가림판: 위에서 볼 때 수평 반평면을 덮는 불투명 판. 지시 부위의 보이는 점 중 정한 비율을 숨긴다.")
out.append("방향은 (평가 시드, env) 로 정해져 PCA·PPO 가 같은 판을 받는다. 물리 충돌은 없다.\n")
out.append(f"PPO v2 학습 시드: {SEEDS}. 학습 때는 가림이 없었다 (PCA 도 가림용 조정이 없다).\n")

# 실제 가림 비율 확인
out.append("## 1. 가림이 의도대로 걸렸는가\n")
out.append("| 가림 수준 | 실행 | 지시 부위 가림 실제 평균 | 최소 | 최대 |")
out.append("|---:|---|---:|---:|---:|")
for L in LEVELS[1:]:
    for name in ["pca_d28_c5", "rl_v2_seed1"]:
        d = load(L, name)
        if d is not None and "occ_target_frac" in d:
            out.append(f"| {L}% | {name} | {d.occ_target_frac.mean()*100:.1f}% | "
                       f"{d.occ_target_frac.min()*100:.1f}% | {d.occ_target_frac.max()*100:.1f}% |")
out.append("")

out.append("## 2. 가림 수준별 PCA vs PPO v2\n")
for L in LEVELS:
    pb, pa = load(L, "pca_d28_c5"), load(L, "pca_d15_c30")
    rl = [load(L, f"rl_v2_seed{s}") for s in SEEDS]
    rl = [r for r in rl if r is not None]
    if pb is None or not rl:
        continue
    rr = [seed_rates(r) for r in rl]
    rlm = {c: np.array([r[c].mean() for r in rr]) for c in CELLS}
    prb, pra = seed_rates(pb), (seed_rates(pa) if pa is not None else None)
    out.append(f"### 가림 {L}%  (PPO 시드 {len(rl)}개)\n")
    out.append("| 지시 부위 | 자세 | PCA B (오리 맞춤) | PCA A (상자 값) | PPO v2 | PPO v2 시드별 | B 판정 | A 판정 |")
    out.append("|---|---|---:|---:|---:|---|---|---|")
    for c in CELLS:
        b = verdict(prb[c], rlm[c])
        a = verdict(pra[c], rlm[c]) if pra is not None else (np.nan,) * 4 + ("-",)
        rows.append(dict(level=L, target=c[0], pose=c[1], pca_b=b[0], pca_a=a[0], rl=b[1],
                         rl_min=rlm[c].min(), rl_max=rlm[c].max(), verdict_b=b[4], verdict_a=a[4],
                         diff_b=b[2], noise_b=b[3]))
        out.append(f"| {TNAME[c[0]]} | {PNAME[c[1]]} | {b[0]*100:.1f}% | {a[0]*100:.1f}% | {b[1]*100:.1f}% | "
                   f"{' / '.join(f'{x*100:.0f}' for x in rlm[c])} | {b[4]} | {a[4]} |")
    out.append("")
    for name, d in [("PCA B", pb)] + [(f"PPO s{s}", r) for s, r in zip(SEEDS, rl)]:
        for t in ("head", "body", "leg"):
            x = d[d.target == t].category.value_counts(normalize=True)
            cats.append(dict(level=L, run=name, target=t, **{k: x.get(k, 0.0) for k in
                        ["success", "heuristic_error", "no_lift", "wrong_part", "unstable"]}))
S = pd.DataFrame(rows)
S.to_csv(D / "summary_occlusion.csv", index=False)

# 실패 유형 (부위별, PPO 는 시드 평균)
C = pd.DataFrame(cats)
if len(C):
    C["who"] = np.where(C.run.str.startswith("PPO"), "PPO v2", "PCA B")
    g = C.groupby(["who", "target", "level"])[["success", "heuristic_error", "no_lift", "wrong_part", "unstable"]].mean()
    out.append("## 3. 실패 유형 (부위별, 자세 합산, PPO 는 시드 평균)\n")
    out.append("| 방식 | 지시 부위 | 가림 | 성공 | 휴리스틱 거부 | 못 들어올림 | 다른 부위 | 들다 놓침 |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for (who, t, L), r in g.iterrows():
        out.append(f"| {who} | {TNAME[t]} | {L}% | {r.success*100:.1f}% | {r.heuristic_error*100:.1f}% | "
                   f"{r.no_lift*100:.1f}% | {r.wrong_part*100:.1f}% | {r.unstable*100:.1f}% |")
    out.append("")

# 그림: 부위별 성공률 vs 가림 (자세 평균)
if len(S):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharey=True)
    for ax, t in zip(axes, ("head", "body", "leg")):
        s = S[S.target == t].groupby("level")[["pca_b", "pca_a", "rl", "rl_min", "rl_max"]].mean().reset_index()
        x = s.level.values
        ax.fill_between(x, s.rl_min * 100, s.rl_max * 100, color=C_RL, alpha=0.15, lw=0)
        ax.plot(x, s.rl * 100, "-o", color=C_RL, lw=2, ms=8, label="PPO v2 (시드 평균, 띠: 최소~최대)")
        ax.plot(x, s.pca_b * 100, "-o", color=C_PCA, lw=2, ms=8, label="PCA B (오리 맞춤)")
        ax.plot(x, s.pca_a * 100, "--o", color=C_PCA_A, lw=2, ms=8, label="PCA A (상자 값)")
        ax.set_xticks(LEVELS, [f"{v}%" for v in LEVELS])
        ax.set_xlabel("지시 부위 가림 비율")
        ax.set_ylim(-3, 105)
        ax.set_title(TNAME[t], loc="left", fontsize=13)
    axes[0].set_ylabel("성공률 (%, 두 자세 평균)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), frameon=False, ncol=3)
    fig.suptitle("가림 실험 — 두 방식이 같은 가림판을 받는다 (인식만, 물리 충돌 없음)", x=0.01, ha="left", fontsize=13)
    fig.subplots_adjust(top=0.76, bottom=0.14, wspace=0.06, left=0.06, right=0.99)
    fig.savefig(F / "fig7_occlusion.png", dpi=200)
    plt.close(fig)

(D / "summary_occlusion.md").write_text("\n".join(out))
print("\n".join(out))
