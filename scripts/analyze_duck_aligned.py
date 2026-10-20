"""조건 맞추기 2차 실험 정리 — 배치 동일성 검증, 시나리오 A/B 판정, PPO v0/v2.

  $PY scripts/analyze_duck_aligned.py --dir /data/isaac-grasp-bench/eval/duck_aligned --fig docs/img/duck

시나리오
  A (둘 다 상자 설정) : PCA 깊이 15 · 바닥 여유 30   vs PPO v2
  A' (상자 보정값)    : PCA 깊이 15 · 바닥 여유 10   vs PPO v2
  B (PCA 만 오리 맞춤): PCA 깊이 28 · 바닥 여유 5    vs PPO v2
PPO v2 = 관측을 PCA 와 같은 '보이는 점군'으로 제한하고, 보상을 손가락 접촉 판정으로 맞춘 정책.
판정 규칙은 1차와 같다: |차이| > max(PPO 시드 범위, PCA 평가 시드 범위) 일 때만 우세.
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import glob  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/data/isaac-grasp-bench/eval/duck_aligned")
ap.add_argument("--fig", default=str(ROOT / "docs/img/duck"))
args = ap.parse_args()
D, F = pathlib.Path(args.dir), pathlib.Path(args.fig)
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
C_PCA, C_RL = "#2a78d6", "#eb6834"
TNAME = {"head": "머리", "body": "몸통", "leg": "다리"}
PNAME = {"prone": "엎드림", "side": "옆으로"}
CELLS = [(t, p) for t in ("head", "body", "leg") for p in ("prone", "side")]
KEY = ["target", "pose", "seed", "env"]
out = ["# 조건 맞추기 2차 — 시나리오 A/B, PPO v2\n"]


def load(name):
    p = D / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


runs = {n: load(n) for n in ["pca_d28_c5", "pca_d28_c5_taskv2", "pca_d15_c30", "pca_d15_c10",
                             "ctrl_wrong_part", "ctrl_offset",
                             "rl_v0_seed1", "rl_v0_seed2", "rl_v0_seed3"]
        + sorted(p.stem for p in D.glob("rl_v2_seed*.csv"))}
runs = {k: v for k, v in runs.items() if v is not None}

# ── 1. 배치 동일성 ───────────────────────────────────────────
ref_name = "pca_d28_c5"
ref = runs[ref_name].set_index(KEY).sort_index()
out.append("## 1. 물체 배치가 정말 같은가\n")
out.append(f"기준: `{ref_name}`. 같은 (지시 부위, 자세, 평가 시드, env) 의 자세를 비교했다.\n")
out.append("| 실행 | 스폰 위치 최대 차 (mm) | 스폰 자세 최대 차 (쿼터니언) | 안착 위치 최대 차 (mm) |")
out.append("|---|---:|---:|---:|")
sp = ["spawn_x", "spawn_y", "spawn_z"]
sq = ["spawn_qw", "spawn_qx", "spawn_qy", "spawn_qz"]
st = ["settled_x", "settled_y", "settled_z"]
for n, d in runs.items():
    if n == ref_name:
        continue
    x = d.set_index(KEY).sort_index().reindex(ref.index)
    dp = np.abs(x[sp].values - ref[sp].values).max() * 1000
    dq = np.abs(x[sq].values - ref[sq].values).max()
    ds = np.abs(x[st].values - ref[st].values).max() * 1000
    out.append(f"| {n} | {dp:.3f} | {dq:.6f} | {ds:.3f} |")
out.append("")

# PCA v0 태스크 vs v2 태스크 — 물리가 같으면 결과가 같아야 한다
if "pca_d28_c5_taskv2" in runs:
    a = runs["pca_d28_c5"].set_index(KEY).sort_index()
    b = runs["pca_d28_c5_taskv2"].set_index(KEY).sort_index().reindex(a.index)
    same = (a.success.values == b.success.values).mean()
    out.append(f"PCA 를 v0 태스크와 v2 태스크에서 돌린 결과의 에피소드별 일치율: **{same*100:.2f}%** "
               f"(성공률 {a.success.mean()*100:.1f}% vs {b.success.mean()*100:.1f}%)\n")


# ── 2. 시나리오 판정 ─────────────────────────────────────────
def seed_rates(df):
    g = df.groupby(["target", "pose", "seed"]).success.mean()
    return {c: g.loc[c].values for c in CELLS}


def rl_block(prefix):
    ds = [runs[k] for k in sorted(runs) if k.startswith(prefix)]
    rr = [seed_rates(d) for d in ds]
    return {c: np.array([r[c].mean() for r in rr]) for c in CELLS}, len(ds)


rl2, n2 = rl_block("rl_v2_")
rl0, n0 = rl_block("rl_v0_")


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


SCEN = [("A", "pca_d15_c30", "둘 다 상자 설정 (PCA 깊이 15 · 바닥 여유 30)"),
        ("A'", "pca_d15_c10", "상자 보정값 (PCA 깊이 15 · 바닥 여유 10)"),
        ("B", "pca_d28_c5", "PCA 만 오리 형상에 맞춤 (깊이 28 · 바닥 여유 5)")]
summary = []
out.append("## 2. 시나리오별 PCA vs PPO v2\n")
out.append(f"PPO v2 = 관측을 PCA 와 같은 보이는 점군으로 제한 + 손가락 접촉 보상, 학습 시드 {n2}개.\n")
for tag, name, desc in SCEN:
    if name not in runs or n2 == 0:
        continue
    pr = seed_rates(runs[name])
    out.append(f"### 시나리오 {tag} — {desc}\n")
    out.append("| 지시 부위 | 자세 | PCA | PPO v2 | PPO v2 시드별 | 차이 | 노이즈 | 판정 |")
    out.append("|---|---|---:|---:|---|---:|---:|---|")
    for c in CELLS:
        pm, rm, d, noise, v = verdict(pr[c], rl2[c])
        summary.append(dict(scenario=tag, target=c[0], pose=c[1], pca=pm, pca_min=pr[c].min(), pca_max=pr[c].max(),
                            rl=rm, rl_min=rl2[c].min(), rl_max=rl2[c].max(), diff=d, noise=noise, verdict=v))
        out.append(f"| {TNAME[c[0]]} | {PNAME[c[1]]} | {pm*100:.1f}% | {rm*100:.1f}% | "
                   f"{' / '.join(f'{x*100:.0f}' for x in rl2[c])} | {d*100:+.1f}%p | {noise*100:.1f}%p | {v} |")
    out.append("")
S = pd.DataFrame(summary)
S.to_csv(D / "summary_aligned.csv", index=False)

# ── 3. PPO v0 vs v2 ──────────────────────────────────────────
if n0 and n2:
    out.append("## 3. PPO 관측을 PCA 와 맞췄을 때 (v0 → v2)\n")
    out.append("| 지시 부위 | 자세 | PPO v0 (실제 위치·자세 관측) | PPO v2 (보이는 점군만) |")
    out.append("|---|---|---:|---:|")
    for c in CELLS:
        out.append(f"| {TNAME[c[0]]} | {PNAME[c[1]]} | {rl0[c].mean()*100:.1f}% ({' / '.join(f'{x*100:.0f}' for x in rl0[c])}) | "
                   f"{rl2[c].mean()*100:.1f}% ({' / '.join(f'{x*100:.0f}' for x in rl2[c])}) |")
    out.append("")

# ── 4. 대조군 ────────────────────────────────────────────────
ctrl = [("PCA 정상 (B)", runs.get("pca_d28_c5")), ("다른 부위 지시", runs.get("ctrl_wrong_part")),
        ("20cm 어긋남", runs.get("ctrl_offset"))]
out.append("## 4. 실패 대조군\n")
out.append("| 조건 | 들어올림 (부위 무관) | 성공 (지시 부위) |")
out.append("|---|---:|---:|")
for n, d in ctrl:
    if d is not None:
        out.append(f"| {n} | {d.category.isin(['success','wrong_part']).mean()*100:.1f}% | {d.success.mean()*100:.1f}% |")
out.append("")

# ── 그림: 시나리오 A vs B ────────────────────────────────────
if len(S):
    tags = [t for t in ["A", "B"] if t in set(S.scenario)]
    fig, axes = plt.subplots(1, len(tags), figsize=(6.6 * len(tags), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    titles = {"A": "A. 둘 다 상자 설정 그대로", "B": "B. PCA 만 오리 형상에 맞춤"}
    for ax, tag in zip(axes, tags):
        s = S[S.scenario == tag].reset_index(drop=True)
        x = np.arange(len(s))
        w = 0.36
        for k, (key, col, lab) in enumerate([("pca", C_PCA, "PCA"), ("rl", C_RL, "PPO v2")]):
            v = s[key].values * 100
            lo, hi = v - s[f"{key}_min"].values * 100, s[f"{key}_max"].values * 100 - v
            xs = x + (k - 0.5) * w
            ax.bar(xs, v, w - 0.04, color=col, label=lab, zorder=2)
            ax.errorbar(xs, v, yerr=[lo, hi], fmt="none", ecolor="#0b0b0b", elinewidth=1.1, capsize=3, zorder=3)
            for xi, vi, ti in zip(xs, v, s[f"{key}_max"].values * 100):
                ax.text(xi, max(vi, ti) + 2, f"{vi:.0f}", ha="center", va="bottom", fontsize=9, color="#52514e")
        ax.set_xticks(x, [f"{TNAME[t]}\n{PNAME[p]}" for t, p in zip(s.target, s.pose)], fontsize=10)
        for xi, vd in zip(x, s.verdict):
            short = vd.replace(" 우세", "↑").replace("판단 보류", "보류")
            ax.text(xi, -22, short, ha="center", va="top", fontsize=10,
                    color="#0b0b0b" if "↑" in short else "#8a8984", fontweight="bold" if "↑" in short else None)
        ax.set_ylim(0, 110)
        ax.set_title(titles[tag], loc="left", fontsize=13)
    axes[0].set_ylabel("성공률 (%)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), frameon=False, ncol=2)
    fig.suptitle("조건을 맞춘 비교 — PPO 는 PCA 와 같은 '보이는 점군'만 관측 (오차막대: 최소~최대)",
                 x=0.01, ha="left", fontsize=13)
    fig.subplots_adjust(bottom=0.22, top=0.80, wspace=0.06)
    fig.savefig(F / "fig6_aligned_scenarios.png", dpi=200)
    plt.close(fig)

(D / "summary_aligned.md").write_text("\n".join(out))
print("\n".join(out))
