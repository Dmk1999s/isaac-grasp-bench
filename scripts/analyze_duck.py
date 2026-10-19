"""오리 '지시 부위 파지' 결과 정리 — 표(summary.md)와 PPT 용 그림.

  $PY scripts/analyze_duck.py --dir /data/isaac-grasp-bench/eval/duck --fig docs/img/duck

입력 (없는 파일은 건너뛴다)
  pca.csv                 PCA 본실험 (depth 28, 고무 근사)
  pca_rigid.csv           같은 조건, 강체 오리 — 고무 근사의 효과
  rl_seed{N}.csv          PPO 시드별
  ctrl_wrong_part.csv     대조군: PCA 에 다른 부위를 넘김
  ctrl_offset.csv         대조군: 파지 위치 20cm 어긋남
  depth_sweep/depth_*.csv PCA 깊이 훑기

판정 (원칙 3, README 와 같은 규칙)
  노이즈 = max(PPO 시드 평균들의 범위, PCA 평가 시드별 성공률의 범위)
  |PCA - PPO| > 노이즈 → 우세 주장, 아니면 '판단 보류'
"""
import argparse
import glob
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/data/isaac-grasp-bench/eval/duck")
ap.add_argument("--fig", default=str(ROOT / "docs/img/duck"))
args = ap.parse_args()
D = pathlib.Path(args.dir)
F = pathlib.Path(args.fig)
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
C_PCA, C_RL = "#2a78d6", "#eb6834"                    # 범주 1, 2 (검증된 순서)
C_TGT = {"head": "#1baf7a", "body": "#eda100", "leg": "#e87ba4"}
CAT = [("success", "성공", "#1baf7a"), ("wrong_part", "다른 부위", "#e87ba4"),
       ("unstable", "들다 놓침", "#eda100"), ("no_lift", "못 들어올림", "#6250d6"),
       ("heuristic_error", "PCA 계산 실패", "#8a8984")]
TNAME = {"head": "머리", "body": "몸통", "leg": "다리"}
PNAME = {"prone": "엎드림", "side": "옆으로"}
CELLS = [(t, p) for t in ("head", "body", "leg") for p in ("prone", "side")]


def load(name):
    p = D / name
    return pd.read_csv(p) if p.exists() else None


def seed_rates(df):
    """셀별 평가 시드마다의 성공률 → {cell: array}"""
    g = df.groupby(["target", "pose", "seed"]).success.mean()
    return {c: g.loc[c].values for c in CELLS if c in g.index.droplevel(2)}


out = ["# 오리 '지시 부위 파지' — PCA vs PPO\n"]

# ── 1. 본 비교 ───────────────────────────────────────────────
pca = load("pca.csv")
rl_files = sorted(D.glob("rl_seed*.csv"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
rls = {int(re.findall(r"\d+", p.stem)[0]): pd.read_csv(p) for p in rl_files}
rows = []
if pca is not None:
    pr = seed_rates(pca)
    rr = {s: seed_rates(d) for s, d in rls.items()}
    for c in CELLS:
        pm = pr[c].mean()
        rl_means = np.array([rr[s][c].mean() for s in rr]) if rr else np.array([])
        rm = rl_means.mean() if len(rl_means) else np.nan
        noise = max(np.ptp(rl_means) if len(rl_means) > 1 else np.inf, np.ptp(pr[c]))
        diff = pm - rm
        if not len(rl_means):
            verdict = "PPO 없음"
        elif len(rl_means) < 2:
            verdict = "판단 보류 (시드 1개)"
        elif abs(diff) > noise:
            verdict = "PCA 우세" if diff > 0 else "PPO 우세"
        else:
            verdict = "판단 보류"
        rows.append(dict(target=c[0], pose=c[1], pca=pm, pca_min=pr[c].min(), pca_max=pr[c].max(),
                         rl=rm, rl_min=rl_means.min() if len(rl_means) else np.nan,
                         rl_max=rl_means.max() if len(rl_means) else np.nan,
                         rl_seeds=" / ".join(f"{x*100:.0f}" for x in rl_means),
                         diff=diff, noise=noise, verdict=verdict))
    S = pd.DataFrame(rows)
    S.to_csv(D / "summary.csv", index=False)
    n_ep = len(pca) // len(CELLS)
    out.append(f"셀당 PCA {n_ep} 에피소드 (평가 시드 {pca.seed.nunique()}개 × env {pca.env.nunique()}개), "
               f"PPO 학습 시드 {len(rls)}개 × 같은 평가 조건.\n")
    out.append("| 지시 부위 | 자세 | PCA | PPO (시드 평균) | PPO 시드별 | 차이 | 노이즈 | 판정 |")
    out.append("|---|---|---:|---:|---|---:|---:|---|")
    for r in rows:
        out.append(f"| {TNAME[r['target']]} | {PNAME[r['pose']]} | {r['pca']*100:.1f}% | "
                   f"{r['rl']*100:.1f}% | {r['rl_seeds']} | {r['diff']*100:+.1f}%p | "
                   f"{r['noise']*100:.1f}%p | {r['verdict']} |")
    out.append("")

    # 그림: 비교 막대
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(CELLS))
    w = 0.36
    for k, (key, col, lab) in enumerate([("pca", C_PCA, "PCA (규칙 기반)"), ("rl", C_RL, "PPO (강화학습)")]):
        v = S[key].values * 100
        lo = v - S[f"{key}_min"].values * 100
        hi = S[f"{key}_max"].values * 100 - v
        xs = x + (k - 0.5) * w
        ax.bar(xs, v, w - 0.04, color=col, label=lab, zorder=2)
        ax.errorbar(xs, v, yerr=[lo, hi], fmt="none", ecolor="#0b0b0b", elinewidth=1.2, capsize=4, zorder=3)
        top = S[f"{key}_max"].values * 100
        for xi, vi, ti in zip(xs, v, top):
            if np.isfinite(vi):
                ax.text(xi, max(vi, ti) + 2, f"{vi:.0f}", ha="center", va="bottom", fontsize=10, color="#52514e")
    ax.set_xticks(x, [f"{TNAME[t]}\n{PNAME[p]}" for t, p in CELLS])
    for xi, vd in zip(x, S.verdict):
        ax.text(xi, -24, vd, ha="center", va="top", fontsize=10,
                color="#0b0b0b" if "우세" in vd else "#8a8984", fontweight="bold" if "우세" in vd else None)
    ax.set_ylim(0, 110)
    ax.set_ylabel("성공률 (%)")
    ax.set_title("지시한 부위를 잡아 들어올린 비율 — 오차막대: 최소~최대 (판정 척도와 같음)", loc="left",
                 fontsize=13, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=2, borderaxespad=0.2)
    fig.subplots_adjust(bottom=0.24, top=0.82)
    fig.savefig(F / "fig3_compare.png", dpi=200)
    plt.close(fig)

    # 그림: 실패 유형
    groups = [("PCA", pca)] + [(f"PPO s{s}", d) for s, d in rls.items()]
    fig, axes = plt.subplots(1, len(CELLS), figsize=(13, 4.6), sharey=True)
    for ax, c in zip(axes, CELLS):
        for gi, (gname, df) in enumerate(groups):
            sub = df[(df.target == c[0]) & (df.pose == c[1])]
            left = 0.0
            for key, lab, col in CAT:
                share = (sub.category == key).mean() * 100
                if share > 0:
                    ax.barh(gi, share, left=left, color=col, height=0.7,
                            edgecolor="#fcfcfb", linewidth=1.5, label=lab)
                left += share
        ax.set_title(f"{TNAME[c[0]]} · {PNAME[c[1]]}", fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100], ["0", "50", "100%"], fontsize=10)
        ax.set_yticks(range(len(groups)), [g for g, _ in groups])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    order = [lab for _, lab, _ in CAT if lab in seen]
    fig.legend([seen[k] for k in order], order, loc="lower center", ncol=len(order), frameon=False)
    fig.suptitle("결과 유형 (%) — '다른 부위' = 들어올렸지만 지시와 다른 곳을 잡음", x=0.01, ha="left", fontsize=13)
    fig.subplots_adjust(bottom=0.2, top=0.82, wspace=0.22)
    fig.savefig(F / "fig4_failures.png", dpi=200)
    plt.close(fig)

    # 실패 사례 표 (PPT 본문용)
    out.append("### 결과 유형 (에피소드 수)\n")
    allc = pd.concat([d.assign(who=g) for g, d in groups], ignore_index=True)
    tab = pd.crosstab([allc.who, allc.target, allc.pose], allc.category)
    out.append("```\n" + tab.to_string() + "\n```\n")

# ── 2. 대조군 ─────────────────────────────────────────────────
ctrl = [("PCA 정상", pca), ("대조군: 다른 부위 지시", load("ctrl_wrong_part.csv")),
        ("대조군: 20cm 어긋남", load("ctrl_offset.csv"))]
ctrl = [(n, d) for n, d in ctrl if d is not None]
if len(ctrl) > 1:
    out.append("### 실패 대조군 — 판정이 '틀린 파지'를 실패로 세는가\n")
    out.append("| 조건 | 들어올려 0.5s 유지 (부위 무관) | 성공 (지시 부위) |")
    out.append("|---|---:|---:|")
    vals = []
    for n, d in ctrl:
        lifted = d.category.isin(["success", "wrong_part"]).mean()
        succ = d.success.mean()
        vals.append((n, lifted, succ))
        out.append(f"| {n} | {lifted*100:.1f}% | {succ*100:.1f}% |")
    out.append("")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(vals))
    ax.bar(x - 0.19, [v[1] * 100 for v in vals], 0.36, color="#8a8984", label="들어올림 (부위 무관)")
    ax.bar(x + 0.19, [v[2] * 100 for v in vals], 0.36, color=C_PCA, label="성공 (지시 부위 + 유지)")
    for xi, v in zip(x, vals):
        ax.text(xi - 0.19, v[1] * 100 + 2, f"{v[1]*100:.0f}", ha="center", fontsize=10, color="#52514e")
        ax.text(xi + 0.19, v[2] * 100 + 2, f"{v[2]*100:.0f}", ha="center", fontsize=10, color="#52514e")
    ax.set_xticks(x, [v[0] for v in vals])
    ax.set_ylim(0, 112)
    ax.set_ylabel("%")
    ax.set_title("실패 대조군 — 들어올려도 엉뚱한 부위면 실패로 센다", loc="left", fontsize=13)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(F / "fig5_controls.png", dpi=200)
    plt.close(fig)

# ── 3. 깊이 훑기 ──────────────────────────────────────────────
sweep = sorted(D.glob("depth_sweep/depth_*.csv"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
if sweep:
    rows = []
    for p in sweep:
        d = pd.read_csv(p)
        dep = int(re.findall(r"\d+", p.stem)[0])
        for t, g in d.groupby("target"):
            rows.append(dict(depth=dep, target=t, rate=g.success.mean()))
    W = pd.DataFrame(rows)
    out.append("### PCA 파지 깊이 훑기 (자세 2개 평균)\n")
    piv = W.pivot(index="target", columns="depth", values="rate") * 100
    out.append("| 부위 \\ 깊이(mm) | " + " | ".join(str(c) for c in piv.columns) + " |")
    out.append("|---|" + "---:|" * len(piv.columns))
    for t in ("head", "body", "leg"):
        if t in piv.index:
            out.append(f"| {TNAME[t]} | " + " | ".join(f"{v:.0f}%" for v in piv.loc[t]) + " |")
    out.append("")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for t in ("head", "body", "leg"):
        g = W[W.target == t].sort_values("depth")
        ax.plot(g.depth, g.rate * 100, color=C_TGT[t], lw=2, marker="o", ms=8, label=TNAME[t])
    for xv, txt, yv in [(25.1, "엎드린 몸통\n예측 25", 6), (27.0, "옆 몸통\n예측 27", 30)]:
        ax.axvline(xv, color="#52514e", ls="--", lw=1.0)
        ax.text(xv + 0.3, yv, txt, fontsize=9, color="#52514e")
    ax.axvspan(28 - 0.4, 28 + 0.4, color="#2a78d6", alpha=0.12)
    ax.text(28.5, 55, "본실험 28", fontsize=10, color="#2a78d6")
    ax.axvline(15, color="#c9c8c2", ls=":", lw=1.2)
    ax.text(15.3, 92, "실기 원래 값", fontsize=10, color="#8a8984")
    ax.set_xlabel("PCA 파지 깊이 grasp_depth_mm")
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("둥근 부위는 '가장 넓은 단면'보다 위를 쥐면 튕겨 나간다 (점선: 형상 계산 예측)", loc="left", fontsize=13)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(F / "fig2_depth_sweep.png", dpi=200)
    plt.close(fig)

# ── 4. 고무 근사 효과 ─────────────────────────────────────────
rig = load("pca_rigid.csv")
if pca is not None and rig is not None:
    out.append("### 고무 근사(부드러운 접촉) vs 강체 — PCA 같은 조건\n")
    out.append("| 부위 | 자세 | 성공 (고무) | 성공 (강체) | 손가락 간격 (고무) | (강체) | 눌림 |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for c in CELLS:
        a = pca[(pca.target == c[0]) & (pca.pose == c[1])]
        b = rig[(rig.target == c[0]) & (rig.pose == c[1])]
        ga, gb = a.finger_gap_mm.median(), b.finger_gap_mm.median()
        out.append(f"| {TNAME[c[0]]} | {PNAME[c[1]]} | {a.success.mean()*100:.1f}% | {b.success.mean()*100:.1f}% | "
                   f"{ga:.1f} | {gb:.1f} | {gb-ga:.1f} mm |")
    out.append("")

# ── 5. 모형 그림 (사진 vs 시뮬레이터) ───────────────────────
photo = ROOT / "assets/duck/reference_photo.png"
mont = ROOT / "eval/duck_asset_check/gpu/montage.png"
if photo.exists() and mont.exists():
    from PIL import Image
    m = Image.open(mont)
    rgb, lab = m.crop((0, 0, 240, 240)), m.crop((0, 240, 240, 480))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, im, t in zip(axes, [Image.open(photo), rgb, lab],
                         ["팀 프로젝트 사진 (부위 마스크)", "Isaac Sim 근사 모형", "카메라 부위 라벨 (머리·몸통·다리)"]):
        ax.imshow(im)
        ax.set_title(t, fontsize=12)
        ax.axis("off")
    fig.text(0.01, 0.02, "근사 모형: 사진 비율 + 가정 치수(길이 144mm, 60g, 고무 근사 접촉). 실측·3D 스캔 아님.",
             fontsize=10, color="#52514e")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(F / "fig1_model.png", dpi=200)
    plt.close(fig)

(D / "summary.md").write_text("\n".join(out))
print("\n".join(out))
print(f"\n그림: {F}")
