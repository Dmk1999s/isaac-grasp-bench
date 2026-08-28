"""README 용 그림 생성. Isaac Sim 없이 돈다 (CSV 만 읽는다).

원칙: 측정하지 않은 것을 그리지 않는다.
      반복 편차를 오차막대로 항상 같이 그린다 — 차이가 편차보다 작으면
      그림만 보고도 '판단 보류'임이 드러나야 한다.
"""
import argparse
import csv
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 한글 라벨이 두부(□)로 깨지지 않게 나눔고딕을 등록한다.
# DejaVu Sans(기본)에는 한글 글리프가 없다.
for _fp in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"):
    if pathlib.Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False   # 마이너스 기호가 깨지는 것 방지

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

# 색: 두 방식을 일관되게. 색약 고려해 명도도 다르게 잡는다.
C_PCA, C_RL = "#1f77b4", "#d62728"


def rates_by_repeat(path):
    """CSV -> 반복별 성공률 배열."""
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["repeat"], []).append(int(r["success"]))
    return np.array([np.mean(v) for v in by.values()]) * 100.0


def fig_reward_terms(run_csv, out):
    rows = list(csv.DictReader(open(run_csv)))
    it = np.array([int(r["iteration"]) for r in rows])
    names = [k for k in rows[0] if k != "iteration"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in names:
        v = np.array([float(r[n]) for r in rows])
        ax.plot(it, v, label=n, lw=1.6)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("에피소드 보상 (항목별)")
    ax.set_title("보상 항목별 기여 — 학습이 단계적으로 진행된다")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"생성: {out}")


def fig_failure_map(out, seed_glob="eval/gridseed_{label}_rl_s*.csv"):
    cells = [("base", "42×42×42\n(기준)"), ("elong15", "63×42×42"),
             ("elong20", "84×42×42"), ("wide12", "42×63×42"),
             ("wide16", "42×84×42"), ("small", "26×26×26")]
    labels, pca_m, pca_e, rl_m, rl_e = [], [], [], [], []
    for key, disp in cells:
        p = ROOT / f"eval/grid_{key}_pca.csv"
        if not p.exists():
            continue
        pr = rates_by_repeat(p)
        # RL 은 시드별 CSV 를 모아 '시드 간' 편차를 낸다
        seed_files = sorted(glob.glob(str(ROOT / seed_glob.format(label=key))))
        if not seed_files:
            continue
        per_seed = np.array([rates_by_repeat(f).mean() for f in seed_files])
        labels.append(disp)
        pca_m.append(pr.mean()); pca_e.append(pr.std(ddof=1) if len(pr) > 1 else 0.0)
        rl_m.append(per_seed.mean())
        rl_e.append(per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0)

    if not labels:
        print("실패 지도용 데이터가 아직 없다 — 건너뜀")
        return

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, pca_m, w, yerr=pca_e, capsize=4, label="PCA 휴리스틱", color=C_PCA)
    ax.bar(x + w/2, rl_m, w, yerr=rl_e, capsize=4, label=f"PPO 정책 (시드 {len(seed_files)}개)", color=C_RL)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("성공률 (%)"); ax.set_ylim(0, 108)
    ax.set_title("물체 격자 실패 지도 — 오차막대는 반복/시드 편차")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"생성: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["rewards", "map", "all"], default="all")
    a = ap.parse_args()
    if a.only in ("rewards", "all"):
        fig_reward_terms(ROOT / "eval/reward_terms_seed1.csv", IMG / "reward_terms.png")
    if a.only in ("map", "all"):
        fig_failure_map(IMG / "failure_map.png")
