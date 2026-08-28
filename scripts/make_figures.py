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


def fig_width_sweep(out):
    """닫힘축 폭 스윕 — 실패가 시작되는 지점을 관측에서 보인다.

    6셀 격자는 성기어서 RL 이 81% 에서 46% 로 떨어지는 사이가 비어 있었다.
    폭을 촘촘히 훑으면 어디서 꺾이는지가 곡선으로 드러난다.
    임계값을 미리 정하지 않는다(원칙 4) — 그림은 관측만 그린다.

    시드는 평균으로 뭉개지 않고 개별 곡선도 함께 그린다.
    시드 간 편차가 폭에 따른 변화만큼 크기 때문에, 평균만 그리면
    '학습 방식의 성질'처럼 보이는 착시가 생긴다.
    """
    # y 스케일 -> 실제 닫힘축 폭(mm). 기준 큐브 한 변 52.5mm.
    scales = [1.0, 1.2, 1.35, 1.5, 1.65, 1.8]
    widths, pca_m, pca_e, seed_curves = [], [], [], {}
    for w in scales:
        tag = str(w).replace(".", "")
        widths.append(52.5 * w)
        f = ROOT / f"eval/width_w{tag}_pca.csv"
        r = rates_by_repeat(f) if f.exists() else np.array([np.nan])
        pca_m.append(r.mean()); pca_e.append(r.max() - r.min())
        for sd in (1, 2, 3, 4):
            fs = ROOT / f"eval/width_w{tag}_rl_s{sd}.csv"
            if fs.exists():
                seed_curves.setdefault(sd, []).append(rates_by_repeat(fs).mean())
            else:
                seed_curves.setdefault(sd, []).append(np.nan)

    rl_mat = np.array([seed_curves[sd] for sd in sorted(seed_curves)])
    rl_m = np.nanmean(rl_mat, axis=0)
    rl_e = np.nanmax(rl_mat, axis=0) - np.nanmin(rl_mat, axis=0)   # 시드 간 폭

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.errorbar(widths, pca_m, yerr=pca_e, color=C_PCA, marker="o", lw=2.2,
                capsize=4, label="PCA 휴리스틱 (반복 편차)")
    ax.errorbar(widths, rl_m, yerr=rl_e / 2, color=C_RL, marker="s", lw=2.2,
                capsize=4, label="PPO 정책 — 시드 4개 평균 (막대=시드 범위)")
    for sd in sorted(seed_curves):
        ax.plot(widths, seed_curves[sd], color=C_RL, alpha=0.35, lw=1.0,
                ls="--", marker=".", ms=4)

    # 그리퍼 개방 한계는 '설계 사양'이지 관측된 실패 시작점이 아니다 — 구분해 표시한다.
    ax.axvline(80.0, color="#8e8e93", ls=":", lw=1.4)
    ax.text(80.6, 8, "Franka 개방 한계 80mm\n(사양, 관측값 아님)",
            fontsize=8, color="#8e8e93", va="bottom")
    # check_criteria.py 가 관측에서 도출한 실패 시작 폭.
    ax.axvline(70.9, color=C_RL, ls="-.", lw=1.4, alpha=0.8)
    ax.text(70.3, 96, "도출된 RL 실패 시작 70.9mm", fontsize=8,
            color=C_RL, ha="right", va="top", weight="bold")

    ax.set_xlabel("물체 닫힘축 폭 (mm)")
    ax.set_ylabel("성공률 (%)")
    ax.set_title("닫힘축 폭 스윕 — 실패는 어디서 시작되는가")
    ax.set_ylim(0, 105); ax.grid(alpha=0.25); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"생성: {out}")


def fig_dr(out):
    """Phase 3 — 랜덤화 강도의 효과와, 그보다 큰 것.

    왼쪽: 강도별 곡선. 세 곡선이 시드 편차 안에서 겹친다 — 회복은 없다.
    오른쪽: 시드 간 편차. 마찰을 고정한 것만으로 편차가 한 자릿수로 떨어진다.
            평균이 아니라 **분산**에 대한 관찰이라 따로 그린다.
    """
    scales = [("1.0", "10"), ("1.2", "12"), ("1.35", "135"), ("1.5", "15"), ("1.8", "18")]
    widths = [52.5 * float(sc) for sc, _ in scales]

    def seed_means(pattern, seeds):
        out_ = []
        for _, tag in scales:
            vals = []
            for sd_ in seeds:
                f = ROOT / pattern.format(tag=tag, s=sd_)
                if f.exists():
                    vals.append(rates_by_repeat(f).mean())
            out_.append(vals)
        return out_

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5))

    colors = {"000": "#1f77b4", "050": "#ff7f0e", "100": "#2ca02c"}
    for stg in ("000", "050", "100"):
        per_w = seed_means(f"eval/dr{stg}_w{{tag}}_s{{s}}.csv", (1, 2, 3))
        m = [np.mean(v) if v else np.nan for v in per_w]
        e = [(np.max(v) - np.min(v)) / 2 if len(v) > 1 else 0 for v in per_w]
        axL.errorbar(widths, m, yerr=e, marker="o", lw=2, capsize=4,
                     color=colors[stg], label=f"DR{stg} (강도 {int(stg)/100:.1f})")
    axL.axvline(70.9, color="#8e8e93", ls="-.", lw=1.2)
    axL.text(70.3, 20, "RL 실패 시작 70.9mm", fontsize=8, color="#8e8e93",
             ha="right", rotation=90, va="bottom")
    axL.set_xlabel("물체 닫힘축 폭 (mm)"); axL.set_ylabel("성공률 (%)")
    axL.set_title("랜덤화 강도별 성공률 — 세 곡선이 편차 안에서 겹친다")
    axL.set_ylim(0, 105); axL.grid(alpha=0.25); axL.legend(fontsize=9)

    orig = seed_means("eval/width_w{tag}_rl_s{s}.csv", (1, 2, 3, 4))
    dr0 = seed_means("eval/dr000_w{tag}_s{s}.csv", (1, 2, 3))
    o_sd = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in orig]
    d_sd = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in dr0]
    x = np.arange(len(widths)); w = 0.38
    axR.bar(x - w / 2, o_sd, w, color="#d62728", label="원래 정책 (마찰 미지정, 시드 4)")
    axR.bar(x + w / 2, d_sd, w, color="#1f77b4", label="DR000 (마찰 0.8 고정, 시드 3)")
    axR.set_xticks(x); axR.set_xticklabels([f"{v:.1f}" for v in widths])
    axR.set_xlabel("물체 닫힘축 폭 (mm)"); axR.set_ylabel("시드 간 표준편차 (%p)")
    axR.set_title("시드 편차 — 마찰을 고정하자 8~10배 줄었다")
    axR.grid(axis="y", alpha=0.25); axR.legend(fontsize=9)

    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"생성: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["rewards", "map", "width", "dr", "all"], default="all")
    a = ap.parse_args()
    if a.only in ("rewards", "all"):
        fig_reward_terms(ROOT / "eval/reward_terms_seed1.csv", IMG / "reward_terms.png")
    if a.only in ("map", "all"):
        fig_failure_map(IMG / "failure_map.png")
    if a.only in ("width", "all"):
        fig_width_sweep(IMG / "width_sweep.png")
    if a.only in ("dr", "all"):
        fig_dr(IMG / "dr_effect.png")
