"""휴리스틱이 파지 축을 어떻게 잡는지, 그리고 어디서 무너지는지를 그림으로 남긴다.

실기(두산 M0609 + RG2)에서는 YOLO 파트 분할 점군에 PCA 를 걸어 장축을 뽑았다.
여기서는 같은 알고리즘에 합성 점군(`src/synthetic_cloud.py`)을 먹인다 —
RTX 렌더러가 이 인스턴스에서 초기화되지 않아 뎁스 카메라를 못 쓰기 때문이고
(docs/WORKLOG.md 4번), 실험 설계상으로도 그게 맞다(관측 양식을 양쪽에 맞춘다).

그림이 보여주는 것은 성공 사례가 아니라 **실패 지도의 원인**이다:
  - 26mm 물체에서 손끝 목표가 물체 윗면보다 위에 선다 (FLOOR_CLEARANCE_MM)
  - 84mm 물체에서 닫힘축 폭이 그리퍼 한계를 넘는다 (GraspTooWideError)

Isaac Sim 없이 돈다 — numpy/matplotlib 만 쓴다.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import grasp_pca_core as heuristic  # noqa: E402
import synthetic_cloud as sc  # noqa: E402

BASE_CUBE_MM = 52.5           # run_eval.py 와 같은 기준 치수
OBJ_POS_M = (0.45, 0.0, 0.0)  # 대표 자세. 바닥 위에 놓인 것으로 둔다.
N_POINTS = 600
SEED = 7

# run_grid.sh 의 셀과 같은 스케일이어야 실패 지도와 대응된다.
# 판정은 eval/grid_summary.txt 의 측정값이다 (셀당 5회 x 32env = 160 에피소드).
CELLS = [
    ("기준 42x42mm",   (0.8, 0.8, 0.8), "PCA 100% / RL 100% (천장)"),
    ("길쭉 63x42mm",   (1.2, 0.8, 0.8), "PCA 100% / RL 100% (천장)"),
    ("넓음 42x84mm",   (0.8, 1.6, 0.8), "PCA 100% / RL 45.6%"),
    ("작음 26x26mm",   (0.5, 0.5, 0.5), "PCA 1.9% / RL 99.4%"),
]


def analyze(scale):
    """한 셀의 점군과 휴리스틱 결과를 낸다. 실패해도 점군은 돌려준다."""
    size_mm = np.array(scale) * BASE_CUBE_MM
    size_m = size_mm / 1000.0
    # 물체를 바닥에 놓는다 — 중심 z 는 높이의 절반.
    pos_m = np.array([OBJ_POS_M[0], OBJ_POS_M[1], size_m[2] / 2.0])
    pts = sc.sample_visible_box_surface(
        size_m=size_m, pos_m=pos_m, quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        view_dir=(0.0, 0.0, -1.0), n_points=N_POINTS, noise_mm=0.0, seed=SEED)

    info = {"pts": pts, "size_mm": size_mm, "top_z_mm": size_mm[2], "error": None}
    try:
        result, _ = heuristic.compute_grasp(
            points_base_mm=pts,
            part_names=["cube"], part_num_points=[len(pts)], target_part="cube",
            capture_posx=(400.0, 0.0, 600.0, 0.0, 180.0, 0.0),
            T_gripper2camera=np.eye(4))
        poses = heuristic.create_grasp_poses(
            result, grasp_depth_mm=15.0, along_axis_deeper_mm=0.0,
            floor_clearance_mm=heuristic.FLOOR_CLEARANCE_MM)
        info.update(result=result, poses=poses,
                    fingertip_z=float(poses["fingertip_grasp_position"][2]),
                    guard_raise=float(poses["floor_guard_raise_mm"]))
    except Exception as exc:
        # 폭 초과는 알고리즘이 '못 잡는다'고 스스로 판정한 것이다 — 그림에 그대로 쓴다.
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def draw_top(ax, info, title, verdict):
    """위에서 본 모습 — 장축과 닫힘축. 실기 시각화와 대응되는 그림이다."""
    pts, c = info["pts"], None
    s = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=4, cmap="viridis", alpha=0.8)
    if info["error"] is None:
        r = info["result"]
        c = r["centroid"]
        size = np.asarray(info["size_mm"], dtype=float)
        for vec, color, label in ((r["major_axis"], "#f5c518", "장축 (PCA)"),
                                  (r["tool_x"], "#00d0ff", "닫힘축 (tool_x)")):
            v = np.asarray(vec, dtype=float)
            # 축정렬 상자에서 그 방향 실제 반치수 — 선 길이가 물체와 맞아야 한다.
            L = float(np.abs(v) @ size) / 2.0 * 1.15
            ax.plot([c[0] - v[0] * L, c[0] + v[0] * L],
                    [c[1] - v[1] * L, c[1] + v[1] * L],
                    color=color, lw=2.5, label=label, zorder=3)
        ax.plot(c[0], c[1], "o", color="#ff3b30", ms=7, zorder=4)
        ax.set_title(f"{title}\n파지폭 {r['grasp_width_mm']:.1f}mm — {verdict}", fontsize=9)
    else:
        ax.set_title(f"{title}\n{verdict}", fontsize=9, color="#c00")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("y (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    return s


def draw_side(ax, info, title):
    """옆에서 본 모습 — 손가락이 물체 윗면 위에 멈추는지를 본다.

    합성 점군은 '보이는 윗면'만 샘플링하므로 측면도에서는 납작한 선으로만 보인다.
    그래서 점군 대신 물체 단면과 그리퍼 손가락을 실제 치수로 그린다.
    """
    sz = info["size_mm"][2]
    top = info["top_z_mm"]
    # 단면은 '닫힘축 방향'으로 잘라야 한다. 셀마다 닫힘축이 x 일 수도 y 일 수도 있다
    # (42x84 는 장축이 y 라 닫힘축이 x 로 간다). 축정렬 상자이므로
    # 닫힘축 방향 폭 = sum(|tool_x_i| * size_i) 이다.
    if info["error"] is None:
        tx = np.abs(np.asarray(info["result"]["tool_x"], dtype=float))
        sy = float(tx @ np.asarray(info["size_mm"], dtype=float))
    else:
        sy = float(info["size_mm"][1])

    # 물체 단면 (y-z). 바닥에 놓인 상자.
    ax.add_patch(plt.Rectangle((-sy / 2, 0), sy, sz,
                               facecolor="#34c759", alpha=0.25,
                               edgecolor="#34c759", lw=1.8, zorder=1))
    ax.axhline(top, color="#34c759", lw=1.6, zorder=2)
    ax.axhline(heuristic.FLOOR_CLEARANCE_MM, color="#8e8e93", lw=1.2, ls=":",
               label=f"바닥 가드 {heuristic.FLOOR_CLEARANCE_MM:.0f}mm", zorder=2)

    if info["error"] is not None:
        ax.text(0, sz * 0.6, info["error"][:40], color="#c00", fontsize=7, ha="center")
        return

    fz = info["fingertip_z"]
    w = info["result"]["grasp_width_mm"]
    above = fz > top
    col = "#ff3b30" if above else "#0a84ff"

    # 그리퍼 손가락 두 개 — 파지폭만큼 벌어져 손끝 목표 높이까지 내려온다.
    finger_top = max(top, fz) + max(sz, 30) * 0.6
    for sign in (-1, 1):
        ax.plot([sign * w / 2, sign * w / 2], [fz, finger_top],
                color=col, lw=4, solid_capstyle="butt", zorder=3)
    ax.plot([-w / 2, w / 2], [fz, fz], color=col, lw=1.2, ls="--", zorder=3,
            label=f"손끝 목표 {fz:.1f}mm")

    if above:
        # 26mm 물체에서 손끝이 윗면 위에 서는 것 — 허공을 집는 이유다.
        ax.annotate("", xy=(0, fz), xytext=(0, top),
                    arrowprops=dict(arrowstyle="<->", color="#ff3b30", lw=1.8))
        ax.text(-w / 2 - 4, (fz + top) / 2, f"+{fz - top:.1f}mm\n허공을 집는다",
                color="#ff3b30", fontsize=8, va="center", ha="right", weight="bold")

    span = max(sy, w) * 0.9 + 12
    ax.set_xlim(-span, span)
    ax.set_ylim(-3, finger_top + 5)
    ax.set_xlabel("닫힘축 방향 (mm)", fontsize=8)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"파지폭 {w:.1f}mm", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left")


def _setup_korean_font():
    """한글 폰트를 파일 경로로 직접 등록한다.

    apt 로 fonts-nanum 을 깔아도 fontconfig 캐시가 갱신되지 않아
    matplotlib 의 폰트 목록에 안 들어온다. 경로로 직접 넣으면 확실하다.
    없으면 조용히 넘어가되, 라벨이 네모로 깨지므로 경고를 남긴다.
    """
    from matplotlib import font_manager as fm
    for path in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).exists():
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            return
    print("경고: 한글 폰트가 없다 — 라벨이 깨진다. sudo apt install fonts-nanum")


def main():
    _setup_korean_font()
    plt.rcParams["axes.unicode_minus"] = False

    infos = [(t, analyze(s), v) for t, s, v in CELLS]
    fig, axes = plt.subplots(2, len(infos), figsize=(4.0 * len(infos), 7.2))
    for i, (title, info, verdict) in enumerate(infos):
        draw_top(axes[0, i], info, title, verdict)
        draw_side(axes[1, i], info, title)
        if info["error"]:
            print(f"[{title}] 예외: {info['error']}")
        else:
            print(f"[{title}] 파지폭 {info['result']['grasp_width_mm']:.1f}mm  "
                  f"손끝 z {info['fingertip_z']:.1f}mm  윗면 {info['top_z_mm']:.1f}mm  "
                  f"바닥가드 상승 {info['guard_raise']:.1f}mm")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("PCA 휴리스틱의 파지 축과 실패 원인 — 합성 점군 입력", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    dst = ROOT / "docs" / "img" / "grasp_pca_failure_map.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=140)
    print(f"\n기록: {dst}")


if __name__ == "__main__":
    main()
