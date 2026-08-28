"""베이스라인: PCA 파지 휴리스틱을 Isaac Lab Lift-Cube 에서 실행하고 측정한다.

측정 대상
  - 성공률 : Isaac Lab 자체 기준을 그대로 쓴다 (object_is_lifted, minimal_height=0.04)
             임계값을 내가 정하지 않는다 (실험 설계 원칙 4).
  - 사이클타임 : 에피소드 시작부터 성공 판정이 처음 참이 될 때까지의 시간(초)

반복 편차
  같은 설정을 --repeats 회 반복한다. 각 반복은 다른 시드를 쓴다.
  이후 RL 과의 차이가 이 편차보다 작으면 '판단 보류'로 보고해야 한다 (원칙 3).
"""
import argparse
import csv
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32, help="한 반복에서 동시에 도는 환경 수")
parser.add_argument("--repeats", type=int, default=10, help="반복 횟수 (반복 편차 측정용)")
parser.add_argument("--base_seed", type=int, default=1000, help="반복 i 는 base_seed+i 를 쓴다")
parser.add_argument("--noise_mm", type=float, default=0.0, help="점군 잡음 표준편차 [mm]")
parser.add_argument("--n_points", type=int, default=600, help="점군 점 개수")
parser.add_argument("--out", type=str, default="eval/baseline_pca.csv")
parser.add_argument("--floor_clearance_mm", type=float, default=30.0,
                    help="손끝이 내려갈 수 있는 최저 높이. 원본 30.0 은 실기 작업대의 "
                         "불확실성을 감안한 안전 여유다. 시뮬에서는 지오메트리가 정확하다.")
parser.add_argument("--close_steps", type=int, default=12,
                    help="그리퍼 닫힘 대기 스텝")
parser.add_argument("--debug_env", type=int, default=-1,
                    help=">=0 이면 그 env 의 스텝별 상태를 출력한다")
parser.add_argument("--grasp_depth_mm", type=float, default=5.0,
                    help="접촉점에서 얼마나 파고들어 손끝을 둘지. 휴리스틱 원본 기본값은 5.0 "
                         "(RG2 기준 안전 여유). 값이 작으면 물체 위를 헛집는다.")
parser.add_argument("--along_axis_deeper_mm", type=float, default=0.0,
                    help="주축 방향으로 파지점을 미는 양. 원본 기본값 20.0 은 실기에서 "
                         "긴 부품을 깊이 잡으려 튜닝한 값이라 작은 큐브에서는 물체 밖으로 나간다.")
parser.add_argument("--grasp_offset_m", type=float, default=0.0,
                    help="대조군: 파지 자세를 이만큼 x 방향으로 일부러 어긋나게 준다. "
                         "여기서도 성공률이 높으면 하네스가 파지를 재고 있지 않다는 뜻이다.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac 기동 후에만 import 가능 ─────────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import grasp_pca_core as heuristic  # noqa: E402
import isaac_adapter as adapter  # noqa: E402
import synthetic_cloud as sc  # noqa: E402

TASK = "Isaac-Lift-Cube-Franka-IK-Abs-v0"
MINIMAL_HEIGHT = 0.04          # Isaac Lab lift 태스크의 기준값. 여기서 정하는 값이 아니다.

# 상태기계
S_ABOVE, S_APPROACH, S_CLOSE, S_LIFT = 0, 1, 2, 3
APPROACH_HEIGHT = 0.10         # 파지점 위 접근 높이 [m]
LIFT_HEIGHT = 0.25             # 들어올릴 높이 [m]
POS_TOL = 0.012                # 도달 판정 [m]
CLOSE_STEPS = 12               # 그리퍼 닫힘 대기 스텝
SETTLE_STEPS = 40              # 스폰 높이에서 테이블로 안착시키는 스텝
                               # (스폰 z=0.055 에서 떨어지며 튀므로 넉넉히 준다)
STATE_TIMEOUT = 45             # 한 상태 최대 스텝 (4상태 x 45 < 에피소드 예산)


def measure_object_half_extent(env):
    """큐브의 실제 치수를 USD 바운딩박스에서 잰다 (가정하지 않는다).

    prim 경로는 버전에 따라 달라질 수 있어 후보를 순서대로 시도하고,
    전부 실패하면 스테이지를 훑어 'Object' 가 들어간 경로를 찾는다.
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cfg_path = env.unwrapped.scene["object"].cfg.prim_path
    candidates = [
        cfg_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0"),
        "/World/envs/env_0/Object",
    ]
    prim = None
    for c in candidates:
        p = stage.GetPrimAtPath(c)
        if p and p.IsValid():
            prim, used = p, c
            break
    if prim is None:
        for p in stage.Traverse():
            sp = str(p.GetPath())
            if sp.startswith("/World/envs/env_0/") and "Object" in sp:
                prim, used = p, sp
                break
    if prim is None:
        raise RuntimeError("물체 prim 을 찾지 못했다 — 경로 규약이 바뀌었는지 확인하라.")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    size = np.array([rng.GetMax()[i] - rng.GetMin()[i] for i in range(3)])
    if not np.all(np.isfinite(size)) or size.min() <= 0:
        raise RuntimeError(f"바운딩박스가 이상하다: {size} (prim={used})")
    print(f"[측정] 물체 prim = {used}")
    return size


def compute_grasp_targets(obj_pos, obj_quat, size_m, seed, noise_mm, n_points,
                          grasp_depth_mm, along_axis_deeper_mm, floor_clearance_mm,
                          debug=False):
    """각 env 마다 합성 점군 -> 휴리스틱 -> 파지 자세.

    반환: (목표 위치 [N,3], 목표 자세 [N,4], 실패 마스크 [N])
    휴리스틱이 예외를 내면(잡을 수 없음 등) 그 env 는 실패로 기록하고
    물체 바로 위를 향하는 자세를 넣어 에피소드는 그대로 굴린다.
    """
    n = len(obj_pos)
    pos_out = np.zeros((n, 3))
    quat_out = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (n, 1))   # 아래를 향하는 기본 자세
    failed = np.zeros(n, dtype=bool)

    reasons = {}
    for i in range(n):
        try:
            pts = sc.sample_visible_box_surface(
                size_m=size_m, pos_m=obj_pos[i], quat_wxyz=obj_quat[i],
                view_dir=(0.0, 0.0, -1.0), n_points=n_points,
                noise_mm=noise_mm, seed=seed * 10000 + i)
            # compute_grasp 의 **kwargs 는 calculate_3d_pca 로 흘러간다.
            # grasp_depth_mm / along_axis_deeper_mm 은 create_grasp_poses 의 인자이므로
            # 여기에 넣으면 TypeError 가 난다. 그래서 poses 는 따로 만든다.
            result, _ = heuristic.compute_grasp(
                points_base_mm=pts,
                part_names=["cube"], part_num_points=[len(pts)], target_part="cube",
                capture_posx=(400.0, 0.0, 600.0, 0.0, 180.0, 0.0),
                T_gripper2camera=np.eye(4))
            poses = heuristic.create_grasp_poses(
                result,
                grasp_depth_mm=grasp_depth_mm,
                along_axis_deeper_mm=along_axis_deeper_mm,
                floor_clearance_mm=floor_clearance_mm)
            # IK-Abs 액션의 body_offset(0.107m)이 이미 TCP 기준이므로
            # 플랜지 자세가 아니라 손끝 위치를 목표로 준다.
            fingertip_mm = np.asarray(poses["fingertip_grasp_position"], dtype=np.float64)
            pos_out[i] = fingertip_mm / adapter.MM_PER_M
            quat_out[i] = adapter.rotmat_to_quat_wxyz(result["rotation_matrix"])
            if i == 0 and debug:
                cp = np.asarray(result["contact_point"], dtype=np.float64)
                tz = np.asarray(result["tool_z"], dtype=np.float64)
                print(f"  [grasp계산] 물체={np.round(obj_pos[0]*1000,1)}mm "
                      f"점군중심={np.round(pts.mean(axis=0),1)}mm")
                print(f"             접촉점={np.round(cp,1)}mm 손끝={np.round(fingertip_mm,1)}mm")
                print(f"             tool_z={np.round(tz,3)} 폭={result['grasp_width_mm']:.1f}mm "
                      f"고유값비={float(np.asarray(result['eigenvalues'])[0]/max(1e-9,float(np.asarray(result['eigenvalues'])[1]))):.2f}")
        except Exception as exc:
            failed[i] = True
            pos_out[i] = obj_pos[i]
            key = f"{type(exc).__name__}: {str(exc)[:120]}"
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"           [휴리스틱 예외 x{v}] {k}")
    return pos_out, quat_out, failed


def run_one_repeat(env, size_m, seed, args):
    """한 반복: 모든 env 를 리셋하고 한 에피소드씩 굴린다."""
    device = env.unwrapped.device
    n = env.unwrapped.num_envs

    env.unwrapped.seed(seed)
    obs, _ = env.reset()

    # 물체가 테이블에 안착하도록 몇 스텝 흘린다 (스폰 높이에서 떨어진다)
    # 안착 동안 팔은 현재 EE 자세를 그대로 명령해 가만히 둔다.
    # 위치를 0 으로 두면 IK 가 팔을 베이스 원점으로 끌어당긴다.
    ee_init = env.unwrapped.scene["ee_frame"]
    settle = torch.zeros((n, env.unwrapped.action_space.shape[1]), device=device)
    settle[:, -1] = 1.0
    for _ in range(SETTLE_STEPS):
        settle[:, :3] = ee_init.data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
        settle[:, 3:7] = ee_init.data.target_quat_w[..., 0, :]
        env.step(settle)

    obj = env.unwrapped.scene["object"]
    origins = env.unwrapped.scene.env_origins
    obj_pos = (obj.data.root_pos_w - origins).cpu().numpy()
    obj_quat = obj.data.root_quat_w.cpu().numpy()

    # 안착 상태 점검: 큐브가 평평히 놓였는지, 높이가 일정한지
    z_rest = obj_pos[:, 2]
    tilt_deg = 2.0 * np.degrees(np.arccos(np.clip(np.abs(obj_quat[:, 0]), 0.0, 1.0)))
    # 크기는 안착 높이에서 유도한다. USD 바운딩박스는 실제 충돌 형상과 어긋난다
    # (bbox 48mm vs 안착높이 유도 42mm). 그리퍼가 실제로 상대하는 것은 후자다.
    measured = float(z_rest.mean()) * 2.0
    size_used = np.array([measured, measured, measured])
    if args.debug_env >= 0:
        print(f"  [안착] z 평균={z_rest.mean()*1000:.1f}mm 표준편차={z_rest.std()*1000:.2f}mm "
              f"| 기울기 평균={tilt_deg.mean():.1f}도 최대={tilt_deg.max():.1f}도")
        print(f"  [크기] 안착높이로부터 추정={z_rest.mean()*2*1000:.1f}mm "
              f"| USD bbox 전체={np.round(np.asarray(size_m)*1000,1)}mm")
        print(f"  [자세] obj_quat[0]={np.round(obj_quat[0],4)} obj_pos[0]={np.round(obj_pos[0]*1000,1)}mm")

    grasp_pos, grasp_quat, heur_failed = compute_grasp_targets(
        obj_pos, obj_quat, size_used, seed, args.noise_mm, args.n_points,
        args.grasp_depth_mm, args.along_axis_deeper_mm, args.floor_clearance_mm,
        debug=(args.debug_env >= 0))

    grasp_pos = grasp_pos.copy()
    grasp_pos[:, 0] += args.grasp_offset_m          # 대조군용 의도적 오차
    gp = torch.tensor(grasp_pos, dtype=torch.float32, device=device)
    gq = torch.tensor(grasp_quat, dtype=torch.float32, device=device)

    state = torch.zeros(n, dtype=torch.long, device=device)
    state_steps = torch.zeros(n, dtype=torch.long, device=device)
    success_step = torch.full((n,), -1, dtype=torch.long, device=device)

    dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    episode_steps = int(env.unwrapped.cfg.episode_length_s / dt)
    # 안착에 쓴 스텝도 에피소드 예산을 소모한다. 남은 만큼만 돌린다.
    # 이걸 빼먹으면 루프 도중 환경이 자동 리셋되고, 리셋 직후 스폰 높이(z=0.055)가
    # 기준(0.04)보다 높아서 '성공'으로 잘못 집계된다.
    max_steps = episode_steps - SETTLE_STEPS - 2

    ee = env.unwrapped.scene["ee_frame"]
    actions = torch.zeros((n, env.unwrapped.action_space.shape[1]), device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    ee_gap = torch.zeros(n, device=device)

    for step in range(max_steps):
        # 판정은 '스텝 이전' 상태로 한다. env.step 은 종료 시 자동 리셋하므로
        # 스텝 뒤에 읽으면 리셋된 물체 위치를 보게 된다.
        lifted = obj.data.root_pos_w[:, 2] > MINIMAL_HEIGHT
        newly = lifted & active & (success_step < 0)
        success_step = torch.where(newly, torch.full_like(success_step, step), success_step)

        target = gp.clone()
        target[state == S_ABOVE, 2] += APPROACH_HEIGHT
        target[state == S_LIFT, 2] += LIFT_HEIGHT
        grip = torch.where(state >= S_CLOSE,
                           -torch.ones(n, device=device), torch.ones(n, device=device))

        actions[:, :3] = target
        actions[:, 3:7] = gq
        actions[:, -1] = grip
        _, _, terminated, truncated, _ = env.step(actions)

        ee_pos = ee.data.target_pos_w[..., 0, :] - origins
        ee_gap = torch.linalg.norm(ee_pos - target, dim=-1)
        reached = ee_gap < POS_TOL
        state_steps += 1

        advance = ((reached | (state_steps > STATE_TIMEOUT)) & (state < S_LIFT))
        in_close = state == S_CLOSE
        advance = torch.where(in_close, state_steps >= args.close_steps, advance)
        state = torch.where(advance, state + 1, state)
        state_steps = torch.where(advance, torch.zeros_like(state_steps), state_steps)

        if args.debug_env >= 0 and step % 10 == 0:
            k = args.debug_env
            fj = env.unwrapped.scene["robot"].data.joint_pos[k, -2:]
            op = obj.data.root_pos_w[k] - origins[k]
            print(f"  step={step:3d} s={int(state[k])} "
                  f"ee=({float(ee_pos[k,0]):.3f},{float(ee_pos[k,1]):.3f},{float(ee_pos[k,2]):.3f}) "
                  f"obj=({float(op[0]):.3f},{float(op[1]):.3f},{float(op[2]):.3f}) "
                  f"tgt=({float(target[k,0]):.3f},{float(target[k,1]):.3f},{float(target[k,2]):.3f}) "
                  f"fing={float(fj[0]):.4f}")

        active = active & ~(terminated | truncated)
        if not bool(active.any()):
            break

    success = (success_step >= 0).cpu().numpy()
    cycle_s = np.where(success, success_step.cpu().numpy() * dt, np.nan)
    diag = {
        "final_state_mean": float(state.float().mean().item()),
        "reached_lift_frac": float((state >= S_LIFT).float().mean().item()),
        "final_obj_z_mean": float(obj.data.root_pos_w[:, 2].mean().item()),
        "ee_to_target_mean": float(ee_gap.mean().item()),
    }
    return success, cycle_s, heur_failed, diag


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)

    size_m = measure_object_half_extent(env)
    print(f"[측정] USD 바운딩박스 = {np.round(size_m * 1000, 1)} mm "
          f"(참고용 — 실제 사용 크기는 반복마다 안착 높이에서 유도한다)")
    print(f"[설정] envs={args_cli.num_envs} repeats={args_cli.repeats} "
          f"noise={args_cli.noise_mm}mm points={args_cli.n_points}")
    print(f"[기준] 성공 = object z > {MINIMAL_HEIGHT} m (Isaac Lab lift 기준)")
    print(f"[휴리스틱] grasp_depth={args_cli.grasp_depth_mm}mm "
          f"along_axis_deeper={args_cli.along_axis_deeper_mm}mm")
    if args_cli.grasp_offset_m != 0.0:
        print(f"[대조군] 파지 자세를 x 방향으로 {args_cli.grasp_offset_m}m 어긋나게 준다.")

    rows = []
    rates, times = [], []
    for r in range(args_cli.repeats):
        seed = args_cli.base_seed + r
        succ, cyc, hfail, diag = run_one_repeat(env, size_m, seed, args_cli)
        rate = float(succ.mean())
        mean_t = float(np.nanmean(cyc)) if succ.any() else float("nan")
        rates.append(rate)
        times.append(mean_t)
        print(f"[반복 {r+1:2d}/{args_cli.repeats}] seed={seed} "
              f"성공률={rate*100:5.1f}%  평균 사이클={mean_t:5.2f}s  "
              f"휴리스틱실패={int(hfail.sum())}/{len(hfail)}")
        print(f"           [진단] 최종상태평균={diag['final_state_mean']:.2f} "
              f"LIFT도달={diag['reached_lift_frac']*100:.0f}% "
              f"최종물체z={diag['final_obj_z_mean']:.3f}m "
              f"EE-목표거리={diag['ee_to_target_mean']:.3f}m")
        for i in range(len(succ)):
            rows.append({"repeat": r, "seed": seed, "env": i,
                         "success": int(succ[i]), "cycle_s": cyc[i],
                         "heuristic_failed": int(hfail[i])})

    out = ROOT / args_cli.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rates = np.asarray(rates)
    times = np.asarray(times)
    print("\n" + "=" * 60)
    print(f"성공률     평균 {rates.mean()*100:.1f}%  표준편차 {rates.std(ddof=1)*100:.1f}%p  "
          f"범위 {rates.min()*100:.1f}~{rates.max()*100:.1f}%")
    print(f"사이클타임 평균 {np.nanmean(times):.2f}s  표준편차 {np.nanstd(times, ddof=1):.2f}s")
    print(f"\n반복 편차(성공률 표준편차) = {rates.std(ddof=1)*100:.1f}%p")
    print("→ 이후 RL 과의 성공률 차이가 이 값보다 작으면 '판단 보류'로 보고한다.")
    print("=" * 60)
    print(f"\n원자료: {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
