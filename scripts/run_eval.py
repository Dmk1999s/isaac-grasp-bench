"""통합 평가 하네스 — 같은 환경에서 컨트롤러만 바꿔 비교한다.

핵심은 '달라지는 변수가 제어 방식 하나뿐'이어야 한다는 것이다(원칙 1).
그래서 두 방식이 **같은 태스크, 같은 액션 인터페이스(IK-Rel), 같은 시드,
같은 성공 기준, 같은 측정 코드**를 통과하게 만들었다.

  --controller pca : 점군 PCA 휴리스틱 + 상태기계
  --controller rl  : 학습된 PPO 정책

성공 기준은 Isaac Lab 자체 정의를 그대로 쓴다 (object_is_lifted, 0.04 m).
임계값을 내가 정하지 않는다(원칙 4).

반복 편차를 먼저 재고, 두 방식의 차이가 그보다 작으면 '판단 보류'다(원칙 3).
"""
import argparse
import csv
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--controller", choices=["pca", "rl"], required=True)
parser.add_argument("--checkpoint", type=str, default=None, help="rl 일 때 정책 체크포인트")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--repeats", type=int, default=10)
parser.add_argument("--base_seed", type=int, default=1000)
parser.add_argument("--n_points", type=int, default=600)
parser.add_argument("--noise_mm", type=float, default=0.0)
parser.add_argument("--grasp_depth_mm", type=float, default=15.0)
parser.add_argument("--along_axis_deeper_mm", type=float, default=0.0)
parser.add_argument("--floor_clearance_mm", type=float, default=30.0)
parser.add_argument("--grasp_offset_m", type=float, default=0.0,
                    help="대조군: 파지 자세를 일부러 어긋나게 준다 (pca 전용)")
parser.add_argument("--object_scale", type=str, default="0.8,0.8,0.8",
                    help="물체 스폰 스케일 sx,sy,sz. 기본값 0.8 은 Isaac Lab Lift-Cube 원래 값. "
                         "기저 DexCube 는 한 변 52.5mm 이므로 실제 치수 = 52.5 x scale [mm]. "
                         "종횡비/폭을 바꿔 실패 지도를 그릴 때 쓴다.")
parser.add_argument("--closing_axis_rot_deg", type=float, default=0.0,
                    help="명령 자세를 tool_z 기준으로 이만큼 회전시킨다. "
                         "휴리스틱은 닫힘축을 tool_x 로 정의하는데 Franka 손가락은 "
                         "hand 프레임 y 축으로 벌어진다 — 이 90도 차이를 검증/보정한다.")
parser.add_argument("--debug_env", type=int, default=-1)
parser.add_argument("--out", type=str, default="eval/result.csv")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac 기동 후에만 import 가능 ─────────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import grasp_pca_core as heuristic  # noqa: E402
import isaac_adapter as adapter  # noqa: E402
import synthetic_cloud as sc  # noqa: E402
import lift_ik_rel_rl  # noqa: E402,F401  (gym 등록)

from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = lift_ik_rel_rl.TASK_ID
MINIMAL_HEIGHT = 0.04          # Isaac Lab lift 기준값. 내가 정하는 값이 아니다.
BASE_CUBE_MM = 52.5            # DexCube 한 변 (측정: scale 0.8 에서 42.0mm)
SETTLE_STEPS = 20              # 안착은 20 스텝이면 충분하다(측정: 기울기 0.0도, z 표준편차 0.00mm)

# 상태기계 (pca 전용)
S_ABOVE, S_APPROACH, S_CLOSE, S_LIFT = 0, 1, 2, 3
APPROACH_HEIGHT = 0.10
LIFT_HEIGHT = 0.25
POS_TOL = 0.012
CLOSE_STEPS = 12
STATE_TIMEOUT = 60
# IK-Rel 의 액션 스케일(0.5)을 되돌려 델타를 명령으로 바꾼다.
IK_REL_SCALE = 0.5
# 델타를 클램프하지 않는 이유:
#   IK-Rel 은 명령을 '현재 자세 + 델타' 로 해석한다. 따라서 델타를
#   (목표 - 현재) 그대로 주면 IK 목표가 곧 절대 목표가 되어, IK-Abs 와
#   동일한 추종 성능을 얻는다. 액션 인터페이스는 RL 과 같게 유지하면서다.
#   델타를 작게 클램프하면 비례 제어기가 되어 추종이 나빠지는데, 그러면
#   비교 결과에 '제어 방식'이 아니라 '내 제어기 품질'이 섞인다.
#   아래 상한은 수치 폭주 방지용이며 실제로는 거의 걸리지 않는다.
MAX_STEP_M = 1.0
MAX_STEP_RAD = 3.15


def quat_delta_axis_angle(q_from, q_to):
    """두 쿼터니언(wxyz) 사이 회전을 축-각 벡터로. [N,3]

    IK-Rel 의 pose_rel 명령은 뒤 3개를 축-각(angle-axis)으로 해석한다
    (isaaclab.utils.math.apply_delta_pose 문서 참고).

    직접 구현하다 부호 처리를 틀렸던 적이 있어(짧은 쪽 회전 선택),
    Isaac Lab 의 검증된 유틸을 쓴다.
    """
    q_err = quat_mul(q_to, quat_conjugate(q_from))
    return axis_angle_from_quat(q_err)


def compute_grasp_targets(obj_pos, obj_quat, size_m, seed, args):
    """합성 점군 -> 휴리스틱 -> 절대 파지 자세."""
    n = len(obj_pos)
    pos_out = np.zeros((n, 3))
    quat_out = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (n, 1))
    failed = np.zeros(n, dtype=bool)
    reasons = {}
    for i in range(n):
        try:
            pts = sc.sample_visible_box_surface(
                size_m=size_m, pos_m=obj_pos[i], quat_wxyz=obj_quat[i],
                view_dir=(0.0, 0.0, -1.0), n_points=args.n_points,
                noise_mm=args.noise_mm, seed=seed * 10000 + i)
            result, _ = heuristic.compute_grasp(
                points_base_mm=pts,
                part_names=["cube"], part_num_points=[len(pts)], target_part="cube",
                capture_posx=(400.0, 0.0, 600.0, 0.0, 180.0, 0.0),
                T_gripper2camera=np.eye(4))
            poses = heuristic.create_grasp_poses(
                result,
                grasp_depth_mm=args.grasp_depth_mm,
                along_axis_deeper_mm=args.along_axis_deeper_mm,
                floor_clearance_mm=args.floor_clearance_mm)
            fingertip_mm = np.asarray(poses["fingertip_grasp_position"], dtype=np.float64)
            pos_out[i] = fingertip_mm / adapter.MM_PER_M
            # 휴리스틱의 닫힘축(tool_x)을 Franka 손가락 축(y)에 맞춘다.
            R = adapter.heuristic_rot_to_franka_hand(
                result["rotation_matrix"], extra_rot_deg=args.closing_axis_rot_deg)
            quat_out[i] = adapter.rotmat_to_quat_wxyz(R)
        except Exception as exc:
            failed[i] = True
            pos_out[i] = obj_pos[i]
            key = f"{type(exc).__name__}: {str(exc)[:100]}"
            reasons[key] = reasons.get(key, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"           [휴리스틱 예외 x{v}] {k}")
    return pos_out, quat_out, failed


def run_repeat(env, policy, seed, args, nominal_mm):
    """한 반복: 리셋 -> 안착 -> 컨트롤러 실행 -> 성공/사이클타임 기록."""
    device = env.unwrapped.device
    n = env.unwrapped.num_envs
    adim = env.unwrapped.action_space.shape[1]

    env.unwrapped.seed(seed)
    obs_dict, _ = env.reset()

    # 안착: 델타 0 을 명령해 팔을 가만히 둔다 (IK-Rel 이라 0 이 곧 정지다)
    hold = torch.zeros((n, adim), device=device)
    hold[:, -1] = 1.0
    for _ in range(SETTLE_STEPS):
        obs_dict = env.step(hold)[0]

    obj = env.unwrapped.scene["object"]
    ee = env.unwrapped.scene["ee_frame"]
    origins = env.unwrapped.scene.env_origins

    obj_pos = (obj.data.root_pos_w - origins).cpu().numpy()
    obj_quat = obj.data.root_quat_w.cpu().numpy()
    z_rest = obj_pos[:, 2]
    # 치수는 설정한 스케일에서 계산하고, 높이는 안착 높이로 교차검증한다.
    # (USD 바운딩박스는 실제 충돌 형상과 다르므로 쓰지 않는다)
    size_used = nominal_mm / 1000.0
    height_measured = float(z_rest.mean()) * 2.0
    if abs(height_measured - size_used[2]) > 0.004:
        print(f"           [경고] 높이 불일치: 설정 {size_used[2]*1000:.1f}mm "
              f"vs 안착 유도 {height_measured*1000:.1f}mm")

    heur_failed = np.zeros(n, dtype=bool)
    if args.controller == "pca":
        gp_np, gq_np, heur_failed = compute_grasp_targets(
            obj_pos, obj_quat, size_used, seed, args)
        gp_np = gp_np.copy()
        gp_np[:, 0] += args.grasp_offset_m
        gp = torch.tensor(gp_np, dtype=torch.float32, device=device)
        gq = torch.tensor(gq_np, dtype=torch.float32, device=device)
        state = torch.zeros(n, dtype=torch.long, device=device)
        state_steps = torch.zeros(n, dtype=torch.long, device=device)

    dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    max_steps = int(env.unwrapped.cfg.episode_length_s / dt) - SETTLE_STEPS - 2
    success_step = torch.full((n,), -1, dtype=torch.long, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)

    for step in range(max_steps):
        # 판정은 스텝 이전 상태로. env.step 은 종료 시 자동 리셋한다.
        lifted = obj.data.root_pos_w[:, 2] > MINIMAL_HEIGHT
        newly = lifted & active & (success_step < 0)
        success_step = torch.where(newly, torch.full_like(success_step, step), success_step)

        if args.controller == "rl":
            with torch.inference_mode():
                # rsl_rl 정책은 관측 '딕셔너리'를 받는다 (내부에서 obs_groups 로 인덱싱).
                # 텐서만 넘기면 IndexError 가 난다.
                actions = policy(obs_dict)
        else:
            ee_pos = ee.data.target_pos_w[..., 0, :] - origins
            ee_quat = ee.data.target_quat_w[..., 0, :]

            target = gp.clone()
            target[state == S_ABOVE, 2] += APPROACH_HEIGHT
            target[state == S_LIFT, 2] += LIFT_HEIGHT

            dpos = torch.clamp(target - ee_pos, -MAX_STEP_M, MAX_STEP_M)
            drot = torch.clamp(quat_delta_axis_angle(ee_quat, gq), -MAX_STEP_RAD, MAX_STEP_RAD)

            actions = torch.zeros((n, adim), device=device)
            actions[:, :3] = dpos / IK_REL_SCALE
            actions[:, 3:6] = drot / IK_REL_SCALE
            actions[:, -1] = torch.where(state >= S_CLOSE, -1.0, 1.0)

        obs_dict, _, terminated, truncated, _ = env.step(actions)

        if args.controller == "pca":
            ee_pos = ee.data.target_pos_w[..., 0, :] - origins
            reached = torch.linalg.norm(ee_pos - target, dim=-1) < POS_TOL
            state_steps += 1
            advance = ((reached | (state_steps > STATE_TIMEOUT)) & (state < S_LIFT))
            advance = torch.where(state == S_CLOSE, state_steps >= CLOSE_STEPS, advance)
            state = torch.where(advance, state + 1, state)
            state_steps = torch.where(advance, torch.zeros_like(state_steps), state_steps)

        if args.debug_env >= 0 and step % 20 == 0:
            k = args.debug_env
            ep = (ee.data.target_pos_w[..., 0, :] - origins)[k]
            op = (obj.data.root_pos_w - origins)[k]
            extra = ""
            if args.controller == "pca":
                extra = f" s={int(state[k])} tgt=({float(target[k,0]):.3f},{float(target[k,1]):.3f},{float(target[k,2]):.3f})"
            print(f"  step={step:3d} ee=({float(ep[0]):.3f},{float(ep[1]):.3f},{float(ep[2]):.3f}) "
                  f"obj_z={float(op[2]):.3f}{extra}")

        active = active & ~(terminated | truncated)
        if not bool(active.any()):
            break

    success = (success_step >= 0).cpu().numpy()
    cycle_s = np.where(success, success_step.cpu().numpy() * dt, np.nan)
    return success, cycle_s, heur_failed


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)

    scale = tuple(float(v) for v in args_cli.object_scale.split(","))
    assert len(scale) == 3, "--object_scale 은 sx,sy,sz 형식이어야 한다"
    env_cfg.scene.object.spawn.scale = scale
    nominal_mm = np.array(scale) * BASE_CUBE_MM

    env = gym.make(TASK, cfg=env_cfg)

    policy = None
    if args_cli.controller == "rl":
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry

        agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None,
                                device=env.unwrapped.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print(f"[정책] {args_cli.checkpoint}")

    print(f"[태스크] {TASK}  (양쪽 동일)")
    print(f"[물체] scale={scale} -> 치수 {np.round(nominal_mm,1)} mm")
    print(f"[컨트롤러] {args_cli.controller}")
    print(f"[설정] envs={args_cli.num_envs} repeats={args_cli.repeats} seed={args_cli.base_seed}~")
    print(f"[기준] 성공 = object z > {MINIMAL_HEIGHT} m (Isaac Lab lift 기준)")
    if args_cli.controller == "pca":
        print(f"[휴리스틱] depth={args_cli.grasp_depth_mm}mm "
              f"along_axis={args_cli.along_axis_deeper_mm}mm "
              f"floor_clr={args_cli.floor_clearance_mm}mm")
        if args_cli.grasp_offset_m:
            print(f"[대조군] 파지 자세를 x 로 {args_cli.grasp_offset_m}m 어긋나게 준다")

    rows, rates, times = [], [], []
    for r in range(args_cli.repeats):
        seed = args_cli.base_seed + r
        succ, cyc, hfail = run_repeat(env, policy, seed, args_cli, nominal_mm)
        rate = float(succ.mean())
        mean_t = float(np.nanmean(cyc)) if succ.any() else float("nan")
        rates.append(rate)
        times.append(mean_t)
        print(f"[반복 {r+1:2d}/{args_cli.repeats}] seed={seed} "
              f"성공률={rate*100:5.1f}%  평균 사이클={mean_t:5.2f}s"
              + (f"  휴리스틱실패={int(hfail.sum())}/{len(hfail)}"
                 if args_cli.controller == "pca" else ""))
        for i in range(len(succ)):
            rows.append({"controller": args_cli.controller,
                         "size_x_mm": round(float(nominal_mm[0]), 2),
                         "size_y_mm": round(float(nominal_mm[1]), 2),
                         "size_z_mm": round(float(nominal_mm[2]), 2),
                         "repeat": r, "seed": seed,
                         "env": i, "success": int(succ[i]), "cycle_s": cyc[i],
                         "heuristic_failed": int(hfail[i])})

    out = ROOT / args_cli.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rates = np.asarray(rates)
    times = np.asarray(times)
    print("\n" + "=" * 62)
    print(f"[{args_cli.controller}] 성공률 평균 {rates.mean()*100:.1f}%  "
          f"표준편차 {rates.std(ddof=1)*100:.1f}%p  "
          f"범위 {rates.min()*100:.1f}~{rates.max()*100:.1f}%")
    print(f"[{args_cli.controller}] 사이클타임 평균 {np.nanmean(times):.2f}s  "
          f"표준편차 {np.nanstd(times, ddof=1):.2f}s")
    print("=" * 62)
    print(f"원자료: {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
