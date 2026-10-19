"""오리 '지시 부위 파지' 평가 하네스 — PCA 와 PPO 가 같은 코드를 통과한다.

  source scripts/env.sh
  $PY scripts/run_duck_eval.py --controller pca --out eval/duck/pca.csv
  $PY scripts/run_duck_eval.py --controller rl --checkpoint <model.pt> --out eval/duck/rl_s1.csv
  # 대조군
  $PY scripts/run_duck_eval.py --controller pca --wrong_part --out eval/duck/ctrl_wrong_part.csv
  $PY scripts/run_duck_eval.py --controller pca --grasp_offset_m 0.2 --out eval/duck/ctrl_offset.csv

같게 두는 것: 태스크(lift_duck_env), 로봇·그리퍼, IK-Rel 액션, 물체 배치(평가 시드),
             자세·지시 부위, 성공 판정 코드(lift_duck_env.duck_step_flags).
다른 것:     컨트롤러 하나.

성공 = 물체 중심 z > 0.10 m 이고 두 손가락 면이 모두 지시 부위에 닿은 상태가 0.5 s 연속.
실패 유형 = heuristic_error | no_lift | wrong_part | unstable
  wrong_part : 들어올려 0.5 s 유지했지만 잡은 부위가 지시와 다름
  unstable   : 0.10 m 를 넘겼지만 0.5 s 를 못 버팀(떨어뜨림·미끄러짐)
"""
import argparse
import csv
import pathlib
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--controller", choices=["pca", "rl"], required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--targets", default="head,body,leg")
parser.add_argument("--poses", default="prone,side")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--base_seed", type=int, default=1000)
parser.add_argument("--grasp_depth_mm", type=float, default=15.0)
parser.add_argument("--floor_clearance_mm", type=float, default=5.0,
                    help="'클리어런스 < 부위 높이' 를 만족해야 한다(README 7절). 발 높이 ~12mm.")
parser.add_argument("--wrong_part", action="store_true",
                    help="대조군: PCA 에 지시와 다른 부위를 넘긴다 (지시·판정은 원래 부위)")
parser.add_argument("--grasp_offset_m", type=float, default=0.0,
                    help="대조군: 파지 위치를 x 로 어긋나게 준다")
parser.add_argument("--usd", default=None, help="오리 USD (기본: assets/duck/duck.usda)")
parser.add_argument("--debug_env", type=int, default=-1, help="이 env 의 궤적을 10스텝마다 출력")
parser.add_argument("--save_debug", action="store_true", help="env0 의 PCA 입력 점군을 npz 로 저장")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

import os  # noqa: E402

if args.usd:
    os.environ["DUCK_USD"] = str(pathlib.Path(args.usd).resolve())

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import grasp_pca_core as heuristic  # noqa: E402
import isaac_adapter as adapter  # noqa: E402
import lift_duck_env as duck_env  # noqa: E402
import toy_duck as td  # noqa: E402

from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = duck_env.TASK_ID
SETTLE_STEPS = 30
S_ABOVE, S_APPROACH, S_CLOSE, S_LIFT = 0, 1, 2, 3
APPROACH_HEIGHT = 0.10
LIFT_HEIGHT = 0.25
POS_TOL = 0.012
CLOSE_STEPS = 12
STATE_TIMEOUT = 60
IK_REL_SCALE = 0.5
WRONG = {"head": "body", "body": "head", "leg": "body"}
SURFACE, SURFACE_LAB = td.sample_surface(12000, seed=0)
PART_FAMILY_NAME = {"head": "head", "body": "body", "leg1": "leg", "leg2": "leg"}


def pca_targets(obj_pos, obj_quat, target, seed, debug_path=None):
    """보이는 면 점군 + 부위 라벨 → 휴리스틱 → 손끝 목표 자세."""
    n = len(obj_pos)
    pos_out = obj_pos.copy()
    quat_out = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (n, 1))
    failed = np.zeros(n, dtype=bool)
    reasons = {}
    for i in range(n):
        w, x, y, z = obj_quat[i]
        world = Rotation.from_quat([x, y, z, w]).apply(SURFACE) + obj_pos[i]
        vis, lab = td.visible_from_above(world, SURFACE_LAB)
        names, counts, chunks = [], [], []
        for key in ("head", "body", "leg1", "leg2"):
            m = lab == key
            if m.sum() == 0:
                continue
            names.append(PART_FAMILY_NAME[key])
            counts.append(int(m.sum()))
            chunks.append(vis[m])
        pts_mm = np.concatenate(chunks) * adapter.MM_PER_M
        if debug_path is not None and i == 0:
            np.savez(debug_path, points_mm=pts_mm, names=np.array(names), counts=np.array(counts))
        try:
            result, _ = heuristic.compute_grasp(
                points_base_mm=pts_mm, part_names=names, part_num_points=counts,
                target_part=target,
                capture_posx=(400.0, 0.0, 600.0, 0.0, 180.0, 0.0), T_gripper2camera=np.eye(4))
            poses = heuristic.create_grasp_poses(
                result, grasp_depth_mm=args.grasp_depth_mm, along_axis_deeper_mm=0.0,
                floor_clearance_mm=args.floor_clearance_mm)
            pos_out[i] = np.asarray(poses["fingertip_grasp_position"], float) / adapter.MM_PER_M
            R = adapter.heuristic_rot_to_franka_hand(result["rotation_matrix"])
            quat_out[i] = adapter.rotmat_to_quat_wxyz(R)
        except Exception as exc:  # 휴리스틱 실패도 결과다. 삼키지 않고 센다.
            failed[i] = True
            k = f"{type(exc).__name__}: {str(exc)[:90]}"
            reasons[k] = reasons.get(k, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      [휴리스틱 예외 x{v}] {k}")
    return pos_out, quat_out, failed


def run_episode(env, policy, target, pose, seed):
    u = env.unwrapped
    dev, n = u.device, u.num_envs
    adim = u.action_space.shape[1]
    u.duck_eval = {"target": target, "pose": pose}
    u.seed(seed)
    obs, _ = env.reset()

    hold = torch.zeros((n, adim), device=dev)
    hold[:, -1] = 1.0
    for _ in range(SETTLE_STEPS):
        obs = env.step(hold)[0]

    obj, ee, robot = u.scene["object"], u.scene["ee_frame"], u.scene["robot"]
    origins = u.scene.env_origins
    finger_ids = robot.find_joints("panda_finger_joint.*")[0]

    heur_failed = np.zeros(n, dtype=bool)
    if args.controller == "pca":
        op = (obj.data.root_pos_w - origins).cpu().numpy()
        oq = obj.data.root_quat_w.cpu().numpy()
        pca_part = WRONG[target] if args.wrong_part else target
        dbg = None
        if args.save_debug:
            dbg = pathlib.Path(args.out).with_suffix("").as_posix() + f"_{target}_{pose}_{seed}_cloud.npz"
        gp_np, gq_np, heur_failed = pca_targets(op, oq, pca_part, seed, dbg)
        gp_np[:, 0] += args.grasp_offset_m
        gp = torch.tensor(gp_np, dtype=torch.float32, device=dev)
        gq = torch.tensor(gq_np, dtype=torch.float32, device=dev)
        state = torch.zeros(n, dtype=torch.long, device=dev)
        state_steps = torch.zeros(n, dtype=torch.long, device=dev)

    dt = u.cfg.sim.dt * u.cfg.decimation
    max_steps = int(u.cfg.episode_length_s / dt) - SETTLE_STEPS - 2
    H = duck_env.SUCCESS_HOLD_STEPS
    run_ok = torch.zeros(n, dtype=torch.long, device=dev)
    run_any = torch.zeros(n, dtype=torch.long, device=dev)
    success_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    sustained_any = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    fam_at_lift = torch.full((n,), -2, dtype=torch.long, device=dev)
    gap_at = torch.full((n,), float("nan"), device=dev)
    max_z = torch.zeros(n, device=dev)
    active = torch.ones(n, dtype=torch.bool, device=dev)

    for step in range(max_steps):
        # 판정은 스텝 이전 상태로 (env.step 은 종료 시 자동 리셋한다)
        lifted, ok, fam = duck_env.duck_step_flags(u)
        lifted, ok = lifted & active, ok & active
        run_ok = torch.where(lifted & ok, run_ok + 1, torch.zeros_like(run_ok))
        run_any = torch.where(lifted, run_any + 1, torch.zeros_like(run_any))
        ever_lifted |= lifted
        newly = (run_ok >= H) & (success_step < 0)
        success_step = torch.where(newly, torch.full_like(success_step, step), success_step)
        first_sus = (run_any >= H) & ~sustained_any
        fam_at_lift = torch.where(first_sus, fam, fam_at_lift)
        gap = robot.data.joint_pos[:, finger_ids].sum(dim=1)
        gap_at = torch.where(first_sus, gap, gap_at)
        sustained_any |= run_any >= H
        max_z = torch.maximum(max_z, torch.where(active, obj.data.root_pos_w[:, 2] - origins[:, 2], max_z))

        if args.controller == "rl":
            with torch.inference_mode():
                actions = policy(obs)
        else:
            ee_pos = ee.data.target_pos_w[..., 0, :] - origins
            ee_quat = ee.data.target_quat_w[..., 0, :]
            target_p = gp.clone()
            target_p[state == S_ABOVE, 2] += APPROACH_HEIGHT
            target_p[state == S_LIFT, 2] += LIFT_HEIGHT
            dpos = torch.clamp(target_p - ee_pos, -1.0, 1.0)
            drot = torch.clamp(axis_angle_from_quat(quat_mul(gq, quat_conjugate(ee_quat))), -3.15, 3.15)
            actions = torch.zeros((n, adim), device=dev)
            actions[:, :3] = dpos / IK_REL_SCALE
            actions[:, 3:6] = drot / IK_REL_SCALE
            actions[:, -1] = torch.where(state >= S_CLOSE, -1.0, 1.0)

        obs, _, term, trunc, _ = env.step(actions)

        if args.controller == "pca":
            ee_pos = ee.data.target_pos_w[..., 0, :] - origins
            reached = torch.linalg.norm(ee_pos - target_p, dim=-1) < POS_TOL
            state_steps += 1
            adv = (reached | (state_steps > STATE_TIMEOUT)) & (state < S_LIFT)
            adv = torch.where(state == S_CLOSE, state_steps >= CLOSE_STEPS, adv)
            state = torch.where(adv, state + 1, state)
            state_steps = torch.where(adv, torch.zeros_like(state_steps), state_steps)

        if args.debug_env >= 0 and step % 10 == 0:
            k = args.debug_env
            e = (ee.data.target_pos_w[..., 0, :] - origins)[k]
            o = (obj.data.root_pos_w - origins)[k]
            st = f" s={int(state[k])} tgt=({float(target_p[k,0]):.3f},{float(target_p[k,1]):.3f},{float(target_p[k,2]):.3f})" \
                if args.controller == "pca" else ""
            from isaaclab.utils.math import quat_apply_inverse
            loc = quat_apply_inverse(obj.data.root_quat_w[k:k+1], (ee.data.target_pos_w[k:k+1, 0, :] - obj.data.root_pos_w[k:k+1]))[0]
            rr = torch.linalg.norm((loc - u.duck_shape_c) / u.duck_shape_s, dim=-1)
            print(f"      tcp_local_mm=({float(loc[0])*1000:.0f},{float(loc[1])*1000:.0f},{float(loc[2])*1000:.0f}) "
                  f"r[head,beak,body,leg1,leg2]={[round(float(v),2) for v in rr]}")
            print(f"    t={step:3d} ee=({float(e[0]):.3f},{float(e[1]):.3f},{float(e[2]):.3f}) "
                  f"obj=({float(o[0]):.3f},{float(o[1]):.3f},{float(o[2]):.3f}) gap={float(gap[k])*1000:.1f}mm "
                  f"part={int(fam[k])}{st}", flush=True)

        active &= ~(term | trunc)
        if not bool(active.any()):
            break

    succ = (success_step >= 0).cpu().numpy()
    sus = sustained_any.cpu().numpy()
    ever = ever_lifted.cpu().numpy()
    fam = fam_at_lift.cpu().numpy()
    cat = np.where(succ, "success",
          np.where(heur_failed, "heuristic_error",
          np.where(sus, "wrong_part",
          np.where(ever, "unstable", "no_lift"))))
    names = {-2: "", -1: "none", 0: "head", 1: "body", 2: "leg", 3: "mixed"}
    return {
        "success": succ.astype(int),
        "category": cat,
        "grasped_part": [names[int(f)] for f in fam],
        "cycle_s": np.where(succ, success_step.cpu().numpy() * dt, np.nan),
        "finger_gap_mm": (gap_at.cpu().numpy() * 1000).round(2),
        "max_obj_z_mm": (max_z.cpu().numpy() * 1000).round(1),
        "heuristic_failed": heur_failed.astype(int),
    }


def main():
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    # 평가에서는 관측 잡음을 끈다 (기존 하네스와 같은 조건)
    env_cfg.observations.policy.enable_corruption = False
    env = gym.make(TASK, cfg=env_cfg)

    policy = None
    if args.controller == "rl":
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry
        agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        # 래퍼는 러너 생성용으로만 쓴다. 스텝은 gym 환경으로 돌려 관측 딕셔너리를 그대로
        # 정책에 넘긴다 (run_eval.py 와 같은 방식).
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    usd = os.environ.get("DUCK_USD", duck_env.DUCK_USD)
    label = args.controller + ("+wrong_part" if args.wrong_part else "") + \
        (f"+offset{args.grasp_offset_m}" if args.grasp_offset_m else "")
    print(f"[태스크] {TASK}  [컨트롤러] {label}  [물체] {usd}")
    print(f"[성공] z>{duck_env.SUCCESS_HEIGHT}m & 지시부위, {duck_env.SUCCESS_HOLD_STEPS}스텝 연속")

    rows = []
    for target in args.targets.split(","):
        for pose in args.poses.split(","):
            rates = []
            for r in range(args.repeats):
                seed = args.base_seed + r
                t0 = time.time()
                res = run_episode(env, policy, target, pose, seed)
                rate = res["success"].mean()
                rates.append(rate)
                cats = {c: int((res["category"] == c).sum()) for c in np.unique(res["category"])}
                print(f"  {target:4s} {pose:5s} seed={seed} 성공 {rate*100:5.1f}%  {cats}  ({time.time()-t0:.0f}s)")
                for i in range(args.num_envs):
                    rows.append({"controller": label, "checkpoint": args.checkpoint or "",
                                 "usd": pathlib.Path(usd).name, "target": target, "pose": pose,
                                 "seed": seed, "env": i,
                                 **{k: v[i] for k, v in res.items()}})
            rates = np.array(rates)
            print(f"[{label}] {target}/{pose}: 평균 {rates.mean()*100:.1f}%  "
                  f"시드간 표준편차 {rates.std(ddof=1)*100 if len(rates) > 1 else 0:.1f}%p")

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"원자료: {out}")


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
