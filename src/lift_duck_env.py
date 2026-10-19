"""오리 근사 모형 + '지시한 부위를 잡아 들어올리기' 태스크 (IK-Rel, PCA·PPO 공용).

기존 Lift-Cube IK-Rel 태스크(lift_ik_rel_rl.py)에서 바뀌는 것만 여기 있다.
로봇·그리퍼·액션 인터페이스·PPO 하이퍼파라미터는 그대로다 (원칙 1).

바뀌는 것
---------
물체     : 큐브 → 오리 근사 모형 (assets/duck/duck.usda, 고무 근사 부드러운 접촉)
리셋     : 자세 {엎드림, 옆으로 누움} × 무작위 yaw, 지시 부위 {head, body, leg} 무작위
관측     : + 물체 자세(쿼터니언), + 지시 부위 위치(로봇 기준), + 지시 부위 원-핫
보상     : 다가가기 = TCP ↔ 지시 부위 기준점 거리
           들기·목표추종 = 기존 항 × '지시 부위를 잡고 있는가'
성공     : duck_step_flags() — 평가 하네스에서 쓴다 (PCA·PPO 동일 코드)
           두 손가락 면이 모두 지시 부위에 닿은 채 0.10m 이상 들려 0.5s 유지.
           ⚠ 보상(학습)은 TCP 기준 part_ok 를 썼다. 평가 판정만 손가락 기준으로 바꿨다.

'지시 부위를 잡고 있는가' (part_ok)
-----------------------------------
TCP(두 손가락 끝 사이 중점)를 물체 로컬로 옮겨 어느 타원체에 가장 깊이 들어가 있는지 본다
(toy_duck.classify_points 와 같은 계산). 그 부위가 지시 부위이고 정규화 반경 ≤ 1.25 면 참.
카메라 의미 분할과 이 형상 판정의 일치율은 99.9~100% 로 확인했다 (assets/duck/README.md).
"""
import os
import pathlib
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

from isaaclab_tasks.manager_based.manipulation.lift.config.franka import ik_rel_env_cfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import toy_duck as td  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DUCK_USD = os.environ.get("DUCK_USD", str(ROOT / "assets/duck/duck.usda"))
TASK_ID = "IsaacGraspBench-Lift-Duck-Franka-IK-Rel-v0"

POSES = ("prone", "side")
# 떨어뜨릴 높이. 옆으로 누움은 좌/우(roll ±90°)를 반반 고른다.
_POSE_Z = {"prone": 0.030, "side": 0.032}

LIFT_HEIGHT_REWARD = 0.05     # 보상용 '들림' 높이 (안착 중심 z ≈ 0.021~0.026)
PART_OK_RADIUS = 1.25


# ─────────────────────────────────────────────────────────────
# 상태: env 마다 지시 부위·자세. 평가 하네스가 고정값을 줄 수 있다.
#   env.duck_eval = {"target": "head", "pose": "prone"}  (None 이면 무작위)
# ─────────────────────────────────────────────────────────────
def _state(env):
    if not hasattr(env, "duck_target"):
        n, dev = env.num_envs, env.device
        env.duck_target = torch.zeros(n, dtype=torch.long, device=dev)
        env.duck_pose = torch.zeros(n, dtype=torch.long, device=dev)
        c, s, fam = td.shape_arrays()
        env.duck_shape_c = torch.tensor(c, dtype=torch.float32, device=dev)
        env.duck_shape_s = torch.tensor(s, dtype=torch.float32, device=dev)
        env.duck_shape_fam = torch.tensor(fam, dtype=torch.long, device=dev)
        a = td.anchor_points()
        env.duck_anchor = torch.tensor(  # [4,3]: head, body, leg1, leg2
            np.stack([a["head"], a["body"], a["leg"][0], a["leg"][1]]), dtype=torch.float32, device=dev)
    return env


def reset_duck(env, env_ids, pose_range, asset_cfg=SceneEntityCfg("object")):
    """자세·yaw·위치를 정하고, 지시 부위를 고른다."""
    _state(env)
    obj = env.scene[asset_cfg.name]
    n = len(env_ids)
    dev = env.device
    fixed = getattr(env, "duck_eval", None) or {}

    if fixed.get("target") is not None:
        env.duck_target[env_ids] = td.TARGETS.index(fixed["target"])
    else:
        env.duck_target[env_ids] = torch.randint(0, len(td.TARGETS), (n,), device=dev)
    if fixed.get("pose") is not None:
        env.duck_pose[env_ids] = POSES.index(fixed["pose"])
    else:
        env.duck_pose[env_ids] = torch.randint(0, len(POSES), (n,), device=dev)

    pose = env.duck_pose[env_ids]
    side = pose == POSES.index("side")
    sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    roll = torch.where(side, sign * torch.pi / 2, torch.zeros(n, device=dev))
    yaw = (torch.rand(n, device=dev) * 2 - 1) * torch.pi
    zero = torch.zeros(n, device=dev)
    q_tilt = quat_from_euler_xyz(roll, zero, zero)
    q_yaw = quat_from_euler_xyz(zero, zero, yaw)
    quat = quat_mul(q_yaw, q_tilt)

    root = obj.data.default_root_state[env_ids].clone()
    lo = torch.tensor([pose_range["x"][0], pose_range["y"][0]], device=dev)
    hi = torch.tensor([pose_range["x"][1], pose_range["y"][1]], device=dev)
    xy = lo + (hi - lo) * torch.rand(n, 2, device=dev)
    pos = root[:, :3] + env.scene.env_origins[env_ids]
    pos[:, :2] += xy
    pos[:, 2] = env.scene.env_origins[env_ids, 2] + torch.where(side, _POSE_Z["side"], _POSE_Z["prone"])
    obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


# ─────────────────────────────────────────────────────────────
# 기하: 지시 부위 기준점, TCP 가 잡고 있는 부위
# ─────────────────────────────────────────────────────────────
def target_anchor_w(env, object_cfg=SceneEntityCfg("object")):
    """지시 부위 기준점의 월드 좌표 [N,3]. leg 는 두 발 중 더 높은(드러난) 쪽."""
    _state(env)
    obj = env.scene[object_cfg.name]
    p, q = obj.data.root_pos_w, obj.data.root_quat_w
    n = env.num_envs
    a = env.duck_anchor  # [4,3]
    aw = quat_apply(q.unsqueeze(1).expand(n, 4, 4).reshape(-1, 4),
                    a.unsqueeze(0).expand(n, 4, 3).reshape(-1, 3)).reshape(n, 4, 3) + p.unsqueeze(1)
    leg = torch.where((aw[:, 2, 2] >= aw[:, 3, 2]).unsqueeze(-1), aw[:, 2], aw[:, 3])
    choices = torch.stack([aw[:, 0], aw[:, 1], leg], dim=1)  # [N,3,3]
    return choices[torch.arange(n, device=env.device), env.duck_target]


def tcp_w(env, ee_frame_cfg=SceneEntityCfg("ee_frame")):
    return env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]


def grasped_family(env, object_cfg=SceneEntityCfg("object")):
    """TCP 가 들어가 있는 부위 (TARGETS 인덱스)와 정규화 반경. 밖이면 -1."""
    _state(env)
    obj = env.scene[object_cfg.name]
    local = quat_apply_inverse(obj.data.root_quat_w, tcp_w(env) - obj.data.root_pos_w)  # [N,3]
    r = torch.linalg.norm((local.unsqueeze(1) - env.duck_shape_c) / env.duck_shape_s, dim=-1)  # [N,S]
    rmin, idx = r.min(dim=1)
    fam = env.duck_shape_fam[idx]
    fam = torch.where(rmin <= PART_OK_RADIUS, fam, torch.full_like(fam, -1))
    return fam, rmin


def part_ok(env):
    fam, _ = grasped_family(env)
    return (fam == env.duck_target).float()


# ─────────────────────────────────────────────────────────────
# 관측
# ─────────────────────────────────────────────────────────────
def obs_object_quat(env, object_cfg=SceneEntityCfg("object")):
    return env.scene[object_cfg.name].data.root_quat_w


def obs_target_part(env, robot_cfg=SceneEntityCfg("robot")):
    """[지시 부위 위치(로봇 기준) 3, 원-핫 3]"""
    robot = env.scene[robot_cfg.name]
    aw = target_anchor_w(env)
    pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, aw)
    onehot = torch.nn.functional.one_hot(env.duck_target, len(td.TARGETS)).float()
    return torch.cat([pos_b, onehot], dim=-1)


# ─────────────────────────────────────────────────────────────
# 보상 — 기존 Lift 항과 같은 모양, 대상만 '지시 부위'로
# ─────────────────────────────────────────────────────────────
def rew_reach_part(env, std):
    d = torch.linalg.norm(tcp_w(env) - target_anchor_w(env), dim=-1)
    return 1 - torch.tanh(d / std)


def rew_lift_part(env, minimal_height, object_cfg=SceneEntityCfg("object")):
    lifted = env.scene[object_cfg.name].data.root_pos_w[:, 2] > minimal_height
    return lifted.float() * part_ok(env)


def rew_goal_part(env, std, minimal_height, command_name, robot_cfg=SceneEntityCfg("robot"),
                  object_cfg=SceneEntityCfg("object")):
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    des_w = quat_apply(robot.data.root_quat_w, cmd[:, :3]) + robot.data.root_pos_w
    d = torch.linalg.norm(des_w - obj.data.root_pos_w, dim=1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    return lifted.float() * part_ok(env) * (1 - torch.tanh(d / std))


# ─────────────────────────────────────────────────────────────
# 평가용 성공 판정 (PCA·PPO 공용)
# ─────────────────────────────────────────────────────────────
SUCCESS_HEIGHT = 0.10      # 물체 중심 z [m]. 안착(≈0.02) 대비 8cm 이상
SUCCESS_HOLD_STEPS = 25    # 0.5 s (dt 0.02)


FINGER_CONTACT_RADIUS = 1.5   # 손가락 안쪽 면 점이 이 정규화 반경 안이면 그 형상에 닿았다고 본다
MIXED = 3                      # 두 손가락이 서로 다른 부위에 닿음


def finger_contact_family(env, object_cfg=SceneEntityCfg("object"), robot_cfg=SceneEntityCfg("robot"),
                          ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """두 손가락 안쪽 면이 닿은 부위. 반환 [N]: TARGETS 인덱스, -1(안 닿음), MIXED(서로 다름).

    처음 판정(TCP 가 부위 안)은 '두 발을 한꺼번에 쥔' 파지를 몸통/없음으로 셌다.
    TCP 가 두 발 사이 빈 공간에 있기 때문이다 (추적: TCP 로컬 (-39,2,-16)mm, 간격 27.5mm).
    그래서 각 손가락 면을 따로 본다: 손가락 면 = TCP ± (간격/2)·손 y축 (Franka 손가락이 벌어지는 축).
    """
    _state(env)
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    tcp = ee.data.target_pos_w[..., 0, :]
    q = ee.data.target_quat_w[..., 0, :]
    y = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(env.num_envs, 3))
    fid = robot.find_joints("panda_finger_joint.*")[0]
    half = robot.data.joint_pos[:, fid].sum(dim=1, keepdim=True) / 2
    fams = []
    for sgn in (1.0, -1.0):
        pw = tcp + sgn * half * y
        local = quat_apply_inverse(obj.data.root_quat_w, pw - obj.data.root_pos_w)
        r = torch.linalg.norm((local.unsqueeze(1) - env.duck_shape_c) / env.duck_shape_s, dim=-1)
        rmin, idx = r.min(dim=1)
        f = env.duck_shape_fam[idx]
        fams.append(torch.where(rmin <= FINGER_CONTACT_RADIUS, f, torch.full_like(f, -1)))
    a, b = fams
    both = torch.where(a == b, a, torch.full_like(a, MIXED))
    return torch.where((a < 0) | (b < 0), torch.full_like(a, -1), both)


def duck_step_flags(env, object_cfg=SceneEntityCfg("object")):
    """이번 스텝의 (들림, 지시부위 파지, 잡은 부위) — 하네스가 연속 유지 시간을 센다.

    지시부위 파지 = 두 손가락 면이 모두 지시 부위에 닿아 있음 (finger_contact_family).
    """
    z = env.scene[object_cfg.name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    fam = finger_contact_family(env)
    return z > SUCCESS_HEIGHT, fam == env.duck_target, fam


# ─────────────────────────────────────────────────────────────
# 환경 설정
# ─────────────────────────────────────────────────────────────
@configclass
class FrankaDuckLiftEnvCfg(ik_rel_env_cfg.FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.03], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=DUCK_USD,
                # 큐브와 같은 솔버 설정 (비교 조건을 바꾸지 않는다)
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.events.reset_object_position = EventTerm(
            func=reset_duck, mode="reset",
            params={"pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25)}},
        )
        self.observations.policy.object_quat = ObsTerm(func=obs_object_quat)
        self.observations.policy.target_part = ObsTerm(func=obs_target_part)

        self.rewards.reaching_object = RewTerm(func=rew_reach_part, params={"std": 0.1}, weight=1.0)
        self.rewards.lifting_object = RewTerm(
            func=rew_lift_part, params={"minimal_height": LIFT_HEIGHT_REWARD}, weight=15.0)
        self.rewards.object_goal_tracking = RewTerm(
            func=rew_goal_part,
            params={"std": 0.3, "minimal_height": LIFT_HEIGHT_REWARD, "command_name": "object_pose"},
            weight=16.0)
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=rew_goal_part,
            params={"std": 0.05, "minimal_height": LIFT_HEIGHT_REWARD, "command_name": "object_pose"},
            weight=5.0)


@configclass
class DuckPPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "franka_lift_duck"   # 하이퍼파라미터는 큐브와 동일


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaDuckLiftEnvCfg,
        "rsl_rl_cfg_entry_point": DuckPPORunnerCfg,
    },
)
