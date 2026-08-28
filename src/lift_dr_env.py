"""도메인 랜덤화 환경 — 강도를 파라미터로 둔다 (Phase 3).

목적은 '랜덤화하면 좋아진다'가 아니라 **강도별 성공률 곡선**을 그리는 것이다.
sim-to-real 갭에 대해 말이 아니라 숫자로 답하는 부분이라서, 강도를 0 부터
연속적으로 올려가며 같은 지표를 재야 한다.

랜덤화 축
---------
물리만 한다 — 마찰(정/동), 반발, 물체 질량.
조명과 카메라 외참은 **의도적으로 뺐다.** 이 인스턴스에서 RTX 렌더러가
초기화되지 않아 렌더링 경로를 쓸 수 없고(docs/WORKLOG.md 4번),
현재 관측이 합성 점군이라 조명이 관측에 영향을 주지도 않는다.
안 한 것을 한 것처럼 적지 않기 위해 여기 남긴다.

강도 s (0.0 ~ 1.0)
------------------
  마찰(정)   0.8 -> [0.8 - 0.6s, 0.8 + 0.6s]
  마찰(동)   0.6 -> [0.6 - 0.45s, 0.6 + 0.45s]
  반발       0.0 -> [0, 0.4s]
  질량 배율  1.0 -> [1 - 0.6s, 1 + 0.6s]

s=0 이면 모든 범위가 한 점으로 붙어 랜덤화가 없는 것과 같다 — 기준선이 된다.
"""
import gymnasium as gym
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka import ik_rel_env_cfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka import agents

TASK_PREFIX = "IsaacGraspBench-Lift-Cube-Franka-IK-Rel-DR"
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _rng(center, half, s):
    """강도 s 에서의 범위. s=0 이면 [center, center] 로 붙는다."""
    return (center - half * s, center + half * s)


def make_cfg_class(strength: float):
    s = float(strength)

    @configclass
    class _Cfg(ik_rel_env_cfg.FrankaCubeLiftEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            # 물체의 마찰·반발
            self.events.object_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "static_friction_range": _rng(0.8, 0.6, s),
                    "dynamic_friction_range": _rng(0.6, 0.45, s),
                    "restitution_range": (0.0, 0.4 * s),
                    "num_buckets": 64,
                },
            )
            # 물체 질량
            self.events.object_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "mass_distribution_params": _rng(1.0, 0.6, s),
                    "operation": "scale",
                },
            )

    _Cfg.__name__ = f"FrankaCubeLiftDREnvCfg_s{int(s*100):03d}"
    return _Cfg


def task_id(strength: float) -> str:
    return f"{TASK_PREFIX}{int(float(strength)*100):03d}-v0"


for _s in STRENGTHS:
    gym.register(
        id=task_id(_s),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": make_cfg_class(_s),
            # 하이퍼파라미터는 건드리지 않는다 — 달라지는 변수는 랜덤화 강도뿐이다.
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        },
    )
