"""IK-Rel Lift-Cube 를 RL 학습이 가능하도록 등록한다.

왜 IK-Rel 인가
--------------
비교 실험에서 달라져도 되는 변수는 '제어 방식' 하나뿐이다(원칙 1).
그런데 Isaac Lab 이 RL 설정을 붙여 둔 태스크는 `Isaac-Lift-Cube-Franka-v0`
하나뿐이고 그건 **관절 위치 액션**을 쓴다. 반면 휴리스틱은 파지 '자세'를
내놓으므로 태스크 공간(IK) 제어가 자연스럽다.
그대로 비교하면 '액션 인터페이스'라는 변수가 하나 더 섞인다.

그래서 **양쪽 모두 IK-Rel(상대 자세 델타)로 맞춘다.**
- IK-Abs 는 매 스텝 절대 자세를 내야 해서 RL 에 불리하다.
- IK-Rel 은 델타라서 RL 에 자연스럽고, 휴리스틱 쪽은 상태기계가
  '목표까지의 델타'를 내보내면 되므로 맞추기 쉽다.

액션: [dx, dy, dz, drx, dry, drz, gripper] = 7 차원 (scale 0.5)
"""
import gymnasium as gym

from isaaclab_tasks.manager_based.manipulation.lift.config.franka import ik_rel_env_cfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka import agents

TASK_ID = "IsaacGraspBench-Lift-Cube-Franka-IK-Rel-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaCubeLiftEnvCfg,
        # PPO 하이퍼파라미터는 Isaac Lab 의 Lift-Cube 설정을 그대로 쓴다.
        # 내가 임의로 손대면 '제어 방식' 외의 변수가 늘어난다.
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
)
