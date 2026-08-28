"""Lift-Cube (IK-Abs) + 상단 뎁스 카메라 환경.

베이스라인(PCA 휴리스틱)은 물체의 실제 자세를 모른 채 뎁스 점군만 보고
파지 자세를 계산해야 한다. 그래서 기본 Lift-Cube 환경에 카메라를 붙인다.

액션 인터페이스로 IK-Abs 를 쓰는 이유: 휴리스틱의 출력이 "절대 파지 자세"이므로
그것을 그대로 명령으로 넣을 수 있어야 한다.

⚠ RL 정책과 비교할 때 두 방식은 **같은 환경·같은 액션 인터페이스**를 써야 한다.
  카메라 유무로 물리가 달라지지는 않지만, 렌더링 부하는 달라진다 —
  사이클타임을 비교할 때 이 점을 감안해야 한다.
"""
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka import ik_abs_env_cfg


# 카메라를 물체 위에 두고 수직으로 내려다본다.
# ROS 관례(광학축 +Z, 아래 +Y)에서 광학축을 월드 -Z 로 향하게 하려면
# X 축 기준 180도 회전 -> 쿼터니언 (w,x,y,z) = (0, 1, 0, 0).
TOPDOWN_CAM_POS = (0.5, 0.0, 0.8)
TOPDOWN_CAM_ROT_WXYZ = (0.0, 1.0, 0.0, 0.0)


@configclass
class LiftCubeCameraEnvCfg(ik_abs_env_cfg.FrankaCubeLiftEnvCfg):
    """IK-Abs Lift-Cube 에 상단 뎁스 카메라를 추가한 구성."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.topdown_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/topdown_cam",
            update_period=0.0,
            height=240,
            width=240,
            # 점군만 필요하므로 뎁스만 받는다. rgb 는 렌더링 비용만 늘린다.
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=TOPDOWN_CAM_POS, rot=TOPDOWN_CAM_ROT_WXYZ, convention="ros"
            ),
        )
        # 리셋 직후 카메라 버퍼가 이전 프레임을 들고 있지 않도록 재렌더링한다.
        self.num_rerenders_on_reset = 3


gym.register(
    id="IsaacGraspBench-Lift-Cube-Franka-Cam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": LiftCubeCameraEnvCfg},
    disable_env_checker=True,
)
