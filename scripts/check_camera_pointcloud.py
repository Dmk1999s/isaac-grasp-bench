"""통합 검증: 뎁스 카메라 점군이 실제 물체 위치와 맞는가.

여기가 틀리면 이후 베이스라인 전체가 무의미해진다. 그래서 물체의 실제 자세
(ground truth)와 점군에서 추정한 위치를 직접 대조한다.
평가에서는 ground truth 를 쓰지 않는다 — 이 검증에서만 쓴다.
"""
import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# --headless 와 --enable_cameras 는 커맨드라인으로 넘긴다.
# parse_args() 뒤에 속성을 덮어쓰면 AppLauncher 의 experience 파일 결정이
# 어긋나 GUI 뷰포트를 띄우려다 세그폴트가 난다.

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 이하 Isaac 기동 후에만 import 가능 ────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import isaac_adapter as adapter  # noqa: E402
import lift_camera_env  # noqa: E402,F401  (gym 등록 부수효과)

TASK = "IsaacGraspBench-Lift-Cube-Franka-Cam-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()

    # 카메라 버퍼가 채워지도록 몇 스텝 돌린다
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    for _ in range(6):
        env.step(actions)

    cam = env.unwrapped.scene["topdown_cam"]
    obj = env.unwrapped.scene["object"]
    origins = env.unwrapped.scene.env_origins

    depth = cam.data.output["distance_to_image_plane"]
    print(f"\n[뎁스] shape={tuple(depth.shape)} dtype={depth.dtype}")
    print(f"[뎁스] 유효 픽셀 비율={torch.isfinite(depth).float().mean().item():.3f}")

    K = cam.data.intrinsic_matrices[0].cpu().numpy()
    print(f"[내참] fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    ok = True
    for i in range(min(args_cli.num_envs, 4)):
        d = depth[i].squeeze(-1).cpu().numpy()
        pts_cam_mm = adapter.depth_to_points_camera_mm(d, K, stride=2)
        if len(pts_cam_mm) < 50:
            print(f"[env {i}] 유효점 부족: {len(pts_cam_mm)}")
            ok = False
            continue

        # 카메라 -> 월드 (ROS 관례 쿼터니언), mm 단위로 맞춘다
        cam_pos_m = cam.data.pos_w[i].cpu().numpy()
        cam_quat = cam.data.quat_w_ros[i].cpu().numpy()
        T_world_cam = adapter.make_transform(cam_pos_m, cam_quat, pos_scale=adapter.MM_PER_M)
        pts_world_mm = adapter.transform_points(pts_cam_mm, T_world_cam)

        # env 원점 기준(= 로봇 base 기준)으로 옮긴다
        origin_mm = origins[i].cpu().numpy() * adapter.MM_PER_M
        pts_base_mm = pts_world_mm - origin_mm

        # 테이블 위 물체만 남긴다 (바닥/테이블 제거)
        z = pts_base_mm[:, 2]
        obj_pts = pts_base_mm[(z > 15.0) & (z < 200.0)]

        gt_mm = (obj.data.root_pos_w[i].cpu().numpy() - origins[i].cpu().numpy()) * adapter.MM_PER_M

        if len(obj_pts) < 30:
            print(f"[env {i}] 물체 점 부족: {len(obj_pts)} (전체 {len(pts_base_mm)})"
                  f" / z범위 {z.min():.0f}~{z.max():.0f}mm")
            ok = False
            continue

        est = obj_pts.mean(axis=0)
        err_xy = np.linalg.norm(est[:2] - gt_mm[:2])
        print(f"[env {i}] 물체점={len(obj_pts):5d}  "
              f"추정 xy=({est[0]:7.1f},{est[1]:7.1f})  "
              f"실제 xy=({gt_mm[0]:7.1f},{gt_mm[1]:7.1f})  xy오차={err_xy:6.1f}mm")
        if err_xy > 30.0:
            ok = False

    print("\n=== 결과:", "통과" if ok else "실패", "===")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
