"""오리 근사 모형을 Isaac Sim 에 넣고 물리·충돌·부위 라벨이 맞는지 확인한다.

  source scripts/env.sh
  $PY scripts/check_duck_asset.py --out /data/isaac-grasp-bench/eval/duck_asset_check

자세 6개로 바닥 위 8cm 에서 떨어뜨리고 3초 뒤를 본다:
  1) 정지했는가          — 선속도·각속도
  2) 바닥을 뚫거나 떠 있는가 — 표면 샘플점의 최저 z
  3) 질량이 USD 대로인가  — PhysX 가 읽은 값
  4) 카메라 부위 라벨     — 의미 분할(head/body/leg) 픽셀을 뎁스로 3D 복원해
                            물체 로컬로 옮긴 뒤 형상 기반 판정(toy_duck.classify_points)과 비교
이 스크립트가 재는 것은 '모형이 시뮬레이터에서 의도대로 표현되는가'이고,
그리퍼 파지는 재지 않는다.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default=str(ROOT / "assets/duck/duck.usda"))
parser.add_argument("--out", default="/data/isaac-grasp-bench/eval/duck_asset_check")
parser.add_argument("--seconds", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

import toy_duck as td  # noqa: E402

# (이름, 회전) — 로컬 +z 가 등. scipy 는 xyzw, Isaac 은 wxyz.
POSES = [
    ("prone", Rotation.identity()),
    ("on_back", Rotation.from_euler("x", 180, degrees=True)),
    ("left_side", Rotation.from_euler("x", 90, degrees=True)),
    ("right_side", Rotation.from_euler("x", -90, degrees=True)),
    ("beak_down", Rotation.from_euler("y", 90, degrees=True)),
    ("tail_down", Rotation.from_euler("y", -90, degrees=True)),
]
SPACING = 0.35
DROP_Z = 0.08
CAM_Z = 0.45
REF_RADIUS = 0.025
CAM_ROT_ROS = (0.0, 1.0, 0.0, 0.0)   # wxyz, 광학축이 월드 -Z 를 향함
SEM_COLORS = {"head": (220, 40, 60), "body": (40, 170, 220), "leg": (230, 60, 230)}


def wxyz(rot):
    x, y, z, w = rot.as_quat()
    return (float(w), float(x), float(y), float(z))


def depth_to_world(depth, K, cam_pos, cam_quat_wxyz_ros):
    """핀홀 역투영. ROS 광학 좌표(+z 전방, +x 오른쪽, +y 아래) → 월드. 반환 [H*W, 3]."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.reshape(-1)
    x = (u.reshape(-1) + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v.reshape(-1) + 0.5 - K[1, 2]) * z / K[1, 1]
    qw, qx, qy, qz = cam_quat_wxyz_ros
    return Rotation.from_quat([qx, qy, qz, qw]).apply(np.stack([x, y, z], 1)) + cam_pos


def main():
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    sim.set_camera_view([1.0, -1.5, 1.0], [0.9, 0.0, 0.0])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    spawn = sim_utils.UsdFileCfg(usd_path=args.usd)
    cams = []
    for i, (_, rot) in enumerate(POSES):
        x = i * SPACING
        spawn.func(f"/World/Duck_{i}", spawn, translation=(x, 0.0, DROP_Z), orientation=wxyz(rot))
        cams.append(Camera(CameraCfg(
            prim_path=f"/World/Cam_{i}", update_period=0.0, height=240, width=240,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                             clipping_range=(0.05, 2.0)),
            offset=CameraCfg.OffsetCfg(pos=(x, 0.0, CAM_Z), rot=CAM_ROT_ROS,
                                       convention="ros"),
        )))
    duck = RigidObject(RigidObjectCfg(prim_path="/World/Duck_.*", spawn=None))
    # 대조군: 형상이 정확히 알려진 해석적 구. 바닥 기준면·접촉 오프셋을 오리와 분리해 잰다.
    ref = RigidObject(RigidObjectCfg(
        prim_path="/World/RefSphere",
        spawn=sim_utils.SphereCfg(
            radius=REF_RADIUS, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=td.MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-SPACING, 0.0, DROP_Z))))

    sim.reset()
    dt = sim.get_physics_dt()
    # 낙하·정지 구간은 물리만 돈다. 카메라 6대를 매 스텝 렌더링하면 수십 배 느려진다.
    for _ in range(int(args.seconds / dt)):
        sim.step(render=False)
        duck.update(dt)
        ref.update(dt)
    for _ in range(4):               # 렌더 버퍼가 마지막 상태를 따라오도록
        sim.step(render=True)
        duck.update(dt)
    for c in cams:
        c.update(dt)

    masses = duck.root_physx_view.get_masses().reshape(-1).tolist()
    pos = duck.data.root_pos_w.cpu().numpy()
    quat = duck.data.root_quat_w.cpu().numpy()          # wxyz
    lin = torch.linalg.norm(duck.data.root_lin_vel_w, dim=1).cpu().numpy()
    ang = torch.linalg.norm(duck.data.root_ang_vel_w, dim=1).cpu().numpy()
    surf, _ = td.sample_surface(6000, seed=0)

    results = []
    for i, (name, _) in enumerate(POSES):
        R = Rotation.from_quat([quat[i][1], quat[i][2], quat[i][3], quat[i][0]])
        world = R.apply(surf) + pos[i]
        min_z_mm = float(world[:, 2].min() * 1000)
        up_local = R.inv().apply([0, 0, 1])               # 로컬에서 본 월드 위쪽
        rest = {"pose": name, "final_pos_m": pos[i].round(4).tolist(),
                "lin_speed_mm_s": round(float(lin[i]) * 1000, 3),
                "ang_speed_deg_s": round(float(np.degrees(ang[i])), 3),
                "lowest_surface_z_mm": round(min_z_mm, 2),
                "world_up_in_local": np.round(up_local, 3).tolist(),
                "mass_kg": round(masses[i], 5)}

        # ── 카메라: 의미 분할 → 3D → 로컬 → 형상 판정과 비교
        cam = cams[i]
        sem = cam.data.output["semantic_segmentation"][0, ..., 0].cpu().numpy()
        info = cam.data.info[0]["semantic_segmentation"]["idToLabels"]
        id2lab = {int(k): v.get("class", "") for k, v in info.items()}
        depth = cam.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
        K = cam.data.intrinsic_matrices[0].cpu().numpy()
        # cam.data.pos_w / quat_w_ros 가 이 스크립트 흐름에서는 0 으로 비어 있었다.
        # 카메라는 고정이므로 직접 준 설정값을 쓴다.
        pts = depth_to_world(depth, K, np.array([i * SPACING, 0.0, CAM_Z]), CAM_ROT_ROS)
        labs = np.vectorize(lambda k: id2lab.get(int(k), ""))(sem.reshape(-1))
        on_duck = np.isin(labs, list(SEM_COLORS)) & np.isfinite(pts).all(axis=1)
        local = R.inv().apply(pts[on_duck] - pos[i])
        geo = np.array([td.SEMANTIC_OF.get(p, "none") if p else "none"
                        for p in td.classify_points(local)])
        cam_lab = labs[on_duck]
        pix = {k: int((cam_lab == k).sum()) for k in SEM_COLORS}
        rest["camera_pixels_by_label"] = pix
        rest["label_agreement"] = (round(float((geo == cam_lab).mean()), 4)
                                   if len(cam_lab) else None)
        rest["camera_points_outside_model"] = int((geo == "none").sum())
        rest["id_to_labels"] = {str(k): v for k, v in id2lab.items()}
        results.append(rest)

        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        over = rgb.copy()
        flat = over.reshape(-1, 3)
        for k, col in SEM_COLORS.items():
            flat[labs == k] = (0.4 * flat[labs == k] + 0.6 * np.array(col)).astype(np.uint8)
        Image.fromarray(np.concatenate([rgb, over], axis=1)).save(out / f"{i}_{name}.png")

    ref_z = float(ref.data.root_pos_w[0, 2])
    reference = {"sphere_radius_mm": REF_RADIUS * 1000,
                 "sphere_center_z_mm": round(ref_z * 1000, 2),
                 "sphere_lowest_z_mm": round((ref_z - REF_RADIUS) * 1000, 2)}
    summary = {"usd": args.usd, "device": args.device, "reference_sphere": reference, "mass_expected_kg": td.MASS_KG,
               "friction": [td.STATIC_FRICTION, td.DYNAMIC_FRICTION],
               "seconds": args.seconds, "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    sys.__stdout__.write(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")


import os  # noqa: E402
import traceback  # noqa: E402

try:
    main()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
# app.close() 가 카메라 6대를 정리하다 멈추는 일이 있었다. 결과는 이미 저장됐으므로 바로 끝낸다.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
