"""합성 점군 생성기 검증."""
import sys
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import synthetic_cloud as sc  # noqa: E402

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def test_returns_mm_not_meters():
    pts = sc.sample_visible_box_surface((0.05, 0.05, 0.05), (0.5, 0.0, 0.1), IDENTITY, seed=0)
    # 중심이 (500, 0, 100) mm 근처여야 한다
    assert np.allclose(pts.mean(axis=0), [500.0, 0.0, 100.0], atol=30.0)


def test_only_visible_faces_sampled():
    """위에서 내려다보면 아래쪽 면(-Z)의 점은 나오지 않는다."""
    size = (0.06, 0.06, 0.06)
    pts = sc.sample_visible_box_surface(size, (0.5, 0.0, 0.1), IDENTITY,
                                        view_dir=(0, 0, -1), n_points=2000, seed=1)
    z = pts[:, 2]
    bottom_z = (0.1 - 0.03) * sc.MM_PER_M      # 70 mm
    # 바닥면에 놓인 점이 사실상 없어야 한다
    assert (z < bottom_z + 1.0).sum() == 0, "가려진 아래쪽 면이 샘플링됐다"
    # 윗면(130mm)은 있어야 한다
    assert (z > 0.1 * sc.MM_PER_M + 29.0).sum() > 0, "보이는 윗면이 비었다"


def test_visible_face_count_from_above():
    """축 정렬 상태에서 위에서 보면 윗면 1개만 마주본다.

    (옆면들은 법선이 시선과 수직이라 내적이 0 -> 제외된다)
    """
    pts = sc.sample_visible_box_surface((0.06, 0.06, 0.06), (0.5, 0.0, 0.1), IDENTITY,
                                        view_dir=(0, 0, -1), n_points=1000, seed=2)
    z = np.unique(np.round(pts[:, 2], 6))
    assert len(z) == 1, f"평면 하나여야 하는데 z 값이 {len(z)}종류다"
    assert np.isclose(z[0], 130.0)


def test_tilted_object_shows_more_faces():
    """기울이면 옆면도 보이기 시작한다 — 부분 관측성이 자세에 의존한다."""
    q = Rotation.from_euler('y', 30, degrees=True).as_quat()   # xyzw
    quat = (q[3], q[0], q[1], q[2])
    pts = sc.sample_visible_box_surface((0.06, 0.06, 0.06), (0.5, 0.0, 0.1), quat,
                                        view_dir=(0, 0, -1), n_points=2000, seed=3)
    # 두 면이 보이므로 z 값이 하나의 평면에 몰리지 않는다
    assert pts[:, 2].std() > 3.0, "기울였는데 한 면만 보인다"


def test_noise_increases_spread():
    args = dict(size_m=(0.06, 0.06, 0.06), pos_m=(0.5, 0.0, 0.1),
                quat_wxyz=IDENTITY, n_points=1500, seed=4)
    clean = sc.sample_visible_box_surface(**args, noise_mm=0.0)
    noisy = sc.sample_visible_box_surface(**args, noise_mm=2.0)
    assert noisy[:, 2].std() > clean[:, 2].std() + 0.5


def test_deterministic_with_seed():
    a = sc.sample_visible_box_surface((0.05, 0.03, 0.03), (0.5, 0.1, 0.1), IDENTITY, seed=7)
    b = sc.sample_visible_box_surface((0.05, 0.03, 0.03), (0.5, 0.1, 0.1), IDENTITY, seed=7)
    assert np.array_equal(a, b), "같은 시드인데 결과가 다르다 — 재현성이 깨진다"


def test_heuristic_accepts_this_cloud():
    """생성한 점군을 휴리스틱이 실제로 처리한다 (통합)."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    import grasp_pca_core as g

    # 얇고 긴 물체: 주축이 뚜렷해 PCA 가 잘 잡히는 조건
    pts = sc.sample_visible_box_surface((0.09, 0.03, 0.03), (0.5, 0.0, 0.1), IDENTITY,
                                        n_points=800, seed=11)
    result, poses = g.compute_grasp(
        points_base_mm=pts, part_names=["cube"], part_num_points=[len(pts)],
        target_part="cube",
        capture_posx=(400.0, 0.0, 600.0, 0.0, 180.0, 0.0),
        T_gripper2camera=np.eye(4))
    assert result["graspable"] is True
    assert np.all(np.isfinite(np.asarray(poses["grasp"], dtype=float)))


def test_near_identity_quaternion_does_not_leak_side_faces():
    """시뮬레이터가 주는 쿼터니언의 수치 잔차가 옆면을 새게 하면 안 된다.

    실제 Isaac Sim 이 준 값이다. 정확한 항등원이 아니다(기울기 0.03도).
    가시성 임계값이 이 잔차보다 작으면 시선과 수직인 옆면 4개가
    노이즈만으로 '보이는 면'이 되어 점군 중심이 물체 중심에서 크게 벗어난다.
    """
    noisy_quat = np.array([9.9999994e-01, -4.5055594e-06, -4.6998775e-06, 7.8002444e-07],
                          dtype=np.float32)
    pos = np.array([0.47861767, 0.01898849, 0.02099994], dtype=np.float32)
    size = np.array([0.048, 0.048, 0.048])

    pts = sc.sample_visible_box_surface(size, pos, noisy_quat,
                                        view_dir=(0.0, 0.0, -1.0), n_points=600, seed=1)

    # 윗면 하나만 보여야 한다 -> z 가 사실상 한 평면
    assert pts[:, 2].std() < 0.5, f"옆면이 샘플링됐다 (z 표준편차 {pts[:,2].std():.2f}mm)"

    # 점군 중심의 xy 가 물체 중심과 일치해야 한다
    xy_err = np.linalg.norm(pts.mean(axis=0)[:2] - pos[:2] * sc.MM_PER_M)
    assert xy_err < 3.0, f"점군 중심이 물체 중심에서 {xy_err:.1f}mm 벗어났다"
