"""좌표·자세 규약 어댑터 검증.

규약 변환은 틀려도 예외가 안 나고 조용히 어긋난다.
그래서 왕복(round-trip)과 기지값(known-value) 둘 다로 건다.
"""
import sys
import pathlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import isaac_adapter as a  # noqa: E402


def test_quat_zyz_roundtrip():
    """쿼터니언 -> ZYZ -> 쿼터니언 왕복에서 회전이 보존된다."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        q_xyzw = Rotation.random(random_state=int(rng.integers(1 << 30))).as_quat()
        q_wxyz = np.array([q_xyzw[3], *q_xyzw[:3]])
        back = a.zyz_deg_to_quat_wxyz(*a.quat_wxyz_to_zyz_deg(q_wxyz))
        # q 와 -q 는 같은 회전이다. 회전행렬로 비교한다.
        R1 = Rotation.from_quat(q_xyzw).as_matrix()
        R2 = Rotation.from_quat([*back[1:], back[0]]).as_matrix()
        assert np.allclose(R1, R2, atol=1e-8)


def test_quat_wxyz_ordering_is_not_xyzw():
    """(w,x,y,z) 를 (x,y,z,w) 로 잘못 읽으면 걸리는 테스트."""
    # z축 90도 회전: scipy(xyzw) = [0,0,sin45,cos45]
    s = np.sqrt(0.5)
    q_wxyz = np.array([s, 0.0, 0.0, s])   # w=cos45, z=sin45
    R = Rotation.from_quat([*q_wxyz[1:], q_wxyz[0]]).as_matrix()
    assert np.allclose(R @ np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), atol=1e-9)
    # 어댑터가 같은 회전을 재현하는지
    back = a.zyz_deg_to_quat_wxyz(*a.quat_wxyz_to_zyz_deg(q_wxyz))
    R2 = Rotation.from_quat([*back[1:], back[0]]).as_matrix()
    assert np.allclose(R, R2, atol=1e-9)


def test_rotmat_to_quat_matches_scipy():
    R = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
    q = a.rotmat_to_quat_wxyz(R)
    assert np.allclose(Rotation.from_quat([*q[1:], q[0]]).as_matrix(), R, atol=1e-9)


def test_depth_backprojection_known_value():
    """광학 중심의 픽셀은 (0, 0, depth) 로 역투영된다."""
    H = W = 9
    K = np.array([[100.0, 0.0, 4.0],
                  [0.0, 100.0, 4.0],
                  [0.0, 0.0, 1.0]])
    depth = np.full((H, W), np.nan)
    depth[4, 4] = 0.5          # 주점(cx=4, cy=4) 에 0.5 m
    pts = a.depth_to_points_camera_mm(depth, K)
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0], [0.0, 0.0, 500.0], atol=1e-9)


def test_depth_backprojection_offset_pixel():
    """주점에서 1픽셀 벗어나면 z/fx 만큼 옆으로 간다."""
    K = np.array([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]])
    depth = np.full((9, 9), np.nan)
    depth[4, 5] = 0.5                      # u=5 -> (5-4)*0.5/100 m = 5mm
    pts = a.depth_to_points_camera_mm(depth, K)
    assert np.allclose(pts[0], [5.0, 0.0, 500.0], atol=1e-9)


def test_depth_filters_invalid():
    """0, NaN, 과도한 거리 값은 버린다."""
    K = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    depth = np.array([[0.0, np.nan, np.inf],
                      [10.0, 0.4, -1.0],
                      [0.0, 0.0, 0.0]])
    pts = a.depth_to_points_camera_mm(depth, K, max_depth_m=3.0)
    assert len(pts) == 1, f"유효점이 1개여야 하는데 {len(pts)}개"
    assert np.isclose(pts[0][2], 400.0)


def test_transform_points_translation_and_rotation():
    T = a.make_transform(pos=[1.0, 0.0, 0.0], quat_wxyz=[np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
                         pos_scale=a.MM_PER_M)   # m -> mm
    pts = np.array([[100.0, 0.0, 0.0]])          # mm
    out = a.transform_points(pts, T)
    # z축 90도 회전 후 x로 1000mm 평행이동
    assert np.allclose(out[0], [1000.0, 100.0, 0.0], atol=1e-9)


def test_grasp_pose_to_isaac_units():
    """mm -> m 단위 변환이 실제로 일어난다."""
    pos, quat = a.grasp_pose_to_isaac([500.0, -100.0, 250.0, 0.0, 180.0, 0.0])
    assert np.allclose(pos, [0.5, -0.1, 0.25])
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_stride_downsamples():
    K = np.array([[100.0, 0.0, 8.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]])
    depth = np.full((16, 16), 0.5)
    full = a.depth_to_points_camera_mm(depth, K, stride=1)
    half = a.depth_to_points_camera_mm(depth, K, stride=2)
    assert len(full) == 256
    assert len(half) == 64


def test_both_orientation_paths_agree():
    """ZYZ 경유 경로와 회전행렬 직행 경로가 같은 회전을 준다.

    둘이 갈라지면 어느 한쪽이 규약을 잘못 읽고 있다는 뜻이다.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from test_grasp_pca_core import run

    result, poses = run((80.0, 30.0, 30.0))
    _, q_zyz = a.grasp_pose_to_isaac(poses["grasp"])
    _, q_mat = a.grasp_result_to_isaac(result, poses["grasp"])

    R1 = Rotation.from_quat([*q_zyz[1:], q_zyz[0]]).as_matrix()
    R2 = Rotation.from_quat([*q_mat[1:], q_mat[0]]).as_matrix()
    # grasp_pose_to_isaac 은 휴리스틱 프레임을 그대로 준다.
    # grasp_result_to_isaac 은 Franka hand 규약(90도 회전)까지 적용한다.
    # 따라서 그 변환을 걸어야 같아진다.
    assert np.allclose(a.heuristic_rot_to_franka_hand(R1), R2, atol=1e-9), \
        np.abs(a.heuristic_rot_to_franka_hand(R1) - R2).max()


def test_grasp_result_position_is_meters():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from test_grasp_pca_core import run

    result, poses = run((80.0, 30.0, 30.0))
    pos, _ = a.grasp_result_to_isaac(result, poses["grasp"])
    assert 0.2 < np.linalg.norm(pos) < 1.5, f"위치 크기가 m 단위로 보이지 않는다: {pos}"


def test_franka_hand_closing_axis_is_rotated_90deg():
    """휴리스틱의 닫힘축(tool_x)이 Franka 손가락 축(y)에 오도록 돌아간다.

    이 변환이 빠지면 정사각 물체에서는 아무 증상이 없다가(대칭),
    길쭉한 물체에서 갑자기 실패한다. 실측: 84x42mm 물체에서 0% -> 100%.
    """
    R = np.eye(3)
    Rf = a.heuristic_rot_to_franka_hand(R)

    # 원래 tool_x 였던 축이 변환 후 y 열에 와야 한다.
    # 부호는 상관없다 — 그리퍼는 대칭으로 닫히므로 닫힘축은 방향이 아니라 '선'이다.
    # (그래서 +90 과 -90 은 물리적으로 같고, 시뮬에서 +90 을 검증했다)
    assert np.allclose(np.abs(Rf[:, 1]), np.abs(R[:, 0]), atol=1e-9), Rf
    # 접근축(tool_z)은 보존되어야 한다
    assert np.allclose(Rf[:, 2], R[:, 2], atol=1e-9), Rf
    # 직교성 유지
    assert np.allclose(Rf @ Rf.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(Rf), 1.0, atol=1e-9)


def test_grasp_result_to_isaac_applies_hand_convention():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from test_grasp_pca_core import run

    result, poses = run((80.0, 30.0, 30.0))
    _, q = a.grasp_result_to_isaac(result, poses["grasp"])
    R_out = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
    R_expected = a.heuristic_rot_to_franka_hand(result["rotation_matrix"])
    assert np.allclose(R_out, R_expected, atol=1e-9)
