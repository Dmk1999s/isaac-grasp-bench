"""이식한 PCA 파지 휴리스틱의 기능 검증.

목적은 알고리즘의 정확도 평가가 아니다. 이식 과정에서 무언가 깨지지 않았는지,
그리고 그리퍼 상수 교체가 실제로 판정에 반영되는지를 확인한다.
Isaac Sim 없이 순수 numpy/scipy 로 돌아야 한다 — 그게 이식의 목표였다.
"""
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import grasp_pca_core as g  # noqa: E402


def make_box_points(size_mm, center_mm=(500.0, 0.0, 100.0), n=800, seed=0):
    """축 정렬 직육면체 표면의 합성 점군 (base 좌표계, mm)."""
    rng = np.random.default_rng(seed)
    half = np.asarray(size_mm, dtype=np.float64) / 2.0
    pts = rng.uniform(-1.0, 1.0, size=(n, 3)) * half
    # 각 점을 가장 가까운 면으로 밀어 표면에 붙인다
    ratio = np.abs(pts) / half
    face = np.argmax(ratio, axis=1)
    pts[np.arange(n), face] = np.sign(pts[np.arange(n), face]) * half[face]
    return pts + np.asarray(center_mm, dtype=np.float64)


# 카메라가 위에서 아래를 내려다보는 자세 (ZYZ 오일러, 도 단위 — M0609 규약)
CAPTURE_POSX = (400.0, 0.0, 600.0, 0.0, 180.0, 0.0)
# 그리퍼->카메라 외참: 손끝에서 조금 뒤/위 (단위 mm)
T_GRIPPER2CAMERA = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -60.0],
    [0.0, 0.0, 1.0, -40.0],
    [0.0, 0.0, 0.0, 1.0],
])


def run(size_mm, **kw):
    pts = make_box_points(size_mm)
    return g.compute_grasp(
        points_base_mm=pts,
        part_names=["cube"],
        part_num_points=[len(pts)],
        target_part="cube",
        capture_posx=CAPTURE_POSX,
        T_gripper2camera=T_GRIPPER2CAMERA,
        **kw)


def test_runs_without_isaac_or_ros():
    """Isaac Sim / ROS2 없이 파지 자세가 나온다."""
    result, poses = run((80.0, 30.0, 30.0))
    assert result["graspable"] is True
    assert poses, "파지 자세가 비어 있다"
    assert "sensor_msgs" not in sys.modules
    assert "rclpy" not in sys.modules


def test_grasp_width_tracks_narrow_dimension():
    """닫힘축 폭은 물체의 좁은 쪽 치수를 따라간다."""
    result, _ = run((80.0, 30.0, 30.0))
    # 표면 샘플링 + 백분위수(2.5%) 절단이 있으므로 정확히 30 은 아니다.
    assert 20.0 < result["grasp_width_mm"] < 40.0, result["grasp_width_mm"]


def test_pose_is_finite_and_near_object():
    """산출된 자세가 유한하고 물체 근처에 있다."""
    result, poses = run((80.0, 30.0, 30.0))
    centroid = np.asarray(result["centroid"], dtype=np.float64)
    assert np.all(np.isfinite(centroid))
    # 물체 중심 (500, 0, 100) 근처여야 한다
    assert np.linalg.norm(centroid - np.array([500.0, 0.0, 100.0])) < 60.0

    # 회전행렬이 정상적인 직교 행렬인지 (이식 중 축 계산이 깨지면 여기서 걸린다)
    R = np.asarray(result["rotation_matrix"], dtype=np.float64)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "회전행렬이 직교가 아니다"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "det(R) != 1"


def test_pregrasp_is_backed_off_along_approach():
    """pregrasp 는 grasp 보다 접근축 뒤쪽에 있어야 한다."""
    result, poses = run((80.0, 30.0, 30.0))
    grasp = np.asarray(poses["grasp"][:3], dtype=np.float64)
    pregrasp = np.asarray(poses["pregrasp"][:3], dtype=np.float64)
    assert np.all(np.isfinite(grasp)) and np.all(np.isfinite(pregrasp))
    gap = np.linalg.norm(pregrasp - grasp)
    assert gap > 1.0, f"pregrasp 가 grasp 와 사실상 같은 위치다 (간격 {gap:.2f}mm)"


def test_franka_width_limit_rejects_what_rg2_would_accept():
    """상수 교체가 실제로 판정을 바꾼다.

    폭 90mm 물체는 RG2(110mm)로는 잡을 수 있지만 Franka(80mm)로는 못 잡는다.
    이 테스트가 통과한다는 것은 그리퍼 상수 교체가 형식적이지 않았다는 뜻이다.
    """
    assert g.GRIPPER_MAX_WIDTH_MM == 80.0
    with pytest.raises(g.GraspTooWideError):
        run((80.0, 90.0, 90.0))


def test_too_few_points_raises():
    """점이 부족하면 조용히 이상한 값을 내놓지 않고 실패한다."""
    pts = make_box_points((80.0, 30.0, 30.0), n=10)
    with pytest.raises(ValueError):
        g.compute_grasp(
            points_base_mm=pts, part_names=["cube"], part_num_points=[len(pts)],
            target_part="cube", capture_posx=CAPTURE_POSX,
            T_gripper2camera=T_GRIPPER2CAMERA)
