"""오리 근사 모형의 형상·부위 판정 검증 (Isaac Sim 불필요)."""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import toy_duck as td  # noqa: E402

FRANKA_MAX_OPENING_MM = 80.0


def test_parts_match_team_project_vocabulary():
    # cobot2_jarvis 의 부위 어휘(head/body/leg) 안에 있어야 한다
    assert set(td.SEMANTIC_OF.values()) == {"head", "body", "leg"}
    assert td.PART_NAMES == ("head", "body", "leg1", "leg2")


def test_overall_size_is_the_stated_assumption():
    lo, hi = td.bounding_box_m()
    length_mm = (hi[0] - lo[0]) * 1000
    assert 135 <= length_mm <= 155, length_mm   # docstring 의 '약 145 mm'
    # 배율이 그대로 반영된다
    lo2, hi2 = td.bounding_box_m(scale=2.0)
    assert np.isclose((hi2[0] - lo2[0]) * 1000, 2 * length_mm)


def test_every_part_has_a_grippable_width():
    """각 부위에 Franka 가 벌릴 수 있는 폭(<80mm)인 방향이 하나 이상 있어야 한다.

    없으면 그 부위를 '지시'하는 조건 자체가 물리적으로 불가능하다.
    """
    for name, dims in td.part_widths_mm().items():
        assert min(dims) < FRANKA_MAX_OPENING_MM, (name, dims)


def test_shape_centers_classify_to_their_part():
    for name, _, c, _ in td.shapes():
        assert td.classify_points(c[None])[0] == name


def test_far_point_is_outside():
    assert td.classify_points(np.array([[1.0, 1.0, 1.0]]))[0] is None


def test_surface_labels_agree_with_classifier():
    pts, labels = td.sample_surface(3000, seed=1)
    pred = np.asarray(td.classify_points(pts))
    agree = (pred == labels).mean()
    # 겹침 경계에서만 어긋날 수 있다
    assert agree > 0.97, agree
    assert set(labels) == set(td.PART_NAMES)


def test_buried_points_are_removed():
    """목·발 이음부처럼 다른 형상 안에 묻힌 점은 바깥에서 안 보이므로 빠져야 한다."""
    pts, _ = td.sample_surface(3000, seed=2)
    r = td.normalized_radius(pts)
    # 모든 점은 어떤 형상의 내부(<0.999)에도 있지 않아야 한다 (자기 표면 = 1)
    assert (r.min(axis=1) > 0.999 - 1e-9).all()


def test_mesh_is_closed_and_on_surface():
    c, s = np.array([0.01, 0.0, 0.0]), np.array([0.03, 0.02, 0.01])
    v, f = td.ellipsoid_mesh(c, s)
    assert np.allclose(np.linalg.norm((v - c) / s, axis=1), 1.0)
    # 닫힌 메시: 모든 모서리가 정확히 두 면에 속한다
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(edges, axis=0, return_counts=True)
    assert (cnt == 2).all()
    # 바깥을 향하는 감김 (부호 있는 부피 > 0)
    vol = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6
    assert vol > 0


def test_collision_mesh_fits_gpu_hull_limit_and_hugs_surface():
    c, sm = np.array([0.0, 0.0, 0.0]), np.array([0.038, 0.0255, 0.0225])
    v, f, e = td.collision_mesh(c, sm)
    assert len(v) <= 64                       # PhysX 단순화가 일어나지 않는 크기
    assert 0 < e < 0.06                       # 반축 대비 최대 오차 6% 미만
    # 꼭짓점은 해석적 표면보다 정확히 (1+e) 배 바깥
    assert np.allclose(np.linalg.norm((v - c) / sm, axis=1), 1 + e)


def test_visible_from_above_hides_underside():
    pts, lab = td.sample_surface(6000, seed=3)
    vis, vlab = td.visible_from_above(pts, lab)
    assert 0 < len(vis) < len(pts)
    # 엎드린 자세에서 보이는 점은 대부분 몸통 중심보다 위에 있다
    assert np.median(vis[:, 2]) > 0
    assert set(vlab) <= set(lab)


def test_shape_arrays_families():
    c, s, fam = td.shape_arrays()
    assert len(c) == len(s) == len(fam) == 5
    assert [td.TARGETS[f] for f in fam] == ["head", "head", "body", "leg", "leg"]
