"""오리 장난감 근사 모형 — 형상·부위·물성 정의 (순수 numpy, Isaac Sim 불필요).

⚠ 이것은 '실제 장난감'이 아니라 **사진 1장에서 비율을 잰 근사 모형**이다.
   3D 스캔·실측 치수·무게가 없다. 결과를 인용할 때 반드시 이 점을 같이 적는다.

출처
----
- 물체 종류와 부위: 팀 프로젝트 cobot2_jarvis
    물체 5종 (bag, cube, doll, duck, robot), 부위 (head, body, arm, leg)
    config.OBJECT_MATERIAL_MAP["duck"] = "rubber"
- 형상 비율: 사용자가 준 사진 1장 (assets/duck/reference_photo.png).
    Mask R-CNN 부위 윤곽(빨강=head, 파랑=body, 분홍=leg×2)이 그려진 위에서 본 영상.
    오리가 엎드린 자세로 보인다: 부리 → 머리 → 몸통 → 뒤로 뻗은 발 2개.

사진에서 잰 것 (단위: 픽셀, 몸통 장축 u / 수직축 n 기준)
-----------------------------------------------------
  몸통 길이 144, 폭 97
  머리 폭 ~90, 부리 돌출 ~40 · 폭 ~45
  발 크기 ~30, 몸통 끝보다 ~20 더 나옴, 좌우 간격 ~38
  부리 끝 ~ 발 끝 전체 길이 ~283

가정 (사진으로 알 수 없는 것 — 전부 파라미터로 바꿀 수 있다)
--------------------------------------------------------------
  절대 크기  : 전체 길이 ≈ 145 mm  (→ 0.53 mm/px)
               근거 없음. 반려견 장난감 오리의 흔한 크기를 택했다.
               DUCK_SCALE 로 통째로 키우거나 줄인다.
  높이(z)    : 위에서 찍은 사진이라 안 보인다. 몸통 높이 = 폭 × 0.88 로 뒀다.
  질량       : 60 g (속이 빈 고무 장난감 가정)
  마찰       : 정 1.0 / 동 0.8 (고무). 반발 0.
  강체       : 실제 장난감은 눌리면 변형되지만 여기서는 강체다.

좌표계 (물체 로컬, 단위 m)
--------------------------
  원점 = 몸통 타원체 중심, +x = 부리 방향, +y = 왼쪽, +z = 등 쪽(엎드린 자세에서 위)

부위 표현
---------
  부위 하나 = 타원체 1개 이상. 각 타원체가 그대로 충돌 형상(볼록)이 된다.
  head = 머리 + 부리 (사진의 head 윤곽이 부리를 포함한다)
  body = 몸통
  leg1, leg2 = 발 (같은 'leg' 계열. grasp_pca_core.is_part_family 와 호환)
  눈은 시각용만 있고 충돌·부위 판정에서 빠진다.
"""
from dataclasses import dataclass, field

import numpy as np

MM = 1.0e-3
DUCK_SCALE = 1.0  # 전체 크기 배율 (1.0 = 길이 약 145 mm)


@dataclass(frozen=True)
class Ellipsoid:
    name: str
    center_mm: tuple
    semi_mm: tuple          # (a, b, c) 반축 길이
    color: tuple            # 렌더링용 RGB 0~1
    collide: bool = True


@dataclass(frozen=True)
class Part:
    name: str               # 정책·판정에 쓰는 부위 키 (head, body, leg1, leg2)
    semantic: str           # 카메라 의미 분할 라벨 (head, body, leg)
    shapes: tuple = field(default_factory=tuple)


GREEN = (0.20, 0.45, 0.20)
ORANGE = (0.95, 0.45, 0.10)
WHITE = (0.95, 0.95, 0.95)

# 치수는 사진 픽셀 × 0.53 mm/px 에서 나왔다 (모듈 docstring 참고).
PARTS = (
    Part("head", "head", (
        Ellipsoid("head_shell", (55.0, 0.0, 10.0), (23.0, 24.0, 22.0), GREEN),
        Ellipsoid("beak", (82.0, 0.0, 6.0), (11.0, 12.0, 5.5), ORANGE),
    )),
    Part("body", "body", (
        Ellipsoid("body_shell", (0.0, 0.0, 0.0), (38.0, 25.5, 22.5), GREEN),
    )),
    Part("leg1", "leg", (
        Ellipsoid("foot", (-42.0, 11.0, -12.0), (9.0, 7.5, 6.0), ORANGE),
    )),
    Part("leg2", "leg", (
        Ellipsoid("foot", (-42.0, -11.0, -12.0), (9.0, 7.5, 6.0), ORANGE),
    )),
)

# 시각용 장식 (충돌·판정 없음)
EYES = (
    Ellipsoid("eye_l", (66.0, 15.0, 22.0), (4.0, 4.0, 4.0), WHITE, collide=False),
    Ellipsoid("eye_r", (66.0, -15.0, 22.0), (4.0, 4.0, 4.0), WHITE, collide=False),
)

MASS_KG = 0.060
STATIC_FRICTION = 1.0
DYNAMIC_FRICTION = 0.8
RESTITUTION = 0.0

# 고무 근사 — PhysX 부드러운 접촉(compliant contact).
#   강체는 그대로 두고 접촉면만 스프링-댐퍼로 눌리게 한다. 몸통이 휘는 전체 변형은 없다.
#   Franka 손가락(stiffness 2e3, 목표 0)이 폭 51mm 몸통을 쥐면 손가락당 약 51N.
#   51N / 2e4 N/m ≈ 2.5mm 눌림. 고무 장난감을 손가락으로 쥘 때 체감에 맞춘 **가정값**이다.
#   감쇠는 질량 60g 기준 임계감쇠 2*sqrt(k*m) ≈ 69 보다 약간 낮게 뒀다.
COMPLIANT_STIFFNESS = 2.0e4     # N/m
COMPLIANT_DAMPING = 50.0        # N·s/m

# 지시 가능한 부위 (팀 프로젝트 어휘). leg 는 두 발 중 어느 쪽이든 인정한다.
TARGETS = ("head", "body", "leg")

PART_NAMES = tuple(p.name for p in PARTS)
SEMANTIC_OF = {p.name: p.semantic for p in PARTS}


def shapes(scale=DUCK_SCALE):
    """[(부위 이름, 타원체, center_m, semi_m)] — 충돌 형상만."""
    out = []
    for p in PARTS:
        for e in p.shapes:
            out.append((p.name, e,
                        np.asarray(e.center_mm) * MM * scale,
                        np.asarray(e.semi_mm) * MM * scale))
    return out


def ellipsoid_mesh(center, semi, n_lat=14, n_lon=24):
    """UV 타원체 삼각 메시. 반환 (vertices[N,3], faces[M,3] int)."""
    center = np.asarray(center, float)
    semi = np.asarray(semi, float)
    verts = [center + semi * np.array([0.0, 0.0, 1.0])]
    for i in range(1, n_lat):
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * np.pi * j / n_lon
            d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            verts.append(center + semi * d)
    verts.append(center + semi * np.array([0.0, 0.0, -1.0]))
    verts = np.asarray(verts)
    faces = []
    for j in range(n_lon):                      # 위 뚜껑
        faces.append([0, 1 + j, 1 + (j + 1) % n_lon])
    for i in range(n_lat - 2):                  # 띠
        a0, b0 = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for j in range(n_lon):
            j1 = (j + 1) % n_lon
            faces.append([a0 + j, b0 + j, b0 + j1])
            faces.append([a0 + j, b0 + j1, a0 + j1])
    last = len(verts) - 1                       # 아래 뚜껑
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        faces.append([last, base + (j + 1) % n_lon, base + j])
    return verts, np.asarray(faces, dtype=np.int64)


# 충돌 메시 해상도. PhysX 는 볼록 껍질을 꼭짓점 64개 이하로 단순화하는데
# (GPU 경로의 상한), 그 단순화가 껍질을 안쪽으로 깎아 오리가 바닥에 ~2.3mm 파묻혔다.
# 대조군 구는 0.0mm 였다. 그래서 단순화가 일어나지 않도록 직접 64개 이하로 만든다.
#   n_lat=6, n_lon=10 → 꼭짓점 52, 다각형 60 (사각 띠는 평면이라 하나로 합쳐진다)
COLLISION_N_LAT = 6
COLLISION_N_LON = 10


def _inscribed_ratio(n_lat, n_lon):
    """단위 구 UV 메시의 볼록 껍질에서 원점까지 최소 면 거리 (≤1)."""
    v, f = ellipsoid_mesh((0, 0, 0), (1, 1, 1), n_lat, n_lon)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return float(np.abs(np.einsum("ij,ij->i", n, v[f[:, 0]])).min())


def collision_mesh(center, semi, n_lat=COLLISION_N_LAT, n_lon=COLLISION_N_LON):
    """충돌용 거친 타원체. 껍질의 반경 오차가 ±로 고르게 퍼지도록 바깥으로 부풀린다.

    타원체는 단위 구의 선형 변환이고, 선형 변환은 '원점에서 나간 광선 위의 비율'을
    보존한다. 그래서 단위 구에서 구한 내접 비율 r 이 그대로 적용된다.
    꼭짓점을 k = 2/(1+r) 배 하면 반경 비율이 [k·r, k] = [1-e, 1+e] 가 된다.
    반환 (vertices, faces, e)  — e 는 반축 대비 최대 상대 오차.
    """
    r = _inscribed_ratio(n_lat, n_lon)
    k = 2.0 / (1.0 + r)
    center = np.asarray(center, float)
    v, f = ellipsoid_mesh((0, 0, 0), np.asarray(semi, float) * k, n_lat, n_lon)
    return v + center, f, k - 1.0


def normalized_radius(points_m, scale=DUCK_SCALE):
    """각 점의 각 타원체에 대한 정규화 반경 ||(p-c)/semi||. 반환 [N, 형상수].

    1 미만 = 안쪽, 1 = 표면.
    """
    p = np.atleast_2d(np.asarray(points_m, float))
    cols = [np.linalg.norm((p - c) / s, axis=1) for _, _, c, s in shapes(scale)]
    return np.stack(cols, axis=1)


def classify_points(points_m, scale=DUCK_SCALE, max_radius=1.25):
    """로컬 좌표 점 → 부위 이름. 가장 '안쪽'인 타원체의 부위를 고른다.

    max_radius 보다 모든 형상에서 멀면 None (물체 밖).
    파지 판정에서는 두 손가락 끝의 중점을 넣는다 — 중점은 잡힌 부위 안에 있다.
    두 타원체가 겹치는 영역(머리-몸통 목 부분 등)에서는 더 깊이 들어간 쪽을 택한다.
    """
    r = normalized_radius(points_m, scale)
    names = [n for n, _, _, _ in shapes(scale)]
    idx = r.argmin(axis=1)
    best = r[np.arange(len(r)), idx]
    return [names[i] if b <= max_radius else None for i, b in zip(idx, best)]


def sample_surface(n_points=4000, scale=DUCK_SCALE, seed=0):
    """합집합 표면 위의 점과 부위 라벨 (로컬 좌표, m).

    다른 형상 내부에 묻힌 점(목·발 이음부)은 버린다 — 바깥에서 보이지 않는다.
    """
    rng = np.random.default_rng(seed)
    sh = shapes(scale)
    # 대략 표면적 비례로 배분
    def area(a, b, c):  # Knud Thomsen 근사
        q = 1.6
        return 4 * np.pi * (((a * b) ** q + (a * c) ** q + (b * c) ** q) / 3) ** (1 / q)

    areas = np.array([area(*s) for _, _, _, s in sh])
    counts = np.maximum(1, np.round(n_points * 1.5 * areas / areas.sum()).astype(int))
    pts, labels, src = [], [], []
    for i, ((name, _, c, s), k) in enumerate(zip(sh, counts)):
        d = rng.normal(size=(k, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts.append(c + s * d)
        labels += [name] * k
        src += [i] * k
    pts = np.concatenate(pts)
    labels = np.asarray(labels)
    r = normalized_radius(pts, scale)
    own = np.arange(len(sh))[None, :] == np.asarray(src)[:, None]
    buried = ((r < 0.999) & ~own).any(axis=1)
    return pts[~buried], labels[~buried]


def bounding_box_m(scale=DUCK_SCALE):
    lo = np.min([c - s for _, _, c, s in shapes(scale)], axis=0)
    hi = np.max([c + s for _, _, c, s in shapes(scale)], axis=0)
    return lo, hi


def part_widths_mm(scale=DUCK_SCALE):
    """부위별 (x, y, z) 외곽 치수 [mm] — 그리퍼 개방폭(Franka 80 mm)과 비교용."""
    out = {}
    for name, _, c, s in shapes(scale):
        lo, hi = (c - s) / MM, (c + s) / MM
        if name in out:
            lo = np.minimum(lo, out[name][0])
            hi = np.maximum(hi, out[name][1])
        out[name] = (lo, hi)
    return {k: tuple(np.round(hi - lo, 1)) for k, (lo, hi) in out.items()}


def shape_arrays(scale=DUCK_SCALE):
    """GPU 판정용 배열. (centers[S,3] m, semis[S,3] m, family[S] ∈ TARGETS 인덱스)."""
    sh = shapes(scale)
    centers = np.stack([c for _, _, c, _ in sh])
    semis = np.stack([s for _, _, _, s in sh])
    family = np.array([TARGETS.index(SEMANTIC_OF[n]) for n, _, _, _ in sh])
    return centers, semis, family


def anchor_points(scale=DUCK_SCALE):
    """부위별 기준점(로컬, m) — 정책 관측의 '지시 부위 위치'.

    head = 머리 타원체 중심(부리 제외), body = 몸통 중심, leg = 두 발 중심 (둘 중 하나를 고른다).
    """
    sh = {(n, e.name): c for n, e, c, _ in shapes(scale)}
    return {"head": sh[("head", "head_shell")], "body": sh[("body", "body_shell")],
            "leg": np.stack([sh[("leg1", "foot")], sh[("leg2", "foot")]])}


def visible_from_above(points_world, labels, cell_m=0.0025):
    """위에서 내려다볼 때 보이는 점만 남긴다 (xy 격자 z-버퍼).

    격자 칸마다 가장 높은 점 하나를 남긴다. 자기 가림(머리가 몸통을 가리는 등)이 반영된다.
    """
    ij = np.floor(points_world[:, :2] / cell_m).astype(np.int64)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    order = np.lexsort((-points_world[:, 2], key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    keep = order[first]
    return points_world[keep], labels[keep]
