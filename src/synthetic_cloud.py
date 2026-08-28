"""물체 자세 + 알려진 형상에서 '한 시점에서 보이는' 점군을 생성한다.

왜 카메라 대신 이것인가
-----------------------
1) 실험 설계상의 이유 (이쪽이 더 중요하다)
   비교 대상인 RL 정책은 상태 관측(물체 위치)을 받는다. 휴리스틱에만 카메라
   점군을 주면 두 방식의 차이에 '관측 양식'이라는 변수가 섞여, 차이를
   '제어 방식'으로 귀속할 수 없다 (실험 설계 원칙 1번).
   여기서 만드는 점군은 물체의 실제 자세에서 유도되므로 정보량이 상태 관측과
   동등하다. 그러면서도 '한 시점에서 보이는 면만 보인다'는 부분 관측성은
   유지해 휴리스틱의 실제 난이도(부분 점군 위의 PCA)를 보존한다.

2) 환경상의 이유
   이 인스턴스에서 Isaac Sim 의 RTX 렌더러가 초기화 단계에서 죽는다
   (librtx.scenedb.plugin.so / carbOnPluginStartup). 공식 예제도 동일하게
   죽으므로 설정 문제가 아니다. RayCasterCamera 는 정적 단일 메시만 지원해
   움직이는 큐브에 쓸 수 없다. 자세한 내용은 docs/WORKLOG.md 참고.

카메라 점군으로 갈아끼우려면 sample_visible_box_surface() 를 대체하면 된다.
휴리스틱은 '점군(base 좌표계, mm)'만 받으므로 그 위쪽은 바뀌지 않는다.
"""
import numpy as np
from scipy.spatial.transform import Rotation

MM_PER_M = 1000.0

# 축 정렬 직육면체의 6면: (법선축, 부호)
_FACES = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]


# 면이 '보인다'고 칠 최소 정면성. 법선과 시선의 내적 절대값이 이보다 커야 한다.
# 0.1 ≈ 입사각 84도. 그보다 스치는 면은 실제 뎁스 카메라에서도 픽셀이 거의 안 잡힌다.
#
# ⚠ 이 값을 0 에 가깝게(예: 1e-6) 두면 안 된다.
#   시뮬레이터가 주는 물체 쿼터니언은 정확한 항등원이 아니라 ~5e-6 수준의
#   수치 잔차를 갖는다(예: w=0.99999994, x=-4.5e-06). 임계값이 그 잔차보다
#   작으면 시선과 수직인 옆면들이 부호 노이즈만으로 '보이는 면'이 되고,
#   옆면은 물체 높이 전체에 걸쳐 있어 점군 중심이 통째로 끌려간다.
#   실제로 이 버그 때문에 파지점이 물체 중심에서 20mm 어긋났다.
MIN_FACING = 0.1


def sample_visible_box_surface(size_m, pos_m, quat_wxyz, view_dir=(0.0, 0.0, -1.0),
                               n_points=600, noise_mm=0.0, seed=None,
                               min_facing=MIN_FACING):
    """직육면체 표면 중 view_dir 로 바라볼 때 보이는 면만 샘플링한다.

    size_m     : (sx, sy, sz) 물체 로컬 치수 [m]
    pos_m      : 물체 중심 위치 (로봇 base 기준) [m]
    quat_wxyz  : 물체 자세 쿼터니언 (w, x, y, z)
    view_dir   : 시선 방향 (base 좌표계). 기본값은 위에서 아래로 내려다보기.
    noise_mm   : 뎁스 센서 잡음을 흉내내는 표면 법선 방향 가우시안 잡음
    반환       : (N, 3) 점군, base 좌표계, **mm**

    보이는 면 판정은 면 법선과 시선의 내적으로 한다.
    단순히 '내적 < 0' 이 아니라 min_facing 만큼 확실히 마주봐야 한다 —
    이유는 위 MIN_FACING 주석 참고.
    자기 가림(self-occlusion)만 다루며, 다른 물체에 의한 가림은 다루지 않는다.
    """
    rng = np.random.default_rng(seed)
    half = np.asarray(size_m, dtype=np.float64) / 2.0
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    view = np.asarray(view_dir, dtype=np.float64)
    view = view / np.linalg.norm(view)

    visible = []
    for axis, sign in _FACES:
        normal_local = np.zeros(3)
        normal_local[axis] = sign
        normal_world = R @ normal_local
        if float(normal_world @ view) < -min_facing:   # 충분히 마주보는 면만
            visible.append((axis, sign, normal_world))

    if not visible:                                  # 수치적 예외 대비
        visible = [(2, +1, R @ np.array([0.0, 0.0, 1.0]))]

    # 면적에 비례해 점을 배분한다 (넓은 면에 점이 더 많은 게 자연스럽다)
    areas = []
    for axis, _, _ in visible:
        others = [i for i in range(3) if i != axis]
        areas.append(4.0 * half[others[0]] * half[others[1]])
    areas = np.asarray(areas)
    counts = np.maximum(1, np.round(n_points * areas / areas.sum()).astype(int))

    pts_local = []
    for (axis, sign, _), cnt in zip(visible, counts):
        others = [i for i in range(3) if i != axis]
        p = np.empty((cnt, 3))
        p[:, axis] = sign * half[axis]
        p[:, others[0]] = rng.uniform(-half[others[0]], half[others[0]], cnt)
        p[:, others[1]] = rng.uniform(-half[others[1]], half[others[1]], cnt)
        pts_local.append(p)
    pts_local = np.concatenate(pts_local, axis=0)

    pts_base_m = pts_local @ R.T + np.asarray(pos_m, dtype=np.float64)
    pts_mm = pts_base_m * MM_PER_M
    if noise_mm > 0.0:
        pts_mm = pts_mm + rng.normal(0.0, noise_mm, pts_mm.shape)
    return pts_mm
