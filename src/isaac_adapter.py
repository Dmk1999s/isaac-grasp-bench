"""Isaac Sim <-> PCA 휴리스틱 규약 어댑터.

두 세계는 단위도 자세 표현도 다르다. 이 모듈이 그 경계를 담당한다.

                    휴리스틱(M0609 유래)      Isaac Sim
    길이 단위       mm                        m
    자세 표현       ZYZ 오일러각(도)          쿼터니언 (w, x, y, z)
    점군 좌표계     로봇 base                 world (env 원점 기준)

휴리스틱 본체는 원본과 동일하게 유지해야 하므로(비교 실험의 타당성),
규약 변환은 전부 이쪽에 모은다.

⚠ 쿼터니언 순서 주의: Isaac Sim 은 (w, x, y, z), scipy 는 (x, y, z, w) 다.
"""
import numpy as np
from scipy.spatial.transform import Rotation

MM_PER_M = 1000.0


# ── 자세 표현 변환 ────────────────────────────────────────────────────
def quat_wxyz_to_zyz_deg(quat_wxyz):
    """Isaac 쿼터니언(w,x,y,z) -> 휴리스틱 ZYZ 오일러각(도)."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    return Rotation.from_quat([x, y, z, w]).as_euler('ZYZ', degrees=True)


def zyz_deg_to_quat_wxyz(rx, ry, rz):
    """휴리스틱 ZYZ 오일러각(도) -> Isaac 쿼터니언(w,x,y,z)."""
    x, y, z, w = Rotation.from_euler('ZYZ', [rx, ry, rz], degrees=True).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def rotmat_to_quat_wxyz(R):
    """3x3 회전행렬 -> Isaac 쿼터니언(w,x,y,z)."""
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


# ── 뎁스 -> 점군 ──────────────────────────────────────────────────────
def depth_to_points_camera_mm(depth_m, intrinsics, stride=1, max_depth_m=3.0):
    """뎁스 이미지(m) -> 카메라 좌표계 점군(mm).

    depth_m     : (H, W) 미터 단위 뎁스. 0 또는 비유한 값은 무효로 본다.
    intrinsics  : 3x3 카메라 내참 행렬 (fx, fy, cx, cy)
    stride      : 다운샘플 간격. 점 수를 줄여 PCA 비용을 낮춘다.

    Isaac Sim 의 카메라 광학축은 -Z, 위쪽이 +Y 다(OpenGL 관례).
    여기서는 OpenCV 관례(광학축 +Z, 아래쪽 +Y)로 맞춰 내보낸다 —
    휴리스틱이 기대하는 것이 그쪽이다.
    """
    depth = np.asarray(depth_m, dtype=np.float64)[::stride, ::stride]
    K = np.asarray(intrinsics, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    h, w = depth.shape
    us = (np.arange(w) * stride).astype(np.float64)
    vs = (np.arange(h) * stride).astype(np.float64)
    uu, vv = np.meshgrid(us, vs)

    valid = np.isfinite(depth) & (depth > 1e-6) & (depth < max_depth_m)
    z = depth[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1) * MM_PER_M


def transform_points(points_mm, T):
    """4x4 동차변환을 점군에 적용. T 의 평행이동 단위는 점군과 같아야 한다."""
    pts = np.asarray(points_mm, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def make_transform(pos, quat_wxyz, pos_scale=1.0):
    """위치+쿼터니언 -> 4x4 동차변환. pos_scale 로 단위를 맞춘다(m->mm 면 1000)."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64) * pos_scale
    return T


# ── 그리퍼 프레임 규약 ────────────────────────────────────────────────
# 휴리스틱은 **닫힘축을 tool_x**(회전행렬 1열)로 정의한다.
#   grasp_pca_core._measure_grasp_width() 가 tool_x 로 폭을 잰다.
# Franka Hand 의 손가락은 **panda_hand 프레임의 y 축**으로 벌어진다.
# 따라서 휴리스틱 자세를 그대로 명령하면 90도 어긋난 축으로 닫는다.
#
# ⚠ 이 오류는 정사각 큐브에서는 보이지 않는다(대칭이라 어느 축으로 닫든 같다).
#   물체를 길쭉하게 만들어야 드러난다. 실측으로 확인한 값:
#     84x42mm 물체, 보정 없음 -> 성공률   0.0%  (84mm 축으로 닫으려 함, 한계 80mm 초과)
#     84x42mm 물체, 90도 보정 -> 성공률 100.0%  (42mm 축으로 닫음)
CLOSING_AXIS_ROT_DEG = 90.0


def heuristic_rot_to_franka_hand(R, extra_rot_deg=0.0):
    """휴리스틱 회전행렬 -> Franka panda_hand 자세.

    tool_z(접근축)는 그대로 두고 tool_z 기준으로 90도 돌려
    휴리스틱의 닫힘축(tool_x)을 Franka 의 손가락 축(y)에 맞춘다.
    """
    R = np.asarray(R, dtype=np.float64)
    ang = CLOSING_AXIS_ROT_DEG + float(extra_rot_deg)
    return R @ Rotation.from_euler("z", ang, degrees=True).as_matrix()


# ── 휴리스틱 출력 -> Isaac 액션 ───────────────────────────────────────
def grasp_pose_to_isaac(pose_mm_deg):
    """휴리스틱 파지 자세 (x,y,z,rx,ry,rz) mm/deg -> (위치 m, 쿼터니언 wxyz).

    상태기계의 IK-Abs 액션이 이 형식을 받는다.

    ⚠ ZYZ 오일러각은 중간각이 0 또는 180도일 때 짐벌락에 걸린다.
      아래를 향하는 파지 자세가 정확히 그 근처(ry~180)라서, 각도 표현 자체는
      유일하지 않다. 회전 자체는 보존되므로 결과는 맞지만, 특이점 근처에서
      수치적으로 민감해질 수 있다.
      회전행렬을 쓸 수 있으면 grasp_result_to_isaac() 쪽이 안전하다.
    """
    x, y, z, rx, ry, rz = np.asarray(pose_mm_deg, dtype=np.float64)
    return np.array([x, y, z]) / MM_PER_M, zyz_deg_to_quat_wxyz(rx, ry, rz)


def grasp_result_to_isaac(result, pose_mm_deg):
    """권장 경로 — 자세를 오일러각이 아니라 회전행렬에서 직접 가져온다.

    result       : compute_grasp 의 첫 반환값 (rotation_matrix 를 가진다)
    pose_mm_deg  : poses["grasp"] 또는 poses["pregrasp"] (위치만 쓴다)

    짐벌락 구간을 우회하므로 특이점 근처에서도 안정적이다.
    """
    pos_mm = np.asarray(pose_mm_deg, dtype=np.float64)[:3]
    R = heuristic_rot_to_franka_hand(result["rotation_matrix"])
    return pos_mm / MM_PER_M, rotmat_to_quat_wxyz(R)
