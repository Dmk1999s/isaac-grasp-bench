"""src/toy_duck.py 의 정의로 오리 근사 모형 USD 를 만든다.

  $PY scripts/build_duck_usd.py [--scale 1.0] [--out assets/duck/duck.usda]

구조
----
/Duck                        RigidBodyAPI + MassAPI(질량)   ← Isaac Lab RigidObject 의 루트
  /Duck/PhysicsMaterial      마찰·반발 (모든 충돌 형상에 physics 용도로 바인딩)
  /Duck/head  (라벨 class=head)
      head_shell, beak                       보이는 고해상도 Mesh (충돌 없음)
      head_shell_collision, beak_collision   보이지 않는 거친 Mesh + 볼록 껍질 충돌
      eye_l, eye_r                           시각용만
  /Duck/body  (class=body)   body_shell (+ _collision)
  /Duck/leg1, /Duck/leg2 (class=leg)  foot (+ _collision)

부위별 Xform 에 의미 분할 라벨을 붙여 뎁스 카메라의 semantic_segmentation 이
픽셀마다 head/body/leg 를 돌려주게 한다. 이것이 PCA 쪽 '부위 점군'의 출처가 된다.
각 타원체는 볼록이라 convexHull 근사가 형상을 거의 그대로 보존한다.
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--out", default=str(ROOT / "assets/duck/duck.usda"))
parser.add_argument("--rigid", action="store_true",
                    help="부드러운 접촉을 끈 강체 버전 (비교 기준)")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade, Vt  # noqa: E402

import toy_duck as td  # noqa: E402


def add_mesh(stage, path, ell, scale, collide):
    """collide=False: 보이는 고해상도 메시. True: 보이지 않는 충돌 전용 거친 메시."""
    c = [x * td.MM * scale for x in ell.center_mm]
    s = [x * td.MM * scale for x in ell.semi_mm]
    if collide:
        v, f, _ = td.collision_mesh(c, s)
    else:
        v, f = td.ellipsoid_mesh(c, s)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in v]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in f.reshape(-1)]))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*ell.color)]))
    lo, hi = v.min(axis=0).tolist(), v.max(axis=0).tolist()
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    prim = mesh.GetPrim()
    if collide:
        mesh.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        # hullVertexLimit=255, restOffset=0 을 명시해 봤지만 파묻힘(2.3mm)이 그대로였다.
        # 원인은 고해상도 메시의 단순화였다 → toy_duck.collision_mesh 참고.
    return prim


def main():
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Duck").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetMetadata("comment", "approximate duck toy from one photo; see src/toy_duck.py")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(td.MASS_KG)

    mat = UsdShade.Material.Define(stage, "/Duck/PhysicsMaterial")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(td.STATIC_FRICTION)
    pm.CreateDynamicFrictionAttr(td.DYNAMIC_FRICTION)
    pm.CreateRestitutionAttr(td.RESTITUTION)
    if not args.rigid:
        # 고무 근사: 접촉면만 스프링-댐퍼로 눌린다 (toy_duck.COMPLIANT_* 참고)
        px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        px.CreateCompliantContactStiffnessAttr(td.COMPLIANT_STIFFNESS)
        px.CreateCompliantContactDampingAttr(td.COMPLIANT_DAMPING)

    colliders = []
    for part in td.PARTS:
        xf = UsdGeom.Xform.Define(stage, f"/Duck/{part.name}").GetPrim()
        labels = UsdSemantics.LabelsAPI.Apply(xf, "class")
        labels.CreateLabelsAttr().Set([part.semantic])
        for ell in part.shapes:
            add_mesh(stage, f"/Duck/{part.name}/{ell.name}", ell, args.scale, collide=False)
            colliders.append(add_mesh(stage, f"/Duck/{part.name}/{ell.name}_collision", ell,
                                      args.scale, collide=True))
        if part.name == "head":
            for eye in td.EYES:
                add_mesh(stage, f"/Duck/head/{eye.name}", eye, args.scale, collide=False)

    for prim in colliders:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    stage.GetRootLayer().Save()
    lo, hi = td.bounding_box_m(args.scale)
    msg = (f"[duck] 저장: {out}\n"
           f"[duck] 외곽(m) lo={lo.round(4).tolist()} hi={hi.round(4).tolist()}\n"
           f"[duck] 질량 {td.MASS_KG} kg, 충돌 형상 {len(colliders)}개, scale={args.scale}\n"
           f"[duck] 접촉: {'강체' if args.rigid else f'부드러운 접촉 k={td.COMPLIANT_STIFFNESS} c={td.COMPLIANT_DAMPING}'}\n")
    sys.__stdout__.write(msg)
    (out.parent / f"build_{out.stem}.log").write_text(msg)


main()
app.close()
