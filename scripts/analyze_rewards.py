"""학습 로그에서 보상 항목별 기여를 분리해 기록한다.

Phase 2 요구사항 — 나중에 ablation 에 쓴다.
어느 항목이 학습을 끌고 갔는지, 어느 항목이 페널티로 눌렀는지를 분리해 둬야
보상 설계를 바꿨을 때 무엇이 달라졌는지 귀속할 수 있다.

Isaac Sim 없이 돈다 (텐서보드 이벤트 파일만 읽는다).
"""
import argparse
import csv
import glob
import pathlib

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_scalars(run_dir):
    files = sorted(glob.glob(str(pathlib.Path(run_dir) / "events.out.tfevents*")))
    if not files:
        raise FileNotFoundError(f"텐서보드 이벤트 파일이 없다: {run_dir}")
    ea = event_accumulator.EventAccumulator(
        files[0], size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    out = {}
    for t in tags:
        ev = ea.Scalars(t)
        out[t] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default="eval/reward_terms.csv")
    args = ap.parse_args()

    sc = load_scalars(args.run_dir)
    terms = {t.split("/", 1)[1]: v for t, v in sc.items() if t.startswith("Episode_Reward/")}
    if not terms:
        raise SystemExit("Episode_Reward/* 태그가 없다.")

    steps = terms[next(iter(terms))][0]
    names = sorted(terms)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration"] + names)
        for i, s in enumerate(steps):
            w.writerow([int(s)] + [f"{terms[n][1][i]:.6f}" for n in names])

    print(f"run: {args.run_dir}")
    print(f"기록: {out}  ({len(steps)} iteration x {len(names)} 항목)\n")

    # 학습 구간별 항목 값
    marks = [0, len(steps) // 4, len(steps) // 2, 3 * len(steps) // 4, len(steps) - 1]
    print(f"{'항목':36s}" + "".join(f"{int(steps[m]):>9d}" for m in marks))
    print("-" * (36 + 9 * len(marks)))
    for n in names:
        vals = terms[n][1]
        print(f"{n:36s}" + "".join(f"{vals[m]:9.3f}" for m in marks))

    # 최종 시점 기여 분해
    final = {n: float(terms[n][1][-1]) for n in names}
    pos = {n: v for n, v in final.items() if v > 0}
    neg = {n: v for n, v in final.items() if v <= 0}
    tot_pos = sum(pos.values())
    print(f"\n최종 iteration 기여 (합계 {sum(final.values()):.2f})")
    print("-" * 62)
    for n, v in sorted(pos.items(), key=lambda kv: -kv[1]):
        print(f"  + {n:36s} {v:9.3f}  ({100*v/tot_pos:5.1f}% of 양의 보상)")
    for n, v in sorted(neg.items(), key=lambda kv: kv[1]):
        print(f"  - {n:36s} {v:9.3f}  (페널티)")

    if "Train/mean_reward" in sc:
        mr = sc["Train/mean_reward"][1]
        print(f"\n총 보상: 시작 {mr[0]:.2f} -> 최종 {mr[-1]:.2f}")
        print("※ 항목 합과 총 보상이 다를 수 있다 — 총 보상은 에피소드 길이 정규화 등이"
              " 다르게 들어간다. 항목 간 '상대 비교'에 쓰는 것이 안전하다.")


if __name__ == "__main__":
    main()
