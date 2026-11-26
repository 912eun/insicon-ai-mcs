"""
recorder — 한 에피소드를 돌려 스냅샷을 JSONL로 저장 (replay 데이터 소스).

라이브 sim 스텝이 데모에서 부담될 때를 대비한 안전망: 미리 한 번 녹화해 두면 대시보드는
가벼운 JSONL 재생만 하면 된다. 서버는 replay/live 둘 다 같은 스냅샷 스키마를 쓴다.

실행 (sim 환경):
    conda activate ai-mcs
    python -m dashboard.recorder --name demo --n-amrs 76 --max-steps 2000
    python -m dashboard.recorder --name demo_batt --battery --n-amrs 76 --max-steps 2000

산출:
    dashboard/runs/<name>.jsonl       스텝별 스냅샷
    dashboard/runs/<name>.meta.json   공장 평면도(grid+ports)  [oht별 1회면 충분하나 run마다 동봉]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.sim_source import export_meta, make_env, snapshots  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="demo", help="저장 이름 (runs/<name>.jsonl)")
    ap.add_argument("--n-amrs", type=int, default=76)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oht", action="store_true", help="OHT 레일 그리드")
    ap.add_argument("--battery", action="store_true", help="배터리 동역학 on")
    ap.add_argument("--battery-scale", type=float, default=1.0,
                    help="방전 가속 배수 (>1이면 충전/방전 장면이 빨리 나타남, 데모용)")
    ap.add_argument("--max-snaps", type=int, default=0,
                    help="녹화 스냅샷 상한 (0=무제한). coupled 충전 env는 sim_days까지 돌아 "
                         "수만 스텝이 되므로 데모 파일은 이걸로 자른다(예: 1500).")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = RUNS_DIR / f"{args.name}.jsonl"
    meta = RUNS_DIR / f"{args.name}.meta.json"

    info = export_meta(meta, oht=args.oht)
    print(f"[meta] {meta.name}  grid={info['grid_h']}x{info['grid_w']}  ports={info['n_ports']}")

    env = make_env(args.n_amrs, args.max_steps, args.seed, args.oht, args.battery,
                   battery_scale=args.battery_scale)
    t0 = time.time()
    n = 0
    last_kpi: dict = {}
    with open(jsonl, "w") as f:
        for snap in snapshots(
            n_amrs=args.n_amrs,
            max_steps=args.max_steps,
            seed=args.seed,
            oht=args.oht,
            battery=args.battery,
            battery_scale=args.battery_scale,
            env=env,
        ):
            f.write(json.dumps(snap) + "\n")
            n += 1
            last_kpi = snap["kpi"]
            if n % 200 == 0:
                print(f"  snap {n:5d}  step={last_kpi.get('step')} "
                      f"throughput={last_kpi.get('throughput')} "
                      f"deadlocks={last_kpi.get('deadlocks')} pending={last_kpi.get('pending')}")
            if args.max_snaps and n >= args.max_snaps:
                print(f"  [cap] {args.max_snaps} 스냅샷 도달 — 중단")
                break

    dt = time.time() - t0
    print(f"[done] {jsonl.name}  {n} steps in {dt:.1f}s "
          f"({n / dt:.0f} steps/s)  final_throughput={last_kpi.get('throughput')}")


if __name__ == "__main__":
    main()
