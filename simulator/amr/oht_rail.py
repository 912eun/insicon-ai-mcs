"""
OHT 레일망 생성 — SMAT2022 fab 레이아웃에서 OHT(천장 반송) 레일 그리드를 유도한다.

배경:
    AMR(floor)은 grid_map의 모든 통행가능 셀(aisle+port)을 자유롭게 A*로 이동한다.
    OHT(레일)는 고정 레일 위로만 이동 → 우회·단일트랙 정체가 발생한다.
    이 모듈은 같은 fab 지오메트리에서 "레일만 통행가능"한 grid를 만들어
    동일 시뮬레이터(FABSimulator)로 OHT를 floor AMR과 통제비교할 수 있게 한다.

레일망 구조 (spine 레이아웃, fab_layout.json bays에서 유도):
    트렁크(interbay) : 수평 코리도 y∈{6,42,78} + 수직 연결 x∈{3,154} → 외곽/중앙 루프
    스퍼(intrabay)   : 각 포트 → 가장 가까운 트렁크 셀까지 floor 최단경로를 카빙
                       (베이 내 자연 aisle 사용, 베이간 이동은 트렁크 강제)

출력 grid_map 규약 (astar와 동일):
    1 = 비통행(레일 아님),  0 = 레일(통행가능). 포트 셀은 스퍼 끝점이라 항상 레일.

실행:
    python simulator/amr/oht_rail.py            # 생성 + 저장 + 통계
    python simulator/amr/oht_rail.py --viz      # + PNG 시각화
"""

import json
import argparse
from collections import deque
from pathlib import Path

import numpy as np

ROOT        = Path(__file__).parent.parent.parent
FLOOR_GRID  = ROOT / "rware_simulator" / "output" / "grid_map.npy"
PORTS_JSON  = ROOT / "rware_simulator" / "output" / "port_locations.json"
OUT_RAIL    = ROOT / "rware_simulator" / "output" / "rail_grid.npy"

# 트렁크 코리도 (floor에서 100% 뚫린 행/열로 선정 — oht_rail 분석 참고)
TRUNK_ROWS = (6, 42, 78)     # 상단/중앙/하단 interbay 수평 코리도
TRUNK_COLS = (3, 154)        # 좌/우 수직 연결 (perimeter loop)

# 4-이웃 (레일은 직교 — 대각선 레일 없음)
NB4 = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _walkable(floor: np.ndarray) -> np.ndarray:
    """floor grid에서 통행가능(장애물 아님) 마스크. 1=장애물."""
    return floor != 1


def _carve_trunk(floor: np.ndarray, rail: np.ndarray, width: int = 1):
    """수평/수직 트렁크 코리도를 레일로 카빙 (floor 통행가능 셀만).

    width>1 이면 코리도를 그만큼 두껍게 카빙 → 추월/양방향 가능(단일트랙 정체 완화).
    실제 OHT 메인라인의 양방향·패싱 특성 반영.
    """
    walk = _walkable(floor)
    H, W = floor.shape
    x_lo, x_hi = min(TRUNK_COLS), max(TRUNK_COLS)
    y_lo, y_hi = min(TRUNK_ROWS), max(TRUNK_ROWS)
    for y0 in TRUNK_ROWS:
        for dy in range(width):
            y = y0 + dy
            if not (0 <= y < H):
                continue
            for x in range(x_lo, x_hi + 1):
                if walk[y, x]:
                    rail[y, x] = True
    for x0 in TRUNK_COLS:
        for dx in range(width):
            x = x0 + dx
            if not (0 <= x < W):
                continue
            for y in range(y_lo, y_hi + 1):
                if walk[y, x]:
                    rail[y, x] = True


def _bfs_path(floor: np.ndarray, start, targets: set):
    """floor 통행가능 셀 위에서 start→targets 중 하나까지 4-이웃 BFS 최단경로.
    경로 셀 리스트 반환 (start..target 포함), 도달 불가면 None."""
    walk = _walkable(floor)
    H, W = floor.shape
    if start in targets:
        return [start]
    seen = {start}
    prev = {}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        for dx, dy in NB4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H) or not walk[ny, nx]:
                continue
            if (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            prev[(nx, ny)] = (cx, cy)
            if (nx, ny) in targets:
                # 경로 복원
                path = [(nx, ny)]
                node = (cx, cy)
                while node != start:
                    path.append(node)
                    node = prev[node]
                path.append(start)
                path.reverse()
                return path
            q.append((nx, ny))
    return None


def build_rail_grid(floor: np.ndarray, ports: dict, verbose: bool = True,
                    trunk_width: int = 1):
    """floor grid + 포트에서 OHT 레일 grid를 생성한다.

    Returns: (rail_grid int8 [1=block,0=rail], stats dict)
    """
    H, W = floor.shape
    rail = np.zeros((H, W), dtype=bool)

    # 1) 트렁크
    _carve_trunk(floor, rail, width=trunk_width)
    trunk_cells = {(x, y) for y in range(H) for x in range(W) if rail[y, x]}

    # 2) 포트별 스퍼: 포트 → 가장 가까운 트렁크 셀
    unreachable = []
    for name, info in ports.items():
        p = (info["grid_x"], info["grid_y"])
        if rail[p[1], p[0]]:
            continue
        path = _bfs_path(floor, p, trunk_cells)
        if path is None:
            unreachable.append(name)
            # 도달 불가 포트도 최소한 자기 셀은 레일로 (고립 노드)
            rail[p[1], p[0]] = True
            continue
        for (x, y) in path:
            rail[y, x] = True

    # 3) 연결성 검사 (최대 연결요소 비율)
    rail_grid = np.where(rail, 0, 1).astype(np.int8)
    comp_size, n_comp = _largest_component(rail)

    stats = {
        "rail_cells":    int(rail.sum()),
        "floor_walkable": int(_walkable(floor).sum()),
        "rail_frac_of_floor": round(rail.sum() / _walkable(floor).sum() * 100, 1),
        "n_ports":       len(ports),
        "unreachable_ports": unreachable,
        "largest_component": comp_size,
        "n_components":   n_comp,
        "connected_frac": round(comp_size / max(int(rail.sum()), 1) * 100, 1),
    }
    if verbose:
        print("=== OHT rail_grid 생성 ===")
        print(f"레일 셀: {stats['rail_cells']} "
              f"(floor 통행가능 {stats['floor_walkable']}의 {stats['rail_frac_of_floor']}%)")
        print(f"포트: {stats['n_ports']} | 도달불가: {len(unreachable)}")
        print(f"연결요소: {n_comp}개 | 최대요소: {comp_size} "
              f"({stats['connected_frac']}% of rail)")
        if unreachable:
            print(f"  ⚠ 도달불가 포트 예시: {unreachable[:5]}")
    return rail_grid, stats


def _largest_component(rail: np.ndarray):
    """레일 마스크의 4-연결 요소 중 최대 크기와 요소 수."""
    H, W = rail.shape
    seen = np.zeros_like(rail, dtype=bool)
    largest, n_comp = 0, 0
    for y in range(H):
        for x in range(W):
            if rail[y, x] and not seen[y, x]:
                n_comp += 1
                size = 0
                q = deque([(x, y)])
                seen[y, x] = True
                while q:
                    cx, cy = q.popleft()
                    size += 1
                    for dx, dy in NB4:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < W and 0 <= ny < H and rail[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((nx, ny))
                largest = max(largest, size)
    return largest, n_comp


def measure_detour(floor: np.ndarray, rail_grid: np.ndarray, ports: dict,
                   n_pairs: int = 300, seed: int = 0):
    """무작위 포트쌍에 대해 rail 최단경로 / floor 최단경로 길이비(우회율) 측정."""
    from simulator.amr.pathfinding import astar
    rng = np.random.default_rng(seed)
    names = list(ports.keys())
    ratios, rail_fail = [], 0
    for _ in range(n_pairs):
        a, b = rng.choice(names, 2, replace=False)
        pa = (ports[a]["grid_x"], ports[a]["grid_y"])
        pb = (ports[b]["grid_x"], ports[b]["grid_y"])
        fp = astar(floor, pa, pb)
        rp = astar(rail_grid, pa, pb)
        if fp is None:
            continue
        if rp is None:
            rail_fail += 1
            continue
        ratios.append((len(rp) - 1) / max(len(fp) - 1, 1))
    ratios = np.array(ratios)
    return {
        "n_ok": len(ratios),
        "rail_unreachable_pairs": rail_fail,
        "detour_mean": round(float(ratios.mean()), 3) if len(ratios) else None,
        "detour_p50":  round(float(np.percentile(ratios, 50)), 3) if len(ratios) else None,
        "detour_p95":  round(float(np.percentile(ratios, 95)), 3) if len(ratios) else None,
    }


def visualize(floor: np.ndarray, rail_grid: np.ndarray, ports: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = floor.shape
    img = np.zeros((H, W, 3), dtype=float)
    img[floor == 1] = (0.85, 0.85, 0.85)   # 장비=회색
    img[floor != 1] = (1, 1, 1)            # aisle=흰색
    img[rail_grid == 0] = (0.1, 0.3, 0.9)  # 레일=파랑
    px = [p["grid_x"] for p in ports.values()]
    py = [p["grid_y"] for p in ports.values()]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img, origin="upper")
    ax.scatter(px, py, s=3, c="red", label="ports")
    ax.set_title("OHT rail network (blue) over fab layout (gray=equipment)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"시각화 저장: {out_png}")


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--out", default=str(OUT_RAIL))
    ap.add_argument("--trunk-width", type=int, default=1,
                    help="트렁크 코리도 두께 (2=양방향/패싱 → 단일트랙 정체 완화)")
    args = ap.parse_args()

    floor = np.load(FLOOR_GRID)
    with open(PORTS_JSON) as f:
        ports = json.load(f)

    rail_grid, stats = build_rail_grid(floor, ports, trunk_width=args.trunk_width)
    np.save(args.out, rail_grid)
    print(f"저장: {args.out}")

    det = measure_detour(floor, rail_grid, ports)
    print(f"\n=== 우회율 (rail경로/floor경로, {det['n_ok']}쌍) ===")
    print(f"평균 {det['detour_mean']}× | p50 {det['detour_p50']}× | p95 {det['detour_p95']}×"
          f" | rail도달불가쌍 {det['rail_unreachable_pairs']}")

    if args.viz:
        visualize(floor, rail_grid, ports, ROOT / "rware_simulator" / "output" / "rail_grid.png")


if __name__ == "__main__":
    main()
