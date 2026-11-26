import heapq
import math
from typing import Optional

import numpy as np


def compute_step_costs(
    path: list,
    cell_mm: float = 2000.0,
    v_max: float = 1500.0,
    acc: float = 2000.0,
    dec: float = 3000.0,
) -> list:
    """
    Returns per-segment step costs for a path (length = len(path) - 1).
    step_cost > 1.0 means the segment takes more than one step at max speed.

    Uses the SMAT2022 vehicle parameters (defaults):
        cell_mm = 2000 mm/cell
        v_max   = 1500 mm/s  →  1 step = cell_mm/v_max ≈ 1.33 s
        acc     = 2000 mm/s²  (acceleration)
        dec     = 3000 mm/s²  (deceleration, faster than acc — typical for OHT/AMR)

    Algorithm: forward/backward velocity scan to compute reachable speed at each
    waypoint, then derive time per segment using constant-acceleration kinematics.
    """
    n = len(path)
    if n <= 1:
        return []

    t_ref = cell_mm / v_max  # reference: 1 step = time for 1 cardinal cell at v_max

    # Euclidean length of each segment in mm (handles diagonals correctly)
    seg_len = [
        math.sqrt(((path[i+1][0] - path[i][0]) * cell_mm) ** 2
                  + ((path[i+1][1] - path[i][1]) * cell_mm) ** 2)
        for i in range(n - 1)
    ]

    # Forward pass: max speed at START of waypoint i reachable from rest at waypoint 0
    v_fwd = [0.0] * n
    for i in range(n - 1):
        v_fwd[i + 1] = min(v_max, math.sqrt(v_fwd[i] ** 2 + 2 * acc * seg_len[i]))

    # Backward pass: max speed at START of waypoint i that still allows stopping at end
    v_bwd = [0.0] * n
    for i in range(n - 2, -1, -1):
        v_bwd[i] = min(v_max, math.sqrt(v_bwd[i + 1] ** 2 + 2 * dec * seg_len[i]))

    # Actual speed at each waypoint
    v = [min(v_fwd[i], v_bwd[i]) for i in range(n)]

    costs = []
    for i in range(n - 1):
        L = seg_len[i]
        v0, v1 = v[i], v[i + 1]

        # Accel/decel distances within this segment
        d_acc = (v_max ** 2 - v0 ** 2) / (2 * acc) if v0 < v_max else 0.0
        d_dec = (v_max ** 2 - v1 ** 2) / (2 * dec) if v1 < v_max else 0.0
        d_const = L - d_acc - d_dec

        if d_const >= 0:
            # Trapezoidal profile: accel → const → decel
            t = (v_max - v0) / acc + d_const / v_max + (v_max - v1) / dec
        else:
            # Triangular profile (can't reach v_max within this segment)
            # v_peak² = 2*acc*dec/(acc+dec) * L + (dec*v0² + acc*v1²)/(acc+dec)
            v_peak_sq = (2 * acc * dec / (acc + dec)) * L + (dec * v0 ** 2 + acc * v1 ** 2) / (acc + dec)
            v_peak = min(v_max, math.sqrt(max(v_peak_sq, 0.0)))
            t = (v_peak - v0) / acc + (v_peak - v1) / dec

        costs.append(max(t / t_ref, 1e-6))

    return costs


def astar(grid_map: np.ndarray, start: tuple, goal: tuple) -> Optional[list]:
    """
    Returns shortest path as list of (x, y) tuples, or None if unreachable.
    grid_map[y][x] == 1 means obstacle.
    대각선 이동 시 코너 컷팅 방지: 인접 두 셀이 모두 통과 가능해야 함.
    """
    grid_h, grid_w = grid_map.shape

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_valid(x, y):
        return 0 <= x < grid_w and 0 <= y < grid_h and grid_map[y][x] != 1

    if start == goal:
        return [start]
    if not is_valid(*start) or not is_valid(*goal):
        return None

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    dir_costs  = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]

    open_set = []
    heapq.heappush(open_set, (0, 0, start))
    came_from = {}
    g_cost = {start: 0}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current == goal:
            path, node = [], goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path
        if current_g > g_cost.get(current, float("inf")):
            continue
        for (dx, dy), cost in zip(directions, dir_costs):
            nb = (current[0] + dx, current[1] + dy)
            if not is_valid(*nb):
                continue
            # 대각선 코너 컷팅 방지
            if dx != 0 and dy != 0:
                if not is_valid(current[0] + dx, current[1]) or \
                   not is_valid(current[0], current[1] + dy):
                    continue
            new_g = g_cost[current] + cost
            if new_g < g_cost.get(nb, float("inf")):
                g_cost[nb] = new_g
                came_from[nb] = current
                heapq.heappush(open_set, (new_g + heuristic(nb, goal), new_g, nb))
    return None


def astar_with_reservation(
    grid_map: np.ndarray,
    start: tuple,
    goal: tuple,
    reservation,
    max_time: int = 500,
    time_offset: int = 0,
    density: Optional[np.ndarray] = None,
    w_cong: float = 0.0,
    lane_w: float = 0.0,
) -> Optional[list]:
    """
    Time-extended A* that avoids vertex/edge conflicts in the reservation table.
    Returns list of (x, y) tuples, or None if no path found within max_time.

    time_offset: 현재 시뮬레이션 절대 스텝. 예약 테이블 조회 시
                 상대 시간(ct)에 더해 절대 시간으로 변환한다.
    대각선 이동 시 코너 컷팅 방지 적용.
    """
    grid_h, grid_w = grid_map.shape

    def is_valid(x, y):
        return 0 <= x < grid_w and 0 <= y < grid_h and grid_map[y][x] != 1

    def heuristic(x, y):
        return abs(x - goal[0]) + abs(y - goal[1])

    # (dx, dy, cost): stay-in-place costs 0.5 to encourage movement
    directions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    dir_costs  = [0.5, 1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]

    start_state = (start[0], start[1], 0)
    open_set = []
    heapq.heappush(open_set, (0, 0, start_state))
    came_from = {}
    g_cost = {start_state: 0}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        cx, cy, ct = current

        if (cx, cy) == goal:
            path, node = [], current
            while node in came_from:
                path.append((node[0], node[1]))
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path

        if current_g > g_cost.get(current, float("inf")):
            continue
        if ct >= max_time:
            continue

        for (dx, dy), cost in zip(directions, dir_costs):
            nx, ny, nt = cx + dx, cy + dy, ct + 1
            abs_t = nt + time_offset
            if not is_valid(nx, ny):
                continue
            # 대각선 코너 컷팅 방지
            if dx != 0 and dy != 0:
                if not is_valid(cx + dx, cy) or not is_valid(cx, cy + dy):
                    continue
            if reservation.is_vertex_conflict(nx, ny, abs_t):
                continue
            if reservation.is_edge_conflict(cx, cy, nx, ny, abs_t):
                continue
            new_g = g_cost[current] + cost
            # 혼잡가중: 주변 AMR 밀도 높은 셀에 추가비용 → 막힌 구역째 우회 (Jang 동적가중 류)
            if density is not None and w_cong > 0.0:
                new_g += w_cong * float(density[ny, nx])
            # 차선제(일방통행): 셀 좌표 기반 선호방향과 반대로 가면 페널티 → 레인 형성.
            #   가로이동: 행 짝수=동(+x) 선호 / 홀수=서. 세로이동: 열 짝수=남(+y) / 홀수=북.
            if lane_w > 0.0:
                if dx != 0 and ((cy % 2 == 0) != (dx > 0)):
                    new_g += lane_w
                if dy != 0 and ((cx % 2 == 0) != (dy > 0)):
                    new_g += lane_w
            nb_state = (nx, ny, nt)
            if new_g < g_cost.get(nb_state, float("inf")):
                g_cost[nb_state] = new_g
                came_from[nb_state] = current
                f = new_g + heuristic(nx, ny)
                heapq.heappush(open_set, (f, new_g, nb_state))
    return None
