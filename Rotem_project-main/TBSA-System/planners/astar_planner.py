"""
astar_planner.py

A* 기반 2D(XZ) 경로 탐색 + 시각화 클래스
--------------------------------------
- tracking mode 에서 사용하는 Flask 서버 코드에서 import 해서 사용하기 위한 용도
- /set_destination API 로 받은 위치까지 최단 경로 계산
- /update_obstacle API 로 받은 장애물(x_min, x_max, z_min, z_max) 정보 사용
- 전차의 크기를 고려한 margin(기본 2.0) 적용
- 필요 시 matplotlib 로 장애물 + 경로 시각화

[수정사항 - 장애물 회피 강화 버전]
- 장애물과의 거리에 따른 proximity cost 추가
- 장애물 근처를 지나가면 비용이 증가하여 자연스럽게 넓은 공간으로 우회
- safety_weight 파라미터로 안전성 vs 최단경로 트레이드오프 조절 가능

예상 사용 시나리오(Flask 서버 쪽):
    from astar_planner import AStarPlanner, ObstacleRect

    # ▶ Terrain 이 300 x 300 이라고 가정한 기본값 예시
    planner = AStarPlanner(
        grid_min_x=0.0,
        grid_max_x=300.0,
        grid_min_z=0.0,
        grid_max_z=300.0,
        cell_size=1.0,
        obstacle_margin=2.0,
        allow_diagonal=True,
        safety_weight=1.5,  # 안전성 가중치 (높을수록 장애물 회피)
    )

    # 1) /update_obstacle 에서 호출
    def update_obstacles_from_payload(payload: dict):
        obs_list = []
        for item in payload.get("obstacles", []):
            obs = ObstacleRect.from_min_max(
                x_min=item["x_min"],
                x_max=item["x_max"],
                z_min=item["z_min"],
                z_max=item["z_max"],
            )
            obs_list.append(obs)
        planner.set_obstacles(obs_list)

    # 2) /set_destination 에서 목적지 저장만 해두고,
    # 3) /get_action 에서 현재 탱크 위치 current_pos, 저장된 dest 를 이용해서
    #    path = planner.find_path(current_pos, dest)
    #    를 호출하여 경로를 얻어 사용.

주의:
- 이 코드는 "평면 상의 최단경로"만 담당한다.
- 실제 전차 이동/회전/가속도/제동 등은 기존 로직에서 이 경로를 따라가도록 구현하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple

import math
from config import Config

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover - matplotlib 미설치 환경 대비
    _HAS_MPL = False

COST_STRAIGHT = Config.ASTAR.COST_STRAIGHT
COST_DIAGONAL = Config.ASTAR.COST_DIAGONAL
SNAP_RADIUS = Config.ASTAR.SNAP_RADIUS

@dataclass
class ObstacleRect:
    """
    XZ 평면에서의 축 정렬 사각형 장애물
    (Unity 에서 전달받는 /update_obstacle payload 형식과 쉽게 매핑하기 위한 구조)
    """
    center_x: float
    center_z: float
    size_x: float
    size_z: float

    @property
    def x_min(self) -> float:
        return self.center_x - self.size_x * 0.5

    @property
    def x_max(self) -> float:
        return self.center_x + self.size_x * 0.5

    @property
    def z_min(self) -> float:
        return self.center_z - self.size_z * 0.5

    @property
    def z_max(self) -> float:
        return self.center_z + self.size_z * 0.5

    @classmethod
    def from_min_max(cls, x_min: float, x_max: float, z_min: float, z_max: float) -> "ObstacleRect":
        """/update_obstacle 의 x_min, x_max, z_min, z_max 형식에서 바로 만들기 편하도록 제공"""
        cx = (x_min + x_max) * 0.5
        cz = (z_min + z_max) * 0.5
        sx = (x_max - x_min)
        sz = (z_max - z_min)
        return cls(center_x=cx, center_z=cz, size_x=sx, size_z=sz)


class _Node:
    """내부용 A* 노드 구조체 (grid index + 비용 정보"""

    __slots__ = (
        "ix", "iz", "walkable",
        "g_cost", "h_cost", "parent",
        "proximity_cost"  # 장애물 근접 비용 추가
    )

    def __init__(self, ix: int, iz: int, walkable: bool, proximity_cost: float = 0.0):
        self.ix = ix
        self.iz = iz
        self.walkable = walkable
        self.g_cost: int = 0
        self.h_cost: int = 0
        self.parent: Optional["_Node"] = None
        self.proximity_cost: float = proximity_cost  # 장애물 근접 비용

    @property
    def f_cost(self) -> int:
        return self.g_cost + self.h_cost


class AStarPlanner:
    """
    A* 경로 탐색 + 시각화 클래스 (장애물 회피 강화 버전)

    - grid_min_x ~ grid_max_x, grid_min_z ~ grid_max_z 범위 안을 cell_size 로 자른 2D 그리드를 구성
    - 장애물 + obstacle_margin 을 고려해서 walkable / blocked 셀 판정
    - 장애물 근접 비용(proximity cost)을 추가하여 장애물에서 멀리 떨어진 경로 선호
    - find_path() 로 시작점(start) ~ 목적지(goal) 사이의 안전한 경로 계산
    - plot() 으로 장애물 + 경로를 matplotlib 으로 시각화 가능

    좌표계:
        - Unity 상의 X / Z 를 그대로 사용한다고 가정
        - (x, z) 튜플을 월드 좌표처럼 사용
    """

    def __init__(
        self,
        grid_min_x: float,
        grid_max_x: float,
        grid_min_z: float,
        grid_max_z: float,
        cell_size: float = 1.0,
        obstacle_margin: float = 2.0,
        allow_diagonal: bool = True,
        safety_weight: float = 1.5,        # 안전성 가중치 (높을수록 장애물 회피)
        proximity_radius: float = 8.0,     # 장애물 영향 반경 (이 거리 내에서 비용 증가)
    ) -> None:
        assert cell_size > 0.0, "cell_size must be > 0"

        self.grid_min_x = float(grid_min_x)
        self.grid_max_x = float(grid_max_x)
        self.grid_min_z = float(grid_min_z)
        self.grid_max_z = float(grid_max_z)
        self.cell_size = float(cell_size)
        self.obstacle_margin = float(obstacle_margin)
        self.allow_diagonal = bool(allow_diagonal)
        
        # 장애물 회피 강화 파라미터
        self.safety_weight = float(safety_weight)
        self.proximity_radius = float(proximity_radius)

        # 그리드 해상도(셀 개수)
        self.grid_size_x = max(1, int(math.ceil((self.grid_max_x - self.grid_min_x) / self.cell_size)))
        self.grid_size_z = max(1, int(math.ceil((self.grid_max_z - self.grid_min_z) / self.cell_size)))

        # 장애물 리스트
        self._obstacles: List[ObstacleRect] = []
        # 마스킹 영역 (No-Go Zone)
        self._mask_zones: List[ObstacleRect] = []
        
        # 노드 그리드 (lazy build)
        self._grid: List[List[_Node]] = []
        self._grid_valid: bool = False
        
        # 장애물 근접 비용 맵 (캐싱용)
        self._proximity_map: List[List[float]] = []

    # Planner의 탐색 범위를 동적으로 변경
    def update_grid_range(self, min_x, max_x, min_z, max_z):
        """플래너의 탐색 범위를 동적으로 변경"""
        self.grid_min_x = float(min_x)
        self.grid_max_x = float(max_x)
        self.grid_min_z = float(min_z)
        self.grid_max_z = float(max_z)
        
        # 해상도 재계산
        self.grid_size_x = max(1, int(math.ceil((self.grid_max_x - self.grid_min_x) / self.cell_size)))
        self.grid_size_z = max(1, int(math.ceil((self.grid_max_z - self.grid_min_z) / self.cell_size)))
        
        # 기존 그리드 무효화 (다음 find_path 호출 시 새로 빌드됨)
        self._grid_valid = False
        print(f"📏 A* 범위 변경 완료: X({min_x}~{max_x}), Z({min_z}~{max_z})")

    # ------------------------------------------------------------------
    # 안전성 파라미터 조정
    # ------------------------------------------------------------------
    def set_safety_weight(self, weight: float):
        """
        안전성 가중치 설정
        - 0: 순수 최단 경로 (기존 A*와 동일)
        - 1~2: 적당한 안전성 (권장)
        - 3+: 매우 안전한 경로 (우회가 많아질 수 있음)
        """
        self.safety_weight = float(weight)
        self._grid_valid = False
        print(f"🛡️ A* 안전성 가중치 변경: {weight}")
    
    def set_proximity_radius(self, radius: float):
        """
        장애물 영향 반경 설정
        - 이 거리 내의 셀들은 장애물과의 거리에 따라 추가 비용 부과
        """
        self.proximity_radius = float(radius)
        self._grid_valid = False
        print(f"📡 A* 장애물 영향 반경 변경: {radius}m")

    # ------------------------------------------------------------------
    # 장애물 & 그리드
    # ------------------------------------------------------------------
    def set_mask_zones(self, zones: List[ObstacleRect]):
        self._mask_zones = zones
        self._grid_valid = False
        print(f"🚫 마스킹 영역(No-Go Zone) {len(zones)}개 설정 완료")

    def set_obstacles(self, obstacles: Iterable[ObstacleRect]) -> None:
        """장애물 리스트를 설정하고, 그리드를 다시 빌드하도록 플래그 표시"""
        self._obstacles = list(obstacles)
        self._grid_valid = False

    def update_obstacles_from_payload(self, payload):
        """/update_obstacle API의 데이터를 A* 장애물로 변환"""
        obs_list = []
        for item in payload.get("obstacles", []):
            obs = ObstacleRect.from_min_max(
                x_min=item["x_min"], x_max=item["x_max"],
                z_min=item["z_min"], z_max=item["z_max"]
            )
            obs_list.append(obs)
        self.set_obstacles(obs_list)
        print(f"🧱 A* 장애물 업데이트 완료: {len(obs_list)}개")

    def _compute_proximity_cost(self, x: float, z: float) -> float:
        """
        해당 좌표의 장애물 근접 비용 계산
        - 장애물과 가까울수록 비용 증가
        - proximity_radius 밖이면 비용 0
        """
        if not self._obstacles or self.safety_weight <= 0:
            return 0.0
        
        min_dist = float('inf')
        
        for obs in self._obstacles:
            # 장애물 경계까지의 최단 거리 계산
            dx = max(obs.x_min - x, 0, x - obs.x_max)
            dz = max(obs.z_min - z, 0, z - obs.z_max)
            dist = math.sqrt(dx * dx + dz * dz)
            min_dist = min(min_dist, dist)
        
        # 마스킹 영역도 고려
        for zone in self._mask_zones:
            dx = max(zone.x_min - x, 0, x - zone.x_max)
            dz = max(zone.z_min - z, 0, z - zone.z_max)
            dist = math.sqrt(dx * dx + dz * dz)
            min_dist = min(min_dist, dist)
        
        # proximity_radius 내에서 거리에 반비례하는 비용
        if min_dist >= self.proximity_radius:
            return 0.0
        
        # 비용 계산: 가까울수록 높은 비용 (지수 함수 사용)
        # normalized_dist: 0(장애물 바로 옆) ~ 1(proximity_radius 경계)
        normalized_dist = min_dist / self.proximity_radius
        
        # 지수적으로 감소하는 비용 (장애물에 가까울수록 급격히 증가)
        # cost = safety_weight * (1 - normalized_dist)^2 * COST_STRAIGHT
        cost = self.safety_weight * ((1 - normalized_dist) ** 2) * COST_STRAIGHT
        
        return cost

    def _build_grid(self) -> None:
        """장애물 + margin + proximity cost를 고려하여 그리드 초기화"""
        self._grid = []
        self._proximity_map = []
        
        for ix in range(self.grid_size_x):
            col: List[_Node] = []
            prox_col: List[float] = []
            
            for iz in range(self.grid_size_z):
                x, z = self.grid_index_to_world(ix, iz)
                walkable = not self._is_blocked(x, z)
                
                # 장애물 근접 비용 계산
                proximity_cost = self._compute_proximity_cost(x, z) if walkable else 0.0
                
                col.append(_Node(ix, iz, walkable, proximity_cost))
                prox_col.append(proximity_cost)
            
            self._grid.append(col)
            self._proximity_map.append(prox_col)
        
        self._grid_valid = True

    def _is_blocked(self, x: float, z: float) -> bool:
        """장애물 또는 마스킹 영역 체크"""
        # 1. 실시간 장애물 체크
        for obs in self._obstacles:
            if self._check_collision(x, z, obs, self.obstacle_margin):
                return True
        
        # 2. 마스킹 영역 체크 (마진을 0으로 하거나 별도 설정 가능)
        for zone in self._mask_zones:
            if self._check_collision(x, z, zone, margin=0.0): # 마스킹은 정확한 범위로
                return True
                
        return False
    
    def _check_collision(self, x, z, rect, margin):
        """충돌 판정 헬퍼 함수"""
        return (rect.x_min - margin <= x <= rect.x_max + margin and
                rect.z_min - margin <= z <= rect.z_max + margin)

    # ------------------------------------------------------------------
    # 좌표 변환
    # ------------------------------------------------------------------
    def world_to_grid(self, x: float, z: float) -> Optional[Tuple[int, int]]:
        """
        월드 좌표 (x, z)를 그리드 index (ix, iz) 로 변환.
        그리드 범위 밖이면 None 반환.

        * 입력 좌표는 소수점 셋째자리까지 반올림하여 사용.
        """
        # 소수점 둘째자리까지로 제한
        x = round(float(x), 3)
        z = round(float(z), 3)

        # 그리드 범위 확인
        if not (self.grid_min_x <= x <= self.grid_max_x and self.grid_min_z <= z <= self.grid_max_z):
            return None

        fx = (x - self.grid_min_x) / self.cell_size
        fz = (z - self.grid_min_z) / self.cell_size
        ix = int(math.floor(fx))
        iz = int(math.floor(fz))

        if ix < 0 or ix >= self.grid_size_x or iz < 0 or iz >= self.grid_size_z:
            return None
        return ix, iz

    def grid_index_to_world(self, ix: int, iz: int) -> Tuple[float, float]:
        """
        그리드 index (ix, iz)를 셀 중앙의 월드 좌표 (x, z) 로 변환.

        * 반환 좌표는 소수점 둘째자리까지 반올림하여 반환.
        """
        x = self.grid_min_x + (ix + 0.5) * self.cell_size
        z = self.grid_min_z + (iz + 0.5) * self.cell_size
        return round(x, 2), round(z, 2)

    # ------------------------------------------------------------------
    # A* 핵심 로직
    # ------------------------------------------------------------------
    def find_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        use_obstacles: bool = True
    ) -> List[Tuple[float, float]]:
        """
        A* 알고리즘으로 start (x, z) -> goal (x, z) 안전한 경로를 계산해서
        월드 좌표 리스트 [(x1, z1), (x2, z2), ...] 형태로 반환.

        Args:
            start: 시작점 (x, z)
            goal: 목적지 (x, z)
            use_obstacles: True일 때 장애물 회피, False일 때 장애물 무시 (순수 경로)

        - 장애물 근접 비용을 고려하여 장애물에서 멀리 떨어진 안전한 경로 선호
        - 장애물 또는 그리드 밖에 있는 start/goal 은 가장 가까운 walkable 노드로 자동 스냅
        """
        # 장애물 무시 모드면 임시로 장애물 백업 후 제거
        obstacles_backup = None
        if not use_obstacles and self._obstacles:
            obstacles_backup = self._obstacles.copy()
            self._obstacles = []
            self._grid_valid = False  # 그리드 재생성 필요
        
        try:
            if not self._grid_valid:
                self._build_grid()

            start_idx = self.world_to_grid(*start)
            goal_idx = self.world_to_grid(*goal)

            # 그리드 범위 밖 체크
            if start_idx is None:
                print(f"⚠️ 시작점이 탐색 범위 밖입니다! (위치: {start}, 범위: X({self.grid_min_x}~{self.grid_max_x}))")
                return []
            if goal_idx is None:
                print(f"⚠️ 목적지가 탐색 범위 밖입니다! (위치: {goal})")
                return []

            if start_idx is None or goal_idx is None:
                # 그리드 범위를 벗어난 경우
                return []

            sx, sz = start_idx
            gx, gz = goal_idx

            start_node = self._grid[sx][sz]
            goal_node = self._grid[gx][gz]

            # 장애물 충돌 체크 및 자동 스냅
            if not start_node.walkable:
                print(f"❌ 시작점이 장애물/마진에 막혀 있습니다! (위치: {start})")
                return []
            
            # 목적지가 막혀있으면 가장 가까운 walkable 셀로 스냅
            if not goal_node.walkable:
                print(f"⚠️ 목적지가 장애물/마진에 막혀 있습니다! (위치: {goal})")
                print(f"🔧 가장 가까운 갈 수 있는 지점을 찾는 중...")
                
                snapped_idx = self._find_nearest_walkable(gx, gz, max_search_radius=SNAP_RADIUS)
                if snapped_idx is None:
                    print(f"❌ 주변에 갈 수 있는 지점이 없습니다!")
                    return []
                
                gx, gz = snapped_idx
                goal_node = self._grid[gx][gz]
                snapped_world = self.grid_index_to_world(gx, gz)
                print(f"✅ 목적지 조정: {goal} → {snapped_world} (거리: {self._distance(goal, snapped_world):.1f}m)")

            open_set: List[_Node] = [start_node]
            closed_set: set[_Node] = set()

            # g/h 비용 초기화
            for ix in range(self.grid_size_x):
                for iz in range(self.grid_size_z):
                    node = self._grid[ix][iz]
                    node.g_cost = 0
                    node.h_cost = 0
                    node.parent = None

            while open_set:
                # f_cost(동점이면 h_cost) 기준으로 최소값 노드 선택
                current = min(open_set, key=lambda n: (n.f_cost, n.h_cost))
                if current is goal_node:
                    # 목표 도달
                    return self._reconstruct_path(start_node, goal_node)

                open_set.remove(current)
                closed_set.add(current)

                for neighbor in self._neighbors(current):
                    if not neighbor.walkable or neighbor in closed_set:
                        continue

                    # 이동 비용 + 장애물 근접 비용
                    move_cost = self._distance_cost(current, neighbor)
                    proximity_penalty = int(neighbor.proximity_cost)
                    new_g = current.g_cost + move_cost + proximity_penalty
                    
                    if new_g < neighbor.g_cost or neighbor not in open_set:
                        neighbor.g_cost = new_g
                        neighbor.h_cost = self._distance_cost(neighbor, goal_node)
                        neighbor.parent = current
                        if neighbor not in open_set:
                            open_set.append(neighbor)

            # here: no path
            return []
        
        finally:
            # 장애물 복구
            if obstacles_backup is not None:
                self._obstacles = obstacles_backup
                self._grid_valid = False  # 다음 호출 시 재생성

    def _find_nearest_walkable(self, grid_x: int, grid_z: int, max_search_radius: int = SNAP_RADIUS) -> Optional[Tuple[int, int]]:
        """BFS로 가장 가까운 walkable 셀 찾기"""
        from collections import deque
        
        visited = set()
        queue = deque([(grid_x, grid_z, 0)])  # (x, z, distance)
        visited.add((grid_x, grid_z))
        
        while queue:
            cx, cz, dist = queue.popleft()
            
            # 최대 탐색 반경 초과
            if dist > max_search_radius:
                break
            
            # walkable 셀 발견
            if 0 <= cx < self.grid_size_x and 0 <= cz < self.grid_size_z:
                if self._grid[cx][cz].walkable:
                    return (cx, cz)
            
            # 8방향 탐색
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dz == 0:
                        continue
                    
                    nx, nz = cx + dx, cz + dz
                    if (nx, nz) not in visited:
                        if 0 <= nx < self.grid_size_x and 0 <= nz < self.grid_size_z:
                            visited.add((nx, nz))
                            queue.append((nx, nz, dist + 1))
        
        return None
    
    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """두 점 사이의 유클리드 거리"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _neighbors(self, node: _Node) -> Iterable[_Node]:
        """상하좌우(+대각선) 이웃 노드"""
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue

                # 대각선 허용 안 할 때는 상하좌우만
                if not self.allow_diagonal and abs(dx) + abs(dz) > 1:
                    continue

                ix = node.ix + dx
                iz = node.iz + dz

                if 0 <= ix < self.grid_size_x and 0 <= iz < self.grid_size_z:
                    yield self._grid[ix][iz]

    @staticmethod
    def _distance_cost(a: _Node, b: _Node) -> int:
        """
        A* 휴리스틱 및 이동 비용 계산용
        - 대각선 비용을 14, 직선 비용을 10 으로 두는 그리드 A* 전통 사용
        """
        dx = abs(a.ix - b.ix)
        dz = abs(a.iz - b.iz)
        diag = min(dx, dz)
        straight = abs(dx - dz)
        return COST_DIAGONAL * diag + COST_STRAIGHT * straight

    def _reconstruct_path(
        self,
        start_node: _Node,
        goal_node: _Node,
    ) -> List[Tuple[float, float]]:
        """goal 에서 parent 를 따라 start 까지 거슬러 올라간 뒤 월드 좌표 리스트로 반환"""
        path_nodes: List[_Node] = []
        cur: Optional[_Node] = goal_node

        while cur is not None and cur is not start_node:
            path_nodes.append(cur)
            cur = cur.parent
        if cur is start_node:
            path_nodes.append(start_node)

        path_nodes.reverse()
        world_path: List[Tuple[float, float]] = [
            self.grid_index_to_world(n.ix, n.iz) for n in path_nodes
        ]
        return world_path

    # ------------------------------------------------------------------
    # 시각화 (교육용)
    # ------------------------------------------------------------------
    def plot(self, path, current_pos, current_yaw, trajectory=None, title="Path", filename="path.png", show_grid=True, global_obstacles=None):
        # plt.figure(figsize=(10, 10))
        if self._obstacles is None: return

        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 0. 장애물 근접 비용 히트맵 표시 (선택적)
        if self._proximity_map and self.safety_weight > 0:
            import numpy as np
            prox_array = np.array(self._proximity_map).T  # 전치하여 올바른 방향으로
            extent = [self.grid_min_x, self.grid_max_x, self.grid_min_z, self.grid_max_z]
            im = ax.imshow(prox_array, extent=extent, origin='lower', 
                          cmap='YlOrRd', alpha=0.3, aspect='auto')
            # plt.colorbar(im, ax=ax, label='Proximity Cost', shrink=0.6)
        
        # 1. 장애물 그리기
        for obs in self._obstacles:
            # 실제 장애물 (회색)
            ax.add_patch(plt.Rectangle((obs.x_min, obs.z_min), obs.size_x, obs.size_z, color='#444444', alpha=0.8))
            # 마진 영역 (붉은 점선)
            ax.add_patch(plt.Rectangle(
                (obs.x_min - self.obstacle_margin, obs.z_min - self.obstacle_margin),
                obs.size_x + self.obstacle_margin*2, obs.size_z + self.obstacle_margin*2,
                color='red', alpha=0.1, linestyle='--'
            ))

        if hasattr(self, '_mask_zones'): # _mask_zones가 정의되어 있는지 확인
            for zone in self._mask_zones:
                ax.add_patch(plt.Rectangle(
                    (zone.x_min, zone.z_min), 
                    zone.size_x, 
                    zone.size_z, 
                    color="#010364",      # 마스킹 영역은 파란색
                    alpha=0.3,         # 투명하게 설정하여 경로와 겹쳐보이게 함
                    label='Masked Zone'
                ))

        # # 2. 경로 그리기
        if path:
            xs = [p[0] for p in path]
            zs = [p[1] for p in path]
            plt.plot(xs, zs, "#0D7200", linewidth=2, label="Path")
            plt.plot(xs[-1], zs[-1], "r*", markersize=15, label="Goal") # 목표 지점 별표

        # 전역 장애물 표시 (빨간 점)
        if global_obstacles:
            for (gx, gz) in global_obstacles:
                ax.plot(gx, gz, 'r.', markersize=3, alpha=0.5)

        #  DWA 예측 경로 그리기 (시안색 굵은 선 + 시작/끝점 표시)
        if trajectory is not None and len(trajectory) > 0:
            tx = trajectory[:, 0]  # x좌표들
            ty = trajectory[:, 1]  # z좌표들
            # DWA 궤적 (시안색 굵은 선)
            plt.plot(tx, ty, "c-", linewidth=3, label="DWA Local Traj", zorder=10)
            # 시작점 (현재 위치)
            plt.plot(tx[0], ty[0], "co", markersize=8, zorder=11)
            # 끝점 (예측 종료 위치)
            plt.plot(tx[-1], ty[-1], "c^", markersize=10, label="DWA End", zorder=11)

        # 3. 현재 탱크 위치 & 방향 그리기
        if current_pos:
            cx, cz = current_pos
            ax.plot(cx, cz, "go", markersize=10, label="Tank") # 탱크 위치 (초록 점)
            
            if current_yaw is not None:
                # 화살표로 방향 표시 (길이 5m)
                arrow_len = 5.0
                # 수학적 각도 변환 (Unity 좌표계 고려)
                # Unity: Y축 회전, 0도가 북쪽(Z+) -> 수학: 90도가 북쪽
                # 간단히 sin, cos으로 표현
                dx = math.sin(math.radians(current_yaw)) * arrow_len
                dy = math.cos(math.radians(current_yaw)) * arrow_len
                ax.arrow(cx, cz, dx, dy, head_width=2, head_length=2, fc='lime', ec='lime')

        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.5)

        ax.set_aspect("equal")
        ax.set_xlim(self.grid_min_x, self.grid_max_x)
        ax.set_ylim(self.grid_min_z, self.grid_max_z)
        if title: ax.set_title(title)
        ax.legend()

        if filename:
            plt.savefig(filename)
            plt.close(fig) # 메모리 해제
        else:
            plt.show()

    def set_obstacle_margin(self, margin: float):
        self.obstacle_margin = float(margin)
        self._grid_valid = False
        print(f"📏 A* obstacle_margin 변경: {margin}")
