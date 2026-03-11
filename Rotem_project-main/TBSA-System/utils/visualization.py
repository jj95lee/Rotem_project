"""
시각화 관리 및 경로 저장 유틸리티
"""
import io
import platform
import numpy as np
import time
import threading
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# OS별 자동 폰트 설정 코드
def set_korean_font():
    os_name = platform.system()

    if os_name == "Windows":
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif os_name == "Linux":
        # sudo apt-get install -y fonts-nanum 으로 설치되어져 있어야 함
        plt.rcParams['font.family'] = 'NanumGothic'
    
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
    print(f"현재 운영체제: {os_name}, 설정된 폰트: {plt.rcParams['font.family'][0]}")

set_korean_font()

# ---------------------------------------------------------
# 경로 이미지 저장 함수 (HybridController에서 호출용)
# ---------------------------------------------------------
def save_path_image(planner, path, current_pos, current_yaw, filename="path_debug.png", title="Path", show_grid=False, state_manager=None):
    """경로 이미지 저장 (AStarPlanner.plot 기능 활용) - DWA trajectory 포함"""
    save_path = os.path.join(os.getcwd(), filename)

    global_obs = state_manager.global_obstacles if state_manager else None
    
    # 🆕 DWA local trajectory 가져오기
    dwa_traj = None
    if state_manager and state_manager.last_dwa_traj is not None:
        dwa_traj = state_manager.last_dwa_traj
    
    planner.plot(
        path=path,
        current_pos=current_pos,
        current_yaw=current_yaw,
        trajectory=dwa_traj,  # 🆕 DWA trajectory 전달
        title=title,
        filename=save_path,
        show_grid=show_grid,
        global_obstacles=global_obs
    )
    print(f"💾 경로 이미지 저장 완료: {save_path}")


class VisualizationManager:
    """시각화 렌더링 관리 (웹 모니터링용)"""
    
    def __init__(self, state_manager, grid_size):
        self.state = state_manager
        self.grid_size = grid_size
        
        # 렌더링 캐시
        self._render_cache = {
            "costmap": {"key": None, "png": None, "ts": 0.0},
            "global":  {"key": None, "png": None, "ts": 0.0},
            "local":   {"key": None, "png": None, "ts": 0.0},
            "path":    {"key": None, "png": None, "ts": 0.0},  # 경로 캐시 추가
        }
        self._render_lock = threading.Lock()
        
        # 실제 이동 경로 기록용
        self.history_trail = []
        self.last_seq = -1

    def get_status_json(self):
        """상태 정보 JSON"""
        # ✅ 3번 개선: 현재 경로 노드 계산
        current_node = 0
        if self.state.global_path and self.state.robot_pose:
            cx, _, cz = self.state.robot_pose
            path_x = [p[0] for p in self.state.global_path]
            path_z = [p[1] for p in self.state.global_path]
            
            # 현재 로봇 위치에서 가장 가까운 경로 노드 찾기
            min_dist = float('inf')
            for i, (px, pz) in enumerate(zip(path_x, path_z)):
                dist = np.sqrt((cx - px)**2 + (cz - pz)**2)
                if dist < min_dist:
                    min_dist = dist
                    current_node = i + 1  # 1부터 시작
        
        return {
            "costmap_version": self.state.costmap_version,
            "global_path_version": self.state.global_path_version,
            "local_traj_version": self.state.local_traj_version,
            "destination": list(self.state.destination) if self.state.destination else None,
            "tank_pose": list(self.state.robot_pose) if self.state.robot_pose else None,
            "tank_yaw_deg": self.state.robot_yaw_deg,
            "path_nodes": len(self.state.global_path) if self.state.global_path else 0,
            "current_node": current_node,  # ✅ 3번 개선: 현재 노드 추가
            "costmap_stats": self.state.costmap_stats,
            "seq": self.state.seq
        }
    
    def render_scene(self, mode, planner=None):
        """장면 렌더링 (costmap, global, local)"""
        # 비차단 방식 락 획득
        if not self._render_lock.acquire(blocking=False):
            cached = self._render_cache.get(mode, None)
            if cached and cached["png"] is not None:
                return io.BytesIO(cached["png"])
            else:
                return self._placeholder_png("Rendering...")
        
        try:
            if self.state.costmap is None or self.state.costmap_origin is None:
                buf = self._placeholder_png("Costmap not ready")
                png_bytes = buf.getvalue()
                self._render_cache[mode] = {"key": None, "png": png_bytes, "ts": time.time()}
                return io.BytesIO(png_bytes)

            costmap = self.state.costmap
            origin = self.state.costmap_origin
            h, w = costmap.shape
            x0, z0 = float(origin[0]), float(origin[1])
            extent = [x0, x0 + w * self.grid_size, z0, z0 + h * self.grid_size]

            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            
            # 1. A* 장애물 정보 표시 (planner가 전달된 경우)
            if planner and mode == "global":
                for obs in planner._obstacles:
                    rect = matplotlib.patches.Rectangle(
                        (obs.x_min, obs.z_min), 
                        obs.size_x, obs.size_z, 
                        color='red', alpha=0.4, label='Obs'
                    )
                    ax.add_patch(rect)

            # 2. Costmap 이미지 배경
            ax.imshow(costmap, origin="lower", extent=extent, 
                      vmin=0.0, vmax=1.0, cmap="gray_r", interpolation="nearest", alpha=0.6)

            # 3. 목적지 표시 (빨간 별 모양)
            if self.state.destination is not None:
                ax.scatter([self.state.destination[0]], [self.state.destination[1]], 
                          s=150, c='red', marker="*", label='Goal', zorder=10)

            # 4. 글로벌 경로 표시
            if mode in ("global", "local"):
                if self.state.global_path:
                    xs = [p[0] for p in self.state.global_path]
                    zs = [p[1] for p in self.state.global_path]
                    ax.plot(xs, zs, 'b-', linewidth=1.5, label='Global', alpha=0.7)

            # 5. 로컬 궤적 (DWA)
            if mode == "local":
                if self.state.last_dwa_traj is not None and len(self.state.last_dwa_traj) > 1:
                    try:
                        ax.plot(self.state.last_dwa_traj[:, 0], 
                               self.state.last_dwa_traj[:, 1], 
                               'r-', linewidth=2.0, label='Local', alpha=0.8)
                    except Exception:
                        pass
                if self.state.last_dwa_target is not None:
                    ax.scatter([self.state.last_dwa_target[0]], 
                              [self.state.last_dwa_target[1]], 
                              s=50, c='orange', marker="x", label='Target')

            # 6. 로봇 현재 위치
            if self.state.robot_pose is not None:
                ax.scatter([self.state.robot_pose[0]], [self.state.robot_pose[1]], 
                          s=80, c='green', marker='o', label='Tank', edgecolors='white', zorder=15)

            # 뷰 범위 설정
            if self.state.robot_pose is not None:
                if mode == 'global':
                    ax.set_xlim(0, 300)
                    ax.set_ylim(0, 300)
                else:
                    cx, cz = float(self.state.robot_pose[0]), float(self.state.robot_pose[1])
                    r = 25.0
                    ax.set_xlim(cx - r, cx + r)
                    ax.set_ylim(cz - r, cz + r)

            # 7. 타이틀 설정 (SEQ 정보 포함)
            if mode == "costmap":
                title_text = f"Costmap v{self.state.costmap_version}"
            elif mode == "global":
                title_text = f"Global Overview (SEQ: {self.state.seq}) v{self.state.global_path_version}"
            else:
                title_text = f"Local Traj (SEQ: {self.state.seq}) v{self.state.local_traj_version}"
            
            ax.set_title(title_text, fontsize=10, fontweight='bold')
            ax.set_xlabel("X", fontsize=9)
            ax.set_ylabel("Z", fontsize=9)
            
            if mode in ("global", "local"):
                ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

            # PNG 저장
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            png_bytes = buf.getvalue()

            self._render_cache[mode] = {"key": None, "png": png_bytes, "ts": time.time()}
            return io.BytesIO(png_bytes)
        
        finally:
            self._render_lock.release()
    
    @staticmethod
    def _placeholder_png(text: str = "No data", w: int = 6, h: int = 6):
        """플레이스홀더 이미지"""
        fig, ax = plt.subplots(figsize=(w, h), dpi=100)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf
    
    def render_path(self, planner):
        """실시간 경로 + 장애물 + 전차 위치 시각화 (path_debug.png 스타일)"""
        if not self.state.global_path or not self.state.robot_pose:
            return self._placeholder_png("No path")
        
        try:
            # 1. path_debug.png와 동일한 가로세로비 설정
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # 2. 장애물 그리기 (참조 이름 수정: _obstacles)
            if planner and hasattr(planner, '_obstacles'):
                for obs in planner._obstacles:
                    # 실제 장애물 (진한 회색/갈색)
                    rect = plt.Rectangle(
                        (obs.x_min, obs.z_min),
                        obs.size_x, obs.size_z,
                        color='#5D4037', alpha=0.8, zorder=2
                    )
                    ax.add_patch(rect)
                    
                    # 마진 영역 (붉은 외곽선 - 플래너 마진 활용)
                    margin = planner.obstacle_margin
                    margin_rect = plt.Rectangle(
                        (obs.x_min - margin, obs.z_min - margin),
                        obs.size_x + margin*2, obs.size_z + margin*2,
                        fill=False, edgecolor='red', linestyle='--', alpha=0.2, zorder=1
                    )
                    ax.add_patch(margin_rect)
            
            # 3. 글로벌 경로 그리기 (파란색 실선)
            path_x = [p[0] for p in self.state.global_path]
            path_z = [p[1] for p in self.state.global_path]
            ax.plot(path_x, path_z, 'b-', linewidth=2, label='경로(Global Path)', zorder=3)
            
            # 4. 현재 위치 및 방향 (초록색 점 + 화살표)
            cx, cz = self.state.robot_pose
            ax.plot(cx, cz, 'go', markersize=10, label='현재 위치', zorder=5)
            
            if self.state.robot_yaw_deg is not None:
                yaw_rad = np.radians(self.state.robot_yaw_deg)
                arrow_len = 5.0
                dx = arrow_len * np.sin(yaw_rad)
                dz = arrow_len * np.cos(yaw_rad)
                ax.arrow(cx, cz, dx, dz, head_width=2, head_length=2, 
                        fc='lime', ec='green', zorder=6)
            
            # 목적지를 검정색 깃발로 표시 (전체 경로 이미지용)
            if self.state.destination:
                dest_x, dest_z = self.state.destination[0], self.state.destination[1]
                
                # 검정색 깃발 기둥
                ax.plot([dest_x, dest_x], [dest_z - 2.5, dest_z + 2.5], 
                       'k-', linewidth=2.5, zorder=4)
                
                # 검정색 깃발 모양 (삼각형)
                from matplotlib.patches import Polygon
                flag_vertices = np.array([
                    [dest_x, dest_z + 2.5],
                    [dest_x + 2.5, dest_z],
                    [dest_x, dest_z - 1]
                ])
                flag = Polygon(flag_vertices, color='black', alpha=0.8, 
                              zorder=4, edgecolor='darkgray', linewidth=1)
                ax.add_patch(flag)
                
                ax.plot(dest_x, dest_z, 'k*', markersize=15, label='목적지(깃발)', zorder=4)
            
            # 범위 설정 (path_debug.png와 동일하게 플래너의 설정 범위 사용)
            # ===== [뷰 범위: Path 기반 자동 크롭] =====
            xs = path_x + [cx]
            zs = path_z + [cz]

            if self.state.destination:
                xs.append(self.state.destination[0])
                zs.append(self.state.destination[1])

            margin = 10.0  # 화면 여유 (m)
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(zs) - margin, max(zs) + margin)

            # 시각적 밀도 강화
            ax.set_aspect('equal')  # 비율 유지
            ax.set_title(f'실시간 경로 추적', fontsize=12)
            ax.legend(loc='upper right', fontsize=8)
            
            # 버퍼 저장 및 반환
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return buf
            
        except Exception as e:
            plt.close('all') # 에러 발생 시에도 모두 닫기
            print(f"⚠️ 대시보드 렌더링 오류: {e}")
            return self._placeholder_png(f"Error: {e}")
    
    def render_realtime_path_image(self, planner, image_size=(640, 640)):
        """
        실시간 경로 이미지 생성
        - 전차: 크고 선명한 빨간색 동그라미
        - 경로: 실제 이동 궤적(회색 실선) + 남은 경로(파란 점선)
        """

        if not self.state.global_path or not self.state.robot_pose:
            return self._placeholder_png("데이터 수신 대기 중...")
        
        try:
            # 이미지 크기 설정
            width, height = image_size
            dpi = 100
            fig_w = width / dpi
            fig_h = height / dpi
            
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.set_facecolor('#f5f5f5')  # 연한 회색 배경
            
            # 현재 로봇 위치
            cx, _, cz = self.state.robot_pose

            # ═══════════════════════════════════════════════════════════════
            # 1. 실제 이동 궤적 업데이트
            # ═══════════════════════════════════════════════════════════════
            if self.state.seq != self.last_seq:
                self.history_trail = []
                self.last_seq = self.state.seq
            
            # 위치가 조금이라도 변하면 기록 (중복 방지)
            if not self.history_trail or np.hypot(self.history_trail[-1][0] - cx, self.history_trail[-1][1] - cz) > 0.1:
                self.history_trail.append((cx, cz))  

            if len(self.history_trail) > 1:
                hx = [p[0] for p in self.history_trail]
                hz = [p[1] for p in self.history_trail]
                # 진한 회색 실선으로 "발자취"를 명확히 표시
                ax.plot(hx, hz, color='#757575', linewidth=3.0, 
                        alpha=0.7, label='이동 궤적', zorder=2)         
            
            # ═══════════════════════════════════════════════════════════════
            # 2. 장애물 그리기 (회색 사각형 + 빨간 마진)
            # ═══════════════════════════════════════════════════════════════
            if planner and hasattr(planner, '_obstacles'):
                for obs in planner._obstacles:
                    # 실제 장애물 (진한 회색)
                    rect = plt.Rectangle(
                        (obs.x_min, obs.z_min),
                        obs.size_x, obs.size_z,
                        color='#5D4037', alpha=0.85, zorder=2,
                        edgecolor='#3E2723', linewidth=1
                    )
                    ax.add_patch(rect)
                    
                    # 마진 영역 (붉은 점선)
                    margin = planner.obstacle_margin
                    margin_rect = plt.Rectangle(
                        (obs.x_min - margin, obs.z_min - margin),
                        obs.size_x + margin*2, obs.size_z + margin*2,
                        fill=False, edgecolor='#FF5252', linestyle=':', 
                        linewidth=1, alpha=0.5, zorder=1
                    )
                    ax.add_patch(margin_rect)
            
            # ═══════════════════════════════════════════════════════════════
            # 3. 남은 계획 경로 (점선, 진한 파란색)
            # ═══════════════════════════════════════════════════════════════
            path_x = [p[0] for p in self.state.global_path]
            path_z = [p[1] for p in self.state.global_path]
            
            # 현재 위치에서 가장 가까운 노드 찾기
            min_dist = float('inf')
            closest_idx = 0
            for i, (px, pz) in enumerate(zip(path_x, path_z)):
                dist = np.hypot(cx - px, cz - pz)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            # 남은 경로만 그림 (파란색 실선 + 점선)       
            if closest_idx < len(path_x) - 1:
                remain_x = path_x[closest_idx:]
                remain_z = path_z[closest_idx:]
                ax.plot(remain_x, remain_z, 
                       color='#2962FF', linewidth=2.5, linestyle='-',
                        label='남은 경로', zorder=3)
            
            # ═══════════════════════════════════════════════════════════════
            # 4. 목적지 (검정색 깃발)
            # ═══════════════════════════════════════════════════════════════
            if self.state.destination:
                dest_x, dest_z = self.state.destination[0], self.state.destination[1]
                ax.plot([dest_x, dest_x], [dest_z, dest_z + 6.0], 'k-', linewidth=3, zorder=4)
                
                from matplotlib.patches import Polygon
                flag_vertices = np.array([
                    [dest_x, dest_z + 6.0], [dest_x + 4.0, dest_z + 4.0], [dest_x, dest_z + 2.0]
                ])
                flag = Polygon(flag_vertices, color='black', alpha=0.9, zorder=4)
                ax.add_patch(flag)
                ax.plot(dest_x, dest_z, 'k*', markersize=18, zorder=4)
                
                # 남은 거리 표시
                dist_to_goal = np.hypot(dest_x - cx, dest_z - cz)
                ax.text(dest_x + 3, dest_z + 3, f'{dist_to_goal:.1f}m', 
                       fontsize=10, color='#333', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # ═══════════════════════════════════════════════════════════════
            # 5. 현재 전차 위치 (빨간색 갈매기 화살표)
            # ═══════════════════════════════════════════════════════════════
            # yaw_rad = np.radians(self.state.player_body_x) if self.state.player_body_x is not None else 0
            
            # # 화살표 크기 대폭 키움 (고정 크기 12m) -> 맵에서 확실히 보임
            # arrow_size = 12.0  
            
            # # 갈매기 모양 좌표
            # seagull_x = np.array([0, -arrow_size*0.6, 0, arrow_size*0.6])
            # seagull_z = np.array([arrow_size, -arrow_size*0.4, 0, -arrow_size*0.4])
            
            # # 회전 변환
            # cos_yaw = np.cos(yaw_rad)
            # sin_yaw = np.sin(yaw_rad)
            # rotated_x = seagull_x * cos_yaw - seagull_z * sin_yaw + cx
            # rotated_z = seagull_x * sin_yaw + seagull_z * cos_yaw + cz
            
            # # [핵심] 빨간색 내부 + 노란색 테두리 (가시성 극대화)
            # from matplotlib.patches import Polygon
            # tank_poly = Polygon(list(zip(rotated_x, rotated_z)), 
            #                     facecolor='#D50000',  # 밝은 빨강
            #                     edgecolor='#FFEA00',  # 형광 노랑 테두리
            #                     linewidth=2.5,        # 두꺼운 테두리
            #                     alpha=1.0, 
            #                     zorder=10)            # 맨 위에 그림
            # ax.add_patch(tank_poly)
            tank_circle = Circle((cx, cz),
                                 radius=4.0,
                                 facecolor='#D50000',
                                 edgecolor='#FFEA00',  # 형광 노랑 테두리
                                 linewidth=2.5,        # 두꺼운 테두리
                                 alpha=1.0, 
                                 zorder=10)
            ax.add_patch(tank_circle)
            
            # ═══════════════════════════════════════════════════════════════
            # 6. 축 범위 설정 (SEQ별 고정 뷰포트 적용)
            # ═══════════════════════════════════════════════════════════════
            current_seq = self.state.seq
            view_margin = 5.0
            
            if current_seq == 1:
                ax.set_xlim(65 - view_margin, 200 + view_margin)
                ax.set_ylim(0, 220 + view_margin)
            elif current_seq == 3:
                ax.set_xlim(0, 200 + view_margin)
                ax.set_ylim(150 - view_margin, 300)
            else:
                all_x = path_x + [cx]
                all_z = path_z + [cz]
                if self.state.destination:
                    all_x.append(self.state.destination[0])
                    all_z.append(self.state.destination[1])
                
                margin = 15.0
                if all_x and all_z:
                    x_min, x_max = min(all_x) - margin, max(all_x) + margin
                    z_min, z_max = min(all_z) - margin, max(all_z) + margin
                    
                    # 비율 유지 로직 (기존 코드 재사용)
                    x_range = x_max - x_min
                    z_range = z_max - z_min
                    if x_range > z_range:
                        diff = (x_range - z_range) / 2
                        z_min -= diff
                        z_max += diff
                    else:
                        diff = (z_range - x_range) / 2
                        x_min -= diff
                        x_max += diff
                        
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(z_min, z_max)
                else:
                    # 데이터 없을 때 기본 맵 전체
                    ax.set_xlim(0, 300)
                    ax.set_ylim(0, 300)
            
            ax.set_aspect('equal')
            
            # ═══════════════════════════════════════════════════════════════
            # 7. 스타일 및 정보 표시
            # ═══════════════════════════════════════════════════════════════
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # 타이틀
            ax.set_title(f'실시간 경로 추적 [{timestamp}]', 
                        fontsize=11, fontweight='bold', pad=10)
            
            # 범례 (좌측 상단)
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
            # 그리드
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # 축 라벨 제거 (깔끔하게)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='both', labelsize=8)
            
            # PNG 저장
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                       facecolor=fig.get_facecolor(), edgecolor='none')
            buf.seek(0)
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            plt.close('all')
            print(f"⚠️ 실시간 경로 이미지 렌더링 오류: {e}")
            import traceback
            traceback.print_exc()
            return self._placeholder_png(f"Error: {e}")

    def render_realtime_snapshot(self, planner):
        """
        실시간 경로 추적 스냅샷 생성
        전역 경로 + 현재 위치 + 로컬 정보를 함께 표시
        """
        if not self.state.global_path or not self.state.robot_pose:
            return self._placeholder_png("No path data")
        
        try:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            
            cx, cz = self.state.robot_pose
            
            # 1. 장애물 그리기
            if planner and hasattr(planner, '_obstacles'):
                for obs in planner._obstacles:
                    rect = plt.Rectangle(
                        (obs.x_min, obs.z_min),
                        obs.size_x, obs.size_z,
                        color='#5D4037', alpha=0.8, zorder=2
                    )
                    ax.add_patch(rect)
                    
                    # 마진 영역
                    margin = planner.obstacle_margin
                    margin_rect = plt.Rectangle(
                        (obs.x_min - margin, obs.z_min - margin),
                        obs.size_x + margin*2, obs.size_z + margin*2,
                        fill=False, edgecolor='red', linestyle='--', alpha=0.3, zorder=1
                    )
                    ax.add_patch(margin_rect)
            
            # ✅ 2번 개선: 실시간 경로는 점선으로 표시
            path_x = [p[0] for p in self.state.global_path]
            path_z = [p[1] for p in self.state.global_path]
            ax.plot(path_x, path_z, 'b--', linewidth=2.5, label='계획 경로(점선)', 
                   dashes=(5, 5), zorder=3)  # 점선 스타일 (5px 선, 5px 간격)
            
            # 3. 경로 노드 표시 (작은 점)
            ax.scatter(path_x, path_z, c='blue', s=10, alpha=0.4, zorder=3)
            
            # 현재 위치 아이콘을 갈매기 화살표로 변경 (회전 가능)
            if self.state.player_body_x is not None:
                yaw_rad = np.radians(self.state.player_body_x)
                # 갈매기 화살표 좌표 계산 (로봇 회전에 따라 회전)
                arrow_size = 5.0
                # 기본 갈매기 모양: 중앙, 좌상단, 우상단
                seagull_x = np.array([0, -arrow_size/2, arrow_size/2])
                seagull_z = np.array([arrow_size, 0, 0])
                
                # 회전 행렬 적용
                cos_yaw = np.cos(yaw_rad)
                sin_yaw = np.sin(yaw_rad)
                rotated_x = seagull_x * cos_yaw - seagull_z * sin_yaw + cx
                rotated_z = seagull_x * sin_yaw + seagull_z * cos_yaw + cz
                
                # 갈매기 화살표 그리기 (빨간색, 채운 다각형)
                from matplotlib.patches import Polygon
                seagull = Polygon(list(zip(rotated_x, rotated_z)), 
                                 color='red', alpha=0.9, zorder=5, 
                                 edgecolor='darkred', linewidth=2)
                ax.add_patch(seagull)
                
                # 범례용 표시
                ax.plot(cx, cz, 'r^', markersize=12, label='로봇(회전)', zorder=5)
            else:
                # yaw 정보 없으면 기본 빨간 마커
                ax.plot(cx, cz, 'r^', markersize=12, label='로봇', zorder=5,
                       markeredgecolor='darkred', markeredgewidth=2)
            
            # 목적지를 검정색 깃발로 표시
            if self.state.destination:
                # 검정색 깃발 모양 (깃발 + 기둥)
                dest_x, dest_z = self.state.destination[0], self.state.destination[1]
                
                # 깃발 기둥 (검정색 수직선)
                ax.plot([dest_x, dest_x], [dest_z - 3, dest_z + 3], 
                       'k-', linewidth=3, zorder=4)
                
                # 깃발 모양 (검정색 삼각형)
                from matplotlib.patches import Polygon
                flag_vertices = np.array([
                    [dest_x, dest_z + 3],      # 위쪽
                    [dest_x + 2.5, dest_z + 1],   # 우측
                    [dest_x, dest_z - 1]       # 아래쪽
                ])
                flag = Polygon(flag_vertices, color='black', alpha=0.85, 
                              zorder=4, edgecolor='darkgray', linewidth=1)
                ax.add_patch(flag)
                
                # ax.plot(dest_x, dest_z, 'k*', markersize=18, label='목적지(깃발)', zorder=4)
                
                # 목적지까지 거리 표시
                dist = np.hypot(self.state.destination[0] - cx, 
                               self.state.destination[1] - cz)
                ax.text(cx, cz - 8, f'목적지까지 {dist:.1f}m', 
                       ha='center', fontsize=11, color='#FF6B6B', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 6. 시간 정보 추가
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            snapshot_num = self.state.realtime_snapshot_index + 1
            
            # 범위 설정 (경로 전체 + 여유)
            all_x = path_x + [cx]
            all_z = path_z + [cz]
            if self.state.destination:
                all_x.append(self.state.destination[0])
                all_z.append(self.state.destination[1])
            
            margin = 15.0
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_z) - margin, max(all_z) + margin)
            
            ax.set_aspect('equal')
            ax.set_title(f'SEQ {self.state.seq} - 경로 추적 스냅샷 #{snapshot_num} [{timestamp}]', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return buf
            
        except Exception as e:
            plt.close('all')
            print(f"⚠️ 스냅샷 렌더링 오류: {e}")
            return self._placeholder_png(f"Error: {e}")
    
    def update_realtime_snapshots(self, planner):
        """
        실시간 스냅샷 업데이트 (5초마다 호출)
        번갈아가며 Realtime 1, Realtime 2 창에 저장
        """
        if self.state.seq not in [1, 3]:
            return  # SEQ 1, 3에서만 동작
        
        if not self.state.global_path or not self.state.robot_pose:
            return
        
        try:
            # 스냅샷 생성
            buf = self.render_realtime_snapshot(planner)
            snapshot_bytes = buf.getvalue()
            
            # 번갈아가며 저장 (0: Realtime 1, 1: Realtime 2)
            if self.state.realtime_snapshot_index == 0:
                self.state.realtime_snapshot_1_bytes = snapshot_bytes
                target_window = "Realtime 1"
            else:
                self.state.realtime_snapshot_2_bytes = snapshot_bytes
                target_window = "Realtime 2"
            
            # 다음 인덱스로 전환
            self.state.realtime_snapshot_index = 1 - self.state.realtime_snapshot_index
            self.state.realtime_snapshot_ts = time.time()
            
            print(f"📸 스냅샷 → {target_window} (SEQ {self.state.seq})")
            
        except Exception as e:
            print(f"⚠️ 스냅샷 업데이트 실패: {e}")

    def render_autonomous(self, planner, lidar_logger=None):
        """SEQ 4 자율주행 모드: Costmap + 경로 + LiDAR 장애물 통합 시각화"""
        if not self._render_lock.acquire(blocking=False):
            cached = self._render_cache.get("autonomous", None)
            if cached and cached.get("png") is not None:
                return io.BytesIO(cached["png"])
            else:
                return self._placeholder_png("Rendering...")

        try:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

            # 1. 가상 라이다 표시 (자주색 점)
            if hasattr(self.state, 'global_obstacles') and self.state.global_obstacles:
                obs_x = [o[0] for o in self.state.global_obstacles]
                obs_z = [o[1] for o in self.state.global_obstacles]
                ax.scatter(obs_x, obs_z, c='magenta', s=15, alpha=0.5, marker='o', label='가상 라이다', zorder=2)

            # 2. Costmap 배경 (있는 경우)
            if self.state.costmap is not None and self.state.costmap_origin is not None:
                costmap = self.state.costmap
                origin = self.state.costmap_origin
                h, w = costmap.shape
                x0, z0 = float(origin[0]), float(origin[1])
                extent = [x0, x0 + w * self.grid_size, z0, z0 + h * self.grid_size]

                # Costmap 이미지 (회색조, 장애물은 어둡게)
                ax.imshow(costmap, origin="lower", extent=extent,
                         vmin=0.0, vmax=1.0, cmap="gray_r", interpolation="nearest", alpha=0.5)

            # 3. 글로벌 경로 표시 (파란색 실선)
            if self.state.global_path:
                path_x = [p[0] for p in self.state.global_path]
                path_z = [p[1] for p in self.state.global_path]
                ax.plot(path_x, path_z, 'b-', linewidth=2.5, label='경로', zorder=5)

            # 4. DWA 로컬 궤적 (가상 라이다를 실제로 피하고 있는지)
            if self.state.last_dwa_traj is not None and len(self.state.last_dwa_traj) > 1:
                try:
                    ax.plot(self.state.last_dwa_traj[:, 0],
                           self.state.last_dwa_traj[:, 1],
                           'c-', linewidth=2.5, label='DWA 회피 궤적', zorder=6)
                except Exception:
                    pass

            # 5. 목적지 표시 (빨간 별)
            if self.state.destination:
                ax.plot(self.state.destination[0], self.state.destination[1],
                       'r*', markersize=18, label='목적지', zorder=9)


            # 뷰 범위 설정 (경로 + 현재 위치 기준 자동 조정)
            all_x = []
            all_z = []

            if self.state.robot_pose:
                all_x.append(self.state.robot_pose[0])
                all_z.append(self.state.robot_pose[1])

            if self.state.destination:
                all_x.append(self.state.destination[0])
                all_z.append(self.state.destination[1])

            if self.state.global_path:
                all_x.extend([p[0] for p in self.state.global_path])
                all_z.extend([p[1] for p in self.state.global_path])

            if all_x and all_z:
                margin = 15.0
                ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                ax.set_ylim(min(all_z) - margin, max(all_z) + margin)
            else:
                ax.set_xlim(0, 300)
                ax.set_ylim(0, 300)

            ax.set_aspect('equal')
            ax.set_title(f'SEQ 4 - 자율주행 (가상 라이다 감지)', fontsize=12, fontweight='bold')
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Z (m)', fontsize=10)
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)

            # PNG 저장 및 반환
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)

            png_bytes = buf.getvalue()

            # 캐시에 저장
            if "autonomous" not in self._render_cache:
                self._render_cache["autonomous"] = {}
            self._render_cache["autonomous"] = {"key": None, "png": png_bytes, "ts": time.time()}

            return io.BytesIO(png_bytes)

        except Exception as e:
            plt.close('all')
            print(f"⚠️ 자율주행 뷰 렌더링 오류: {e}")
            return self._placeholder_png(f"Error: {e}")

        finally:
            self._render_lock.release()

    def render_seq4_detailed(self, planner, image_size=(800, 800)):
        """
        SEQ 4 전용 상세 시각화 (장애물 + A* 경로 + PPO 궤적 + 로그)

        표시 항목:
        1. 장애물 사각형 (obstacle_rects) - 빨간색 테두리
        2. A* 전역 경로 - 파란색 실선
        3. PPO/DWA 로컬 궤적 - 주황색 점선
        4. 가상 LiDAR 스캔 - 자주색 점
        5. 현재 위치 및 방향 - 초록색 화살표
        6. 목적지 - 검정색 깃발
        7. 로그 정보 오버레이
        """
        if not self.state.robot_pose:
            return self._placeholder_png("SEQ 4 - 데이터 수신 대기 중...")

        try:
            width, height = image_size
            dpi = 100
            fig_w = width / dpi
            fig_h = height / dpi

            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.set_facecolor('#f0f0f0')  # 연한 회색 배경

            cx, _, cz = self.state.robot_pose

            # ═══════════════════════════════════════════════════════════════
            # 1. 장애물 사각형 표시 (obstacle_rects)
            # ═══════════════════════════════════════════════════════════════
            obstacle_count = 0
            if hasattr(self.state, 'obstacle_rects') and self.state.obstacle_rects:
                for obs in self.state.obstacle_rects:
                    x_min = obs.get("x_min", 0)
                    x_max = obs.get("x_max", 0)
                    z_min = obs.get("z_min", 0)
                    z_max = obs.get("z_max", 0)

                    width_obs = x_max - x_min
                    height_obs = z_max - z_min

                    # 장애물 본체 (진한 빨간색)
                    rect = plt.Rectangle(
                        (x_min, z_min), width_obs, height_obs,
                        facecolor='#D32F2F', edgecolor='#B71C1C',
                        alpha=0.7, linewidth=2, zorder=3
                    )
                    ax.add_patch(rect)

                    # 안전 마진 영역 (점선)
                    margin = 5.0  # SEQ4 기본 마진
                    if planner and hasattr(planner, 'obstacle_margin'):
                        margin = planner.obstacle_margin

                    margin_rect = plt.Rectangle(
                        (x_min - margin, z_min - margin),
                        width_obs + margin * 2, height_obs + margin * 2,
                        fill=False, edgecolor='#FF8A80',
                        linestyle='--', linewidth=1.5, alpha=0.6, zorder=2
                    )
                    ax.add_patch(margin_rect)
                    obstacle_count += 1

            # ═══════════════════════════════════════════════════════════════
            # 2. 가상 LiDAR 포인트 표시 (global_obstacles)
            # ═══════════════════════════════════════════════════════════════
            lidar_count = 0
            if hasattr(self.state, 'global_obstacles') and self.state.global_obstacles:
                obs_x = [o[0] for o in self.state.global_obstacles]
                obs_z = [o[1] for o in self.state.global_obstacles]
                ax.scatter(obs_x, obs_z, c='#9C27B0', s=20, alpha=0.6,
                          marker='o', label=f'가상 LiDAR ({len(obs_x)}점)', zorder=4)
                lidar_count = len(obs_x)

            # ═══════════════════════════════════════════════════════════════
            # 3. A* 전역 경로 표시 (파란색 실선)
            # ═══════════════════════════════════════════════════════════════
            path_nodes = 0
            if self.state.global_path:
                path_x = [p[0] for p in self.state.global_path]
                path_z = [p[1] for p in self.state.global_path]
                ax.plot(path_x, path_z, color='#1565C0', linewidth=3,
                       linestyle='-', label=f'A* 경로 ({len(path_x)}점)', zorder=5)

                # 경로 노드 점 표시
                ax.scatter(path_x, path_z, c='#1976D2', s=15, alpha=0.5, zorder=5)
                path_nodes = len(path_x)

            # ═══════════════════════════════════════════════════════════════
            # 4. PPO/DWA 로컬 궤적 표시 (주황색 점선)
            # ═══════════════════════════════════════════════════════════════
            if self.state.last_dwa_traj is not None and len(self.state.last_dwa_traj) > 1:
                try:
                    traj = self.state.last_dwa_traj
                    ax.plot(traj[:, 0], traj[:, 1],
                           color='#FF6F00', linewidth=2.5, linestyle='--',
                           label='PPO 궤적', zorder=6)

                    # 궤적 끝점 표시
                    ax.scatter([traj[-1, 0]], [traj[-1, 1]],
                              c='#FF6F00', s=80, marker='>', zorder=6)
                except Exception:
                    pass

            # ═══════════════════════════════════════════════════════════════
            # 5. 현재 위치 및 방향 (초록색 화살표)
            # ═══════════════════════════════════════════════════════════════
            yaw_rad = np.radians(self.state.robot_yaw_deg) if self.state.robot_yaw_deg else 0

            # 큰 화살표 표시
            arrow_size = 8.0
            seagull_x = np.array([0, -arrow_size*0.5, 0, arrow_size*0.5])
            seagull_z = np.array([arrow_size, -arrow_size*0.3, 0, -arrow_size*0.3])

            cos_yaw = np.cos(yaw_rad)
            sin_yaw = np.sin(yaw_rad)
            rotated_x = seagull_x * cos_yaw - seagull_z * sin_yaw + cx
            rotated_z = seagull_x * sin_yaw + seagull_z * cos_yaw + cz

            from matplotlib.patches import Polygon
            tank_poly = Polygon(list(zip(rotated_x, rotated_z)),
                               facecolor='#4CAF50', edgecolor='#1B5E20',
                               linewidth=2.5, alpha=0.9, zorder=10)
            ax.add_patch(tank_poly)

            # ═══════════════════════════════════════════════════════════════
            # 6. 목적지 표시 (검정색 깃발)
            # ═══════════════════════════════════════════════════════════════
            dist_to_goal = None
            if self.state.destination:
                dest_x, dest_z = self.state.destination
                dist_to_goal = np.hypot(dest_x - cx, dest_z - cz)

                # 깃발 기둥
                ax.plot([dest_x, dest_x], [dest_z, dest_z + 8], 'k-', linewidth=3, zorder=8)

                # 깃발 모양
                flag_vertices = np.array([
                    [dest_x, dest_z + 8],
                    [dest_x + 5, dest_z + 5.5],
                    [dest_x, dest_z + 3]
                ])
                flag = Polygon(flag_vertices, color='black', alpha=0.9, zorder=8)
                ax.add_patch(flag)

                ax.scatter([dest_x], [dest_z], c='black', s=100, marker='*', zorder=8)

            # ═══════════════════════════════════════════════════════════════
            # 7. 로그 정보 오버레이
            # ═══════════════════════════════════════════════════════════════
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # 정보 박스 텍스트
            info_lines = [
                f"시간: {timestamp}",
                f"위치: ({cx:.1f}, {cz:.1f})",
                f"방향: {self.state.robot_yaw_deg:.1f}°" if self.state.robot_yaw_deg else "방향: N/A",
                f"장애물: {obstacle_count}개",
                f"LiDAR 포인트: {lidar_count}개",
                f"경로 노드: {path_nodes}개",
                f"목표 거리: {dist_to_goal:.1f}m" if dist_to_goal else "목표: 없음"
            ]

            # 로그 메시지 추가
            if hasattr(self.state, 'last_log') and self.state.last_log:
                # 긴 로그는 자르기
                log_text = self.state.last_log[:40] + "..." if len(self.state.last_log) > 40 else self.state.last_log
                info_lines.append(f"로그: {log_text}")

            info_text = "\n".join(info_lines)

            # 정보 박스 표시 (좌측 상단)
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='gray', alpha=0.9),
                   family='monospace', zorder=20)

            # ═══════════════════════════════════════════════════════════════
            # 8. 뷰 범위 설정
            # ═══════════════════════════════════════════════════════════════
            all_x = [cx]
            all_z = [cz]

            if self.state.destination:
                all_x.append(self.state.destination[0])
                all_z.append(self.state.destination[1])

            if self.state.global_path:
                all_x.extend([p[0] for p in self.state.global_path])
                all_z.extend([p[1] for p in self.state.global_path])

            if hasattr(self.state, 'obstacle_rects') and self.state.obstacle_rects:
                for obs in self.state.obstacle_rects:
                    all_x.extend([obs.get("x_min", 0), obs.get("x_max", 0)])
                    all_z.extend([obs.get("z_min", 0), obs.get("z_max", 0)])

            margin = 20.0
            if all_x and all_z:
                x_min, x_max = min(all_x) - margin, max(all_x) + margin
                z_min, z_max = min(all_z) - margin, max(all_z) + margin

                # 비율 유지
                x_range = x_max - x_min
                z_range = z_max - z_min
                if x_range > z_range:
                    diff = (x_range - z_range) / 2
                    z_min -= diff
                    z_max += diff
                else:
                    diff = (z_range - x_range) / 2
                    x_min -= diff
                    x_max += diff

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(z_min, z_max)
            else:
                ax.set_xlim(0, 300)
                ax.set_ylim(0, 300)

            ax.set_aspect('equal')
            ax.set_title('SEQ 4 - PPO + A* 하이브리드 자율주행', fontsize=12, fontweight='bold', pad=10)
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Z (m)', fontsize=10)

            # PNG 저장 및 반환
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            buf.seek(0)
            plt.close(fig)

            return buf

        except Exception as e:
            plt.close('all')
            print(f"⚠️ SEQ 4 상세 시각화 오류: {e}")
            import traceback
            traceback.print_exc()
            return self._placeholder_png(f"Error: {e}")