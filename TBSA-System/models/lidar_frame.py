"""
LiDAR 데이터 처리 및 지형 분석
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation, generic_filter
from config import Config

GRID_SIZE = Config.Lidar.GRID_SIZE

# 전차 반경과 장애물 파라미터
TANK_RADIUS = Config.Terrain.TANK_RADIUS
INFLATION_RADIUS = Config.Terrain.INFLATION_RADIUS

# 지형 분석 파라미터
SLOPE_THRESH = Config.Terrain.SLOPE_THRESH
PLANE_THRESH = Config.Terrain.PLANE_THRESH
GROUND_RATIO_TH = Config.Terrain.GROUND_RATIO_TH
HEIGHT_STD_TH = Config.Terrain.HEIGHT_STD_TH
MIN_PTS_CLASSIFY = Config.Terrain.MIN_PTS_CLASSIFY
h_range = Config.Terrain.OBSTACLE_HEIGHT_TH # cond_vertical_wall

# 지형 비용 계산 변수
ROUGH_STD_NORM = Config.Terrain.ROUGH_STD_NORM
UNKNOWN_COST = Config.Terrain.UNKNOWN_COST
FILL_TERRAIN_COST = Config.Terrain.FILL_TERRAIN_COST
FILL_FINAL_COST = Config.Terrain.FILL_FINAL_COST

# 지형 비용 가중치
W_SLOPE = Config.Terrain.W_SLOPE
W_ROUGH = Config.Terrain.W_ROUGH
W_GROUND = Config.Terrain.W_GROUND


class LidarFrame:
    """LiDAR 프레임 데이터"""
    
    def __init__(self, lidar_pts, timestamp):
        self.lidar_pts = lidar_pts
        self.timestamp = timestamp

    def to_dataframe(self):
        """DataFrame 변환"""
        rows = []
        for pt in self.lidar_pts:
            rows.append({
                "time": float(self.timestamp),
                "angle": float(pt['angle']),
                "verticalAngle": float(pt['verticalAngle']),
                "distance": float(pt['distance']),
                "x": float(pt['position']['x']),
                "y": float(pt['position']['y']),
                "z": float(pt['position']['z']),
                "ringID": int(pt['channelIndex']),
                "isDetected": pt['isDetected'],
            })
        return pd.DataFrame(rows)


def point_plane_dist(row, coef):
    """점-평면 거리"""
    a, b, c = coef
    return abs(a*row["x_world"] + b*row["z_world"] + c - row["h_world"])


def gridify(df, grid_size=GRID_SIZE):
    """그리드화"""
    df = df.copy()
    df.rename(columns={
        'x':'x_world', 
        'y':'h_world', 
        'z':'z_world'}, 
        inplace=True)
    df['grid_x'] = np.floor(df['x_world'] / grid_size).astype(int)
    df['grid_z'] = np.floor(df['z_world'] / grid_size).astype(int)
    df['cell_id'] = df['grid_x'].astype(str) + '_' + df['grid_z'].astype(str)
    return df


def fit_local_planes(df, min_pts=10, plane_th=PLANE_THRESH):
    """로컬 평면 피팅"""
    df['is_ground'] = False
    df[['a','b','c']] = np.nan

    for _, cell in df.groupby('cell_id'):
        if len(cell) < min_pts:
            continue

        X = np.c_[cell['x_world'], cell['z_world'], np.ones(len(cell))]
        Y = cell['h_world'].values
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)

        dists = cell.apply(point_plane_dist, axis=1, coef=coef)
        ground_mask = dists < plane_th

        df.loc[cell.index, ['a','b','c']] = coef
        df.loc[cell.index, 'is_ground'] = ground_mask

    return df


def obstacle_distance_cost_vectorized(d, tank_radius=TANK_RADIUS, 
                                      inflation_radius=INFLATION_RADIUS, alpha=1.0):
    """장애물 거리 비용 (벡터화)"""
    cost = np.zeros_like(d, dtype=np.float32)

    lethal_mask = d <= tank_radius
    inflation_mask = (d > tank_radius) & (d < inflation_radius)

    cost[lethal_mask] = 1.0

    x = (d[inflation_mask] - tank_radius) / (inflation_radius - tank_radius)
    cost[inflation_mask] = np.exp(-alpha * x)

    return cost


def compute_cell_features(df):
    """셀별 특징 계산"""
    cell_df = df.groupby('cell_id').agg(
        grid_x=('grid_x', 'first'),
        grid_z=('grid_z', 'first'),
        n_pts=('cell_id', 'size'),
        ground_ratio=('is_ground', 'mean'),
        h_max = ('h_world', 'max'),
        h_min = ('h_world', 'min'),
        h_std=('h_world', 'std'),
        a=('a', 'mean'),
        b=('b', 'mean'),
    ).reset_index()
    
    # 전차와 가장 가까이에 있는 포인트들의 중앙 값을 현재 지면 높이로 사용
    near_points = df[df['distance'] < 10.0]['h_world']
    robot_ground_y = near_points.median() if not near_points.empty else np.percentile(df['h_world'], 5)

    cell_df['h_range'] = cell_df['h_max'] - cell_df['h_min']

    # NaN 처리
    cell_df['a'] = cell_df['a'].fillna(0.0)
    cell_df['b'] = cell_df['b'].fillna(0.0)
    cell_df['h_std'] = cell_df['h_std'].fillna(0.0)
    cell_df['ground_ratio'] = cell_df['ground_ratio'].fillna(0.0)

    # slope 계산
    denominator = np.sqrt(cell_df['a']**2 + 1 + cell_df['b']**2)
    arccos_arg = 1.0 / denominator
    arccos_arg = np.clip(arccos_arg, -1.0, 1.0)
    cell_df['slope_deg'] = np.degrees(np.arccos(arccos_arg))
    cell_df['slope_deg'] = cell_df['slope_deg'].fillna(0.0)

    # 장애물 판정
    enough_pts = cell_df['n_pts'] >= MIN_PTS_CLASSIFY
    cond_steep = cell_df['slope_deg'] > SLOPE_THRESH
    cond_vertical_wall = (cell_df['h_range'] > h_range) & (cell_df['ground_ratio'] < GROUND_RATIO_TH)
    
    min_x, max_x = cell_df['grid_x'].min(), cell_df['grid_x'].max()
    min_z, max_z = cell_df['grid_z'].min(), cell_df['grid_z'].max()
    width = int(max_x - min_x + 1)
    height = int(max_z - min_z + 1)
    dense_h = np.full((height, width), np.nan)

    for _, r in cell_df.iterrows():
        iz = int(r['grid_z'] - min_z)
        ix = int(r['grid_x'] - min_x)
        dense_h[iz, ix] = r['h_min']

    def max_diff(window):
        if np.all(np.isnan(window)): return 0.0
        return np.nanmax(window) - np.nanmin(window)
    
    local_step = generic_filter(dense_h, max_diff, size = 3)

    def get_step_val(r):
        iz, ix = int(r['grid_z'] - min_z), int(r['grid_x'] - min_x)
        return local_step[iz, ix] if (0 <= iz < height and 0 <= ix < width) else 0.0  

    step_map = cell_df.apply(get_step_val, axis = 1) 
    max_step_height = Config.Terrain.MAX_STEP_HEIGHT
    cond_relative_step = step_map > max_step_height

    cell_df['is_obstacle'] = (
        enough_pts & (cond_steep | cond_vertical_wall | cond_relative_step)
    )

    ### terrain cost 계산 ###
    slope_cost  = np.clip(cell_df['slope_deg'] / SLOPE_THRESH, 0, 1)
    
    # HEIGHT_STD_TH 이하 구간은 rough_cost = 0 (노이즈 데드존)
    eps = 1e-6
    den = max(ROUGH_STD_NORM - HEIGHT_STD_TH, eps)
    rough_cost = np.clip((cell_df['h_std'] - HEIGHT_STD_TH) / den, 0.0, 1.0)
    
    ground_cost = np.clip((GROUND_RATIO_TH - cell_df['ground_ratio']) / 
                         max(GROUND_RATIO_TH, eps), 0, 1)

    cell_df['terrain_cost'] = (
        W_SLOPE * slope_cost +
        W_ROUGH * rough_cost +
        W_GROUND * ground_cost
    )
    cell_df['terrain_cost'] = cell_df['terrain_cost'].fillna(FILL_TERRAIN_COST)

    # obstacle distance
    obs_xy = cell_df.loc[cell_df['is_obstacle'], ['grid_x', 'grid_z']].values * GRID_SIZE
    cell_xy = cell_df[['grid_x', 'grid_z']].values * GRID_SIZE

    if len(obs_xy) == 0:
        cell_df['distance_cost'] = 0.0
    else:
        tree = cKDTree(obs_xy)
        dist, _ = tree.query(cell_xy)
        cell_df['distance_cost'] = obstacle_distance_cost_vectorized(dist)

    # final cost
    cell_df['final_cost'] = np.maximum(
        cell_df['terrain_cost'],
        cell_df['distance_cost']
    )

    # unknown 셀 처리
    if 'n_pts' in cell_df.columns:
        unknown_mask = cell_df['n_pts'] < MIN_PTS_CLASSIFY
        cell_df.loc[unknown_mask, 'is_obstacle'] = False
        cell_df.loc[unknown_mask, 'final_cost'] = UNKNOWN_COST
    
    cell_df['final_cost'] = cell_df['final_cost'].fillna(FILL_FINAL_COST)

    return cell_df


def build_costmap(cell_df, inflation = Config.Terrain.COSTMAP_INFLATION):
    """Costmap 생성"""
    gx = cell_df['grid_x']
    gz = cell_df['grid_z']

    min_x, min_z = gx.min(), gz.min()
    max_x, max_z = gx.max(), gz.max()

    W = max_x - min_x + 1
    H = max_z - min_z + 1

    # float costmap (기본값 0.2)
    costmap = np.full((H, W), UNKNOWN_COST, dtype=np.float32)
    
    obstacle_count = 0

    for _, r in cell_df.iterrows():
        ix = int(r['grid_x'] - min_x)
        iz = int(r['grid_z'] - min_z)

        if r['is_obstacle']:
            costmap[iz, ix] = 1.0
            obstacle_count += 1
        else:
            cost = r['final_cost']
            if np.isnan(cost):
                cost = FILL_FINAL_COST
            costmap[iz, ix] = np.clip(cost, 0.0, 0.99)

    # obstacle inflation
    obstacle_mask = costmap >= 1.0
    inflated = binary_dilation(obstacle_mask, iterations=inflation)
    costmap[inflated & (~obstacle_mask)] = 0.99
    costmap[obstacle_mask] = 1.0
    
    origin = (min_x * GRID_SIZE, min_z * GRID_SIZE)
    
    print(f"📊 Costmap 통계: {len(cell_df)}개 셀, {obstacle_count}개 장애물, inflation={inflation}")

    return costmap, origin