"""
PPO Feature Extractor for core.zip trained models
원본 프로젝트의 MyFeatureExtractor와 관측 공간 상수 정의
"""

import math
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ========================================
# 관측 공간 상수 정의 (core.zip 방식)
# ========================================

SPEED_MAX_W = 19.44
SPD_MAX_STD = 10.0

# Tank Challenge의 맵 크기
MAP_W = 300
MAP_H = 300

# [설계 파라미터]
LOOKAHEAD_POINTS = 4  # 경로 추종을 위한 앞의 N개 점
LIDAR_NUM = 65
LIDAR_RANGE = 30
LIDAR_START = -math.pi
LIDAR_END = math.pi

# 상태: 플레이어 속도(방향빼고) + 경로 점들, 라이다 거리점수, 라이다 최근접 거리점수, 라이다 최근접 각도
OBSERVATION_IND_SPD = 0
OBSERVATION_IND_GOAL_0 = 0 + 1
OBSERVATION_IND_GOAL_1 = 0 + 1 + (1 * LOOKAHEAD_POINTS)
# NOTE: ppo.zip was trained with GOAL_2 = 11 (not 9) to include first 2 complete lookahead points
OBSERVATION_IND_GOAL_2 = 11  # ppo.zip expects: 1 speed + 10 goal features (2 full points × 5 features)
OBSERVATION_IND_SCALAR_S = 0
OBSERVATION_IND_SCALAR_E = 0 + 1 + (5 * LOOKAHEAD_POINTS)
OBSERVATION_DIM_SCALAR = OBSERVATION_IND_SCALAR_E - OBSERVATION_IND_SCALAR_S

OBSERVATION_IND_LIDAR_DIS_S = 0 + 1 + (5 * LOOKAHEAD_POINTS)
OBSERVATION_IND_LIDAR_DIS_E = 0 + 1 + (5 * LOOKAHEAD_POINTS) + LIDAR_NUM
OBSERVATION_DIM_LIDAR = LIDAR_NUM

OBSERVATION_DIM = 0 + 1 + (5 * LOOKAHEAD_POINTS) + LIDAR_NUM  # 1 + 20 + 65 = 86


# ========================================
# MyFeatureExtractor (core.zip 원본)
# ========================================

class MyFeatureExtractor(BaseFeaturesExtractor):
    """
    core.zip 프로젝트의 Feature Extractor

    관측 공간 구조 (86차원):
    - Speed: 1차원 (index 0)
    - Goal features: 5 features × 4 lookahead points = 20차원 (index 1-20)
    - LiDAR distances: 65차원 (index 21-85)

    Feature Extraction:
    - layer0: Scalar data (21차원) → 32차원 (Linear + ReLU)
    - layer1: LiDAR data (65차원) → 64차원 (Conv1D + MaxPool + Linear)
    - layer2: Combined → 128차원 (Linear + ReLU)
    - 총 출력: speed_and_goal1 (9차원) + 32 + 64 + 128 = 233차원
    """

    def __init__(self, observation_space: gym.spaces.Box):

        speed_and_goal1_dim = OBSERVATION_IND_GOAL_2 - OBSERVATION_IND_SPD
        cnn_channel_num = 2
        feature0_dim = 32
        feature1_dim = 64
        feature2_dim = 128
        total_feature_dim = speed_and_goal1_dim + feature0_dim + feature1_dim + feature2_dim

        super(MyFeatureExtractor, self).__init__(observation_space, features_dim=total_feature_dim)

        self.layer0 = nn.Sequential(
            nn.Linear(OBSERVATION_DIM_SCALAR, feature0_dim),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            # 노이즈 제거 및 로컬 특징 추출
            nn.Conv1d(in_channels=1, out_channels=cnn_channel_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=feature1_dim),
            nn.Flatten(),
            nn.Linear(cnn_channel_num * feature1_dim, feature1_dim)
        )

        self.layer2 = nn.Sequential(
            # 위치 정보 매핑 (Linear)
            nn.Linear(speed_and_goal1_dim + feature1_dim, feature2_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        speed_and_goal1 = observations[:, OBSERVATION_IND_SPD: OBSERVATION_IND_GOAL_2]
        scalar_data = observations[:, OBSERVATION_IND_SCALAR_S: OBSERVATION_IND_SCALAR_E]
        lidar_dis_data = observations[:, OBSERVATION_IND_LIDAR_DIS_S: OBSERVATION_IND_LIDAR_DIS_E]

        output0 = self.layer0(scalar_data)
        output1 = self.layer1(lidar_dis_data.unsqueeze(1))
        output2 = self.layer2(torch.cat((speed_and_goal1, output1), dim=1))

        return torch.cat((speed_and_goal1, output0, output1, output2), dim=1)


# ========================================
# 헬퍼 함수
# ========================================

def get_policy_kwargs_for_ppo_zip():
    """
    ppo.zip 모델을 로드할 때 필요한 policy_kwargs 반환
    """
    return dict(
        features_extractor_class=MyFeatureExtractor,
        features_extractor_kwargs=dict(),
    )
