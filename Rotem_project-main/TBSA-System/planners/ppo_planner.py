"""
통합 PPO 플래너 - 여러 모델 형식 지원

지원 모델:
1. models/ppo_models/cnn/withobs_model/ (디렉토리 형식, 완성도 높은 모델)
2. models/ppo.zip (ZIP 파일 형식)
3. *.zip 파일 (일반 ZIP 모델)

우선순위:
1. withobs_model (가장 완성도 높음)
2. ppo.zip (백업 모델)
3. Potential Field (폴백)
"""
import numpy as np
import math
import os
from typing import Dict, Optional, Tuple, List


class UnifiedPPOPlanner:
    """
    통합 PPO 플래너

    Stable-Baselines3 PPO 모델을 다양한 형식으로 로드 지원
    - 디렉토리 형식 (withobs_model)
    - ZIP 파일 형식 (ppo.zip)
    """

    def __init__(self, config, state_manager):
        """
        Args:
            config: 시스템 설정 객체
            state_manager: 상태 관리자
        """
        self.config = config
        self.state = state_manager
        self.model = None
        self.model_loaded = False
        self.model_source = None  # "withobs_model", "ppo_zip", 또는 None

        # 통계
        self.call_count = 0
        self.success_count = 0
        self.fail_count = 0

        # LiDAR 설정
        self.lidar_num_rays = 32
        self.lidar_max_range = 50.0

        # 관측/행동 공간 크기
        self.expected_obs_dim = 35  # LiDAR(32) + Goal(2) + Velocity(1)
        self.expected_act_dim = 2   # [steering, speed]

        # 모델 로드
        self._load_best_model()

        # 🆕 Warm-up: 첫 추론 지연 제거
        if self.model_loaded:
            self._warmup_model()

    def _warmup_model(self):
        """모델 warm-up: 더미 데이터로 첫 추론 실행하여 GPU 초기화"""
        try:
            print(f"⏳ [Unified PPO] 모델 warm-up 중... (GPU 초기화)")
            import time
            start = time.time()

            # 더미 관측값 생성 (86차원)
            dummy_obs = np.zeros(self.expected_obs_dim, dtype=np.float32)

            # 첫 추론 실행 (GPU 초기화)
            _, _ = self.model.predict(dummy_obs, deterministic=True)

            elapsed = time.time() - start
            print(f"✅ [Unified PPO] Warm-up 완료 ({elapsed:.1f}초) - 이제 즉시 반응 가능!")
        except Exception as e:
            print(f"⚠️ [Unified PPO] Warm-up 실패 (무시): {e}")

    def _load_best_model(self):
        """모든 사용 가능한 PPO 모델을 우선순위대로 로드 시도"""
        base_dir = os.path.dirname(os.path.dirname(__file__))

        # 시도할 모델 목록 (우선순위 순서)
        models_to_try = [
            ("ppo.zip [core]", os.path.join(base_dir, "models", "ppo.zip")),  # core.zip 학습 모델 - 최우선
            ("3_withobs_2.zip", os.path.join(base_dir, "models", "ppo_models", "cnn", "3_withobs_2.zip")),
            ("3_withobs.zip", os.path.join(base_dir, "models", "ppo_models", "cnn", "3_withobs.zip")),
            ("1_naive.zip", os.path.join(base_dir, "models", "ppo_models", "cnn", "1_naive.zip")),
            ("2_plain.zip", os.path.join(base_dir, "models", "ppo_models", "cnn", "2_plain.zip")),
            ("2.plain.zip", os.path.join(base_dir, "models", "ppo_models", "2.plain.zip")),
            ("0____.zip", os.path.join(base_dir, "models", "ppo_models", "cnn", "0____.zip")),
            ("best.zip", os.path.join(base_dir, "best.zip")),  # YOLO - 마지막
        ]

        print(f"🔍 [Unified PPO] 사용 가능한 모델 검색 중...")

        for model_name, model_path in models_to_try:
            if os.path.isfile(model_path):
                print(f"🎓 [Unified PPO] {model_name} 로드 시도: {model_path}")
                if self._load_zip_model(model_path):
                    obs_dim = self.model.observation_space.shape[0]
                    print(f"✅ {model_name} 로드 성공! 관측 공간: {obs_dim}차원")
                    self.expected_obs_dim = obs_dim
                    self.model_source = model_name.replace(".zip", "")

                    if obs_dim == 35:
                        print(f"   📊 35차원 구조: LiDAR(32) + Goal(2) + Velocity(1)")
                    elif obs_dim == 86:
                        print(f"   📊 86차원 구조: LiDAR(80) + Goal(2) + Vel(1) + Yaw(2) + Dist(1)")

                    return
                else:
                    print(f"   ❌ {model_name} 로드 실패")

        print(f"⚠️ [Unified PPO] 호환되는 PPO 모델을 찾을 수 없습니다.")
        print(f"   시도한 모델: {[name for name, _ in models_to_try]}")

    def _load_directory_model(self, model_path: str) -> bool:
        """
        디렉토리 형식 모델 로드

        Args:
            model_path: 모델 디렉토리 경로

        Returns:
            bool: 로드 성공 여부
        """
        try:
            from stable_baselines3 import PPO

            # 필수 파일 확인
            policy_path = os.path.join(model_path, "policy.pth")
            data_path = os.path.join(model_path, "data")

            if not os.path.exists(policy_path):
                print(f"⚠️ [Unified PPO] 필수 파일 없음: policy.pth")
                return False
            if not os.path.exists(data_path):
                print(f"⚠️ [Unified PPO] 필수 파일 없음: data")
                return False

            # ✨ PPO 모델 로드 개선 - .zip으로 압축된 디렉토리로 변환
            # SB3는 디렉토리를 직접 로드할 수 없으므로 ZIP 방식 사용
            import zipfile
            import tempfile

            # 임시 ZIP 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                tmp_zip_path = tmp_zip.name

            try:
                # 디렉토리를 ZIP으로 압축
                with zipfile.ZipFile(tmp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(model_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, model_path)
                            zf.write(file_path, arc_name)

                # ZIP 파일에서 로드
                self.model = self._load_with_bypass(tmp_zip_path)

            finally:
                # 임시 파일 삭제
                try:
                    os.unlink(tmp_zip_path)
                except:
                    pass

            if self.model is None:
                return False

            self.model_loaded = True

            # 모델 정보 출력
            obs_space = self.model.observation_space
            act_space = self.model.action_space

            print(f"✅ [Unified PPO] withobs_model 로드 성공!")
            print(f"   관측 공간: {obs_space.shape}")
            print(f"   행동 공간: {act_space.shape}")

            # 차원 검증
            if obs_space.shape[0] != self.expected_obs_dim:
                print(f"⚠️ [Unified PPO] 관측 공간 차원 불일치!")
                print(f"   기대: {self.expected_obs_dim}, 실제: {obs_space.shape[0]}")

            if act_space.shape[0] != self.expected_act_dim:
                print(f"⚠️ [Unified PPO] 행동 공간 차원 불일치!")
                print(f"   기대: {self.expected_act_dim}, 실제: {act_space.shape[0]}")

            return True

        except ImportError:
            print(f"❌ [Unified PPO] stable-baselines3 미설치")
            print(f"   설치: pip install stable-baselines3")
            return False
        except Exception as e:
            print(f"❌ [Unified PPO] 디렉토리 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_zip_model(self, model_path: str) -> bool:
        """
        ZIP 파일 형식 모델 로드

        Args:
            model_path: ZIP 파일 경로

        Returns:
            bool: 로드 성공 여부
        """
        try:
            import time
            from stable_baselines3 import PPO

            # ZIP 모델 로드 (drivingppo 우회)
            print(f"   ⏳ PyTorch 모델 로딩 중... (30초~1분 소요될 수 있음)")
            start_time = time.time()
            self.model = self._load_with_bypass(model_path)
            elapsed = time.time() - start_time
            print(f"   ✅ 모델 로딩 완료 ({elapsed:.1f}초 소요)")

            if self.model is None:
                return False

            self.model_loaded = True

            # 모델 정보
            obs_space = self.model.observation_space
            act_space = self.model.action_space

            print(f"✅ [Unified PPO] ppo.zip 로드 성공!")
            print(f"   관측 공간: {obs_space.shape}")
            print(f"   행동 공간: {act_space.shape}")

            return True

        except ImportError:
            print(f"❌ [Unified PPO] stable-baselines3 미설치")
            return False
        except Exception as e:
            print(f"❌ [Unified PPO] ZIP 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_with_bypass(self, model_path: str):
        """
        drivingppo 모듈 의존성을 우회하여 모델 로드
        MyFeatureExtractor 지원 추가

        Args:
            model_path: 모델 파일/디렉토리 경로

        Returns:
            PPO 모델 또는 None
        """
        try:
            from stable_baselines3 import PPO

            # 일반 로드 시도
            try:
                return PPO.load(model_path)
            except ModuleNotFoundError as e:
                error_msg = str(e)
                if 'drivingppo' not in error_msg:
                    raise
            except AttributeError as e:
                # MyFeatureExtractor 관련 에러
                error_msg = str(e)
                if 'MyFeatureExtractor' in error_msg or 'FakeCNN' in error_msg:
                    print(f"   MyFeatureExtractor 필요 - custom_objects로 재시도...")
                else:
                    raise

            # drivingppo 모듈 + MyFeatureExtractor 우회
            print(f"   drivingppo 의존성 우회 중...")

            import sys
            from types import ModuleType

            # gymnasium 또는 gym 사용
            try:
                import gymnasium as gym
                from gymnasium import Env
                from gymnasium.spaces import Box
            except ImportError:
                import gymnasium as gym
                from gymnasium import Env
                from gymnasium.spaces import Box

            # MyFeatureExtractor import
            try:
                from drivingppo.ppo_feature_extractor import MyFeatureExtractor, OBSERVATION_DIM
                print(f"   ✅ MyFeatureExtractor 로드 성공 (관측 공간: {OBSERVATION_DIM}차원)")
            except ImportError as e:
                print(f"   ⚠️ MyFeatureExtractor import 실패: {e}")
                MyFeatureExtractor = None
                OBSERVATION_DIM = 86

            # Dummy 환경 클래스
            class DummyDrivingEnv(Env):
                def __init__(self):
                    super().__init__()
                    self.observation_space = Box(
                        low=-np.inf, high=np.inf,
                        shape=(OBSERVATION_DIM,),  # core.zip 방식
                        dtype=np.float32
                    )
                    self.action_space = Box(
                        low=-1, high=1,
                        shape=(2,),
                        dtype=np.float32
                    )

                def reset(self, seed=None, options=None):
                    if hasattr(super(), 'reset'):
                        super().reset(seed=seed)
                    obs = self.observation_space.sample()
                    return obs, {}

                def step(self, action):
                    obs = self.observation_space.sample()
                    return obs, 0.0, False, False, {}

            # drivingppo 더미 모듈 생성 (서브모듈 포함)
            dummy_main = ModuleType('drivingppo')
            dummy_main.DrivingEnv = DummyDrivingEnv
            if MyFeatureExtractor:
                dummy_main.MyFeatureExtractor = MyFeatureExtractor

            # drivingppo.model 서브모듈 생성
            dummy_model = ModuleType('drivingppo.model')
            dummy_model.DrivingEnv = DummyDrivingEnv
            if MyFeatureExtractor:
                dummy_model.MyFeatureExtractor = MyFeatureExtractor

            # sys.modules에 등록
            sys.modules['drivingppo'] = dummy_main
            sys.modules['drivingppo.model'] = dummy_model

            # custom_objects로 로드
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.2,
            }

            # MyFeatureExtractor 추가
            if MyFeatureExtractor:
                custom_objects["MyFeatureExtractor"] = MyFeatureExtractor

            try:
                model = PPO.load(model_path, custom_objects=custom_objects)
                print(f"   ✅ 모델 로드 성공!")
                if MyFeatureExtractor:
                    print(f"   ✅ MyFeatureExtractor 적용됨 (core.zip 방식)")
            finally:
                # 더미 모듈 제거
                if 'drivingppo' in sys.modules:
                    del sys.modules['drivingppo']
                if 'drivingppo.model' in sys.modules:
                    del sys.modules['drivingppo.model']

            return model

        except Exception as e:
            print(f"❌ [Unified PPO] 우회 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def is_available(self) -> bool:
        """PPO 모델 사용 가능 여부"""
        return self.model_loaded

    def get_action(self,
                   lidar_data: List[float],
                   curr_x: float,
                   curr_z: float,
                   curr_yaw: float,
                   goal_x: float,
                   goal_z: float,
                   curr_velocity: float = 0.0) -> Optional[Dict[str, float]]:
        """
        PPO 모델로 행동 결정

        Args:
            lidar_data: LiDAR 스캔 데이터 (거리 리스트)
            curr_x, curr_z: 현재 위치
            curr_yaw: 현재 방향 (도)
            goal_x, goal_z: 목표 위치
            curr_velocity: 현재 속도

        Returns:
            {"steering": float, "speed": float} 또는 None
        """
        if not self.model_loaded:
            self.fail_count += 1
            if self.call_count % 50 == 0:
                print(f"⚠️ [Unified PPO] 모델 로드 안됨 (실패: {self.fail_count}회)")
            return None

        try:
            self.call_count += 1

            # 관측값 생성
            observation = self._build_observation(
                lidar_data, curr_x, curr_z, curr_yaw,
                goal_x, goal_z, curr_velocity
            )

            if observation is None:
                self.fail_count += 1
                return None

            # PPO 추론
            action, _ = self.model.predict(observation, deterministic=True)

            # 행동 파싱
            if len(action) >= 2:
                steering = float(np.clip(action[0], -1.0, 1.0))
                speed = float(np.clip(action[1], 0.0, 1.0))
            else:
                steering = float(np.clip(action[0], -1.0, 1.0))
                speed = 0.8  # 고정 속도

            self.success_count += 1

            # 주기적 로그
            if self.call_count % 20 == 1:
                success_rate = (self.success_count / self.call_count) * 100
                print(f"🎓 [Unified PPO #{self.call_count}] "
                      f"모델={self.model_source}, 성공률={success_rate:.1f}%")
                print(f"   위치: ({curr_x:.1f}, {curr_z:.1f}) → 목표: ({goal_x:.1f}, {goal_z:.1f})")
                print(f"   → 조향={steering:.3f}, 속도={speed:.3f}")
                if speed < 0.1:
                    print(f"   ⚠️ 속도가 매우 낮음! action raw: {action}")

            return {
                "steering": steering,
                "speed": speed
            }

        except Exception as e:
            self.fail_count += 1
            # 첫 10번은 항상 출력, 이후는 20번마다
            if self.call_count <= 10 or self.call_count % 20 == 1:
                print(f"❌ [Unified PPO] 추론 실패 (호출 #{self.call_count}): {e}")
                if self.call_count <= 3:
                    import traceback
                    traceback.print_exc()
            return None

    def _build_observation(self,
                           lidar_data: List[float],
                           curr_x: float,
                           curr_z: float,
                           curr_yaw: float,
                           goal_x: float,
                           goal_z: float,
                           curr_velocity: float) -> Optional[np.ndarray]:
        """
        관측값 생성: 모델 차원에 맞춤 (35차원 또는 86차원)

        Returns:
            np.ndarray 또는 None
        """
        if self.expected_obs_dim is None:
            print(f"⚠️ [Unified PPO] 모델이 로드되지 않음")
            return None

        try:
            if self.expected_obs_dim == 86:
                # core.zip 방식 (65 LiDAR + 4 lookahead goals)
                if self.model_source and "core" in self.model_source:
                    return self._build_observation_86d_core(
                        lidar_data, curr_x, curr_z, curr_yaw,
                        goal_x, goal_z, curr_velocity
                    )
                else:
                    # 기존 86차원 방식
                    return self._build_observation_86d(
                        lidar_data, curr_x, curr_z, curr_yaw,
                        goal_x, goal_z, curr_velocity
                    )
            elif self.expected_obs_dim == 35:
                return self._build_observation_35d(
                    lidar_data, curr_x, curr_z, curr_yaw,
                    goal_x, goal_z, curr_velocity
                )
            else:
                print(f"⚠️ [Unified PPO] 지원하지 않는 관측 공간: {self.expected_obs_dim}차원")
                return None

        except Exception as e:
            print(f"⚠️ [Unified PPO] 관측값 생성 실패: {e}")
            return None

    def _build_observation_35d(self,
                               lidar_data: List[float],
                               curr_x: float,
                               curr_z: float,
                               curr_yaw: float,
                               goal_x: float,
                               goal_z: float,
                               curr_velocity: float) -> Optional[np.ndarray]:
        """35차원 관측값 생성: LiDAR(32) + Goal(2) + Velocity(1)"""
        obs_list = []

        # 1. LiDAR (32개)
        if lidar_data and len(lidar_data) > 0:
            if len(lidar_data) != 32:
                indices = np.linspace(0, len(lidar_data) - 1, 32)
                lidar_resampled = [lidar_data[int(i)] for i in indices]
            else:
                lidar_resampled = lidar_data
            lidar_normalized = [min(d / self.lidar_max_range, 1.0) for d in lidar_resampled]
        else:
            lidar_normalized = [1.0] * 32

        obs_list.extend(lidar_normalized)

        # 2. Goal (2개)
        dx = goal_x - curr_x
        dz = goal_z - curr_z
        goal_dx_norm = np.clip(dx / 100.0, -1.0, 1.0)
        goal_dz_norm = np.clip(dz / 100.0, -1.0, 1.0)
        obs_list.append(goal_dx_norm)
        obs_list.append(goal_dz_norm)

        # 3. Velocity (1개)
        vel = float(curr_velocity) if not isinstance(curr_velocity, (list, np.ndarray)) else float(curr_velocity[0])
        vel_norm = np.clip(vel / 5.0, 0.0, 1.0)
        obs_list.append(vel_norm)

        observation = np.array(obs_list, dtype=np.float32)

        if observation.shape[0] != 35:
            print(f"⚠️ [35d] 차원 오류: {observation.shape}")
            return None

        return observation

    def _build_observation_86d(self,
                               lidar_data: List[float],
                               curr_x: float,
                               curr_z: float,
                               curr_yaw: float,
                               goal_x: float,
                               goal_z: float,
                               curr_velocity: float) -> Optional[np.ndarray]:
        """
        86차원 관측값 생성

        구조 추정:
        - LiDAR: 80개 방향 (더 세밀한 스캔)
        - Goal: 2개 (dx, dz)
        - Velocity: 1개
        - Yaw (sin, cos): 2개
        - Distance to goal: 1개
        총 86차원
        """
        obs_list = []

        # 1. LiDAR (80개 - 더 세밀한 스캔)
        if lidar_data and len(lidar_data) > 0:
            if len(lidar_data) != 80:
                # 32개를 80개로 보간
                indices = np.linspace(0, len(lidar_data) - 1, 80)
                lidar_resampled = [lidar_data[int(i)] for i in indices]
            else:
                lidar_resampled = lidar_data

            # 정규화 (0~1)
            lidar_normalized = [min(d / self.lidar_max_range, 1.0) for d in lidar_resampled]
        else:
            lidar_normalized = [1.0] * 80

        obs_list.extend(lidar_normalized)

        # 2. Goal 상대 위치 (2개)
        dx = goal_x - curr_x
        dz = goal_z - curr_z
        goal_dist = math.hypot(dx, dz)

        goal_dx_norm = np.clip(dx / 100.0, -1.0, 1.0)
        goal_dz_norm = np.clip(dz / 100.0, -1.0, 1.0)
        obs_list.append(goal_dx_norm)
        obs_list.append(goal_dz_norm)

        # 3. Velocity (1개)
        vel = float(curr_velocity) if not isinstance(curr_velocity, (list, np.ndarray)) else float(curr_velocity[0])
        vel_norm = np.clip(vel / 5.0, 0.0, 1.0)
        obs_list.append(vel_norm)

        # 4. Yaw (sin, cos) (2개)
        curr_yaw_rad = math.radians(curr_yaw)
        obs_list.append(math.sin(curr_yaw_rad))
        obs_list.append(math.cos(curr_yaw_rad))

        # 5. Distance to goal 정규화 (1개)
        dist_norm = np.clip(goal_dist / 200.0, 0.0, 1.0)
        obs_list.append(dist_norm)

        observation = np.array(obs_list, dtype=np.float32)

        # 검증
        if observation.shape[0] != 86:
            print(f"⚠️ [86d] 차원 오류: {observation.shape} (목표: 86)")
            print(f"   LiDAR: 80, Goal: 2, Vel: 1, Yaw: 2, Dist: 1")
            return None

        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"⚠️ [86d] NaN/Inf 포함")
            return None

        return observation

    def _build_observation_86d_core(self,
                                    lidar_data: List[float],
                                    curr_x: float,
                                    curr_z: float,
                                    curr_yaw: float,
                                    goal_x: float,
                                    goal_z: float,
                                    curr_velocity: float) -> Optional[np.ndarray]:
        """
        86차원 관측값 생성 (core.zip 정확한 방식)

        구조:
        - Speed: 1차원
        - Goal features (5 features × 4 lookahead points): 20차원
          - Feature 0: a_fp_norm (이전 점 기준 각도)
          - Feature 1: a_fa_norm (에이전트 기준 각도)
          - Feature 2: cos(a_fa_norm)
          - Feature 3: distance_score_near
          - Feature 4: distance_score_far
        - LiDAR: 65차원
        총 86차원
        """
        try:
            from drivingppo.ppo_feature_extractor import (
                LOOKAHEAD_POINTS, LIDAR_NUM, LIDAR_RANGE,
                SPEED_MAX_W, SPD_MAX_STD
            )
        except ImportError:
            LOOKAHEAD_POINTS = 4
            LIDAR_NUM = 65
            LIDAR_RANGE = 30
            SPEED_MAX_W = 19.44
            SPD_MAX_STD = 10.0

        # Distance score 함수들 (core.zip 정확한 구현)
        def _distance_score_near(x: float) -> float:
            d = x + 10.0
            x_val = 100.0 / (d * d)
            return min(x_val, 1.0)

        distance_score_near_base = _distance_score_near(LIDAR_RANGE)

        def distance_score_near(x: float) -> float:
            return max(0.0, _distance_score_near(x) - distance_score_near_base)

        def distance_score_far(distance: float) -> float:
            return math.log(distance + 1.0) / 10.0

        obs_list = []

        # 1. Speed (1차원)
        vel = float(curr_velocity) if not isinstance(curr_velocity, (list, np.ndarray)) else float(curr_velocity[0])
        vel_normalized = min(vel / SPD_MAX_STD, 1.0)
        obs_list.append(vel_normalized)

        # 디버깅: 첫 번째 호출 시에만 로그 출력
        debug_log = not hasattr(self, '_obs_debug_logged')
        if debug_log:
            self._obs_debug_logged = True
            print(f"\n🔍 [ppo.zip 관측값 디버깅]")
            print(f"   현재 위치: ({curr_x:.2f}, {curr_z:.2f}), 방향: {curr_yaw:.2f}°")
            print(f"   목표 위치: ({goal_x:.2f}, {goal_z:.2f})")
            print(f"   현재 속도: {vel:.2f} → 정규화: {vel_normalized:.4f}")

        # 2. Goal features (5 features × 4 lookahead points = 20차원)
        # 직선 경로를 4개 점으로 나눔
        dx_total = goal_x - curr_x
        dz_total = goal_z - curr_z

        # 이전 점의 좌표와 각도 (초기값은 현재 위치)
        x_prev = curr_x
        z_prev = curr_z
        angle_prev = math.radians(curr_yaw)

        for i in range(1, LOOKAHEAD_POINTS + 1):
            # i번째 lookahead point (균등 분할)
            ratio = i / (LOOKAHEAD_POINTS + 1)
            x_point = curr_x + dx_total * ratio
            z_point = curr_z + dz_total * ratio

            # 이전 점으로부터의 거리와 각도
            d_from_prev = math.hypot(x_point - x_prev, z_point - z_prev)
            angle_to_point = math.atan2(z_point - z_prev, x_point - x_prev)

            # 에이전트(현재 위치)로부터의 각도
            angle_from_agent = math.atan2(z_point - curr_z, x_point - curr_x)

            # 각도 정규화 (-π ~ π → -1 ~ 1)
            pi = math.pi
            pi2 = 2 * math.pi
            a_from_prev = angle_to_point - angle_prev
            a_from_agnt = angle_from_agent - math.radians(curr_yaw)

            a_fp_norm = ((a_from_prev + pi) % pi2 - pi) / pi
            a_fa_norm = ((a_from_agnt + pi) % pi2 - pi) / pi
            cos_a_fa = math.cos(a_fa_norm)  # core.zip 원본대로
            d_near = distance_score_near(d_from_prev)
            d_far = distance_score_far(d_from_prev)

            # 5개 특징 추가 (core.zip 정확한 순서)
            obs_list.extend([a_fp_norm, a_fa_norm, cos_a_fa, d_near, d_far])

            # 디버깅: 첫 번째 lookahead point 정보
            if debug_log and i == 1:
                print(f"   Goal point 1: ({x_point:.2f}, {z_point:.2f})")
                print(f"   - a_fp_norm: {a_fp_norm:.4f}, a_fa_norm: {a_fa_norm:.4f}")
                print(f"   - cos(a_fa): {cos_a_fa:.4f}, d_near: {d_near:.4f}, d_far: {d_far:.4f}")

            # 다음 반복을 위해 이전 값 업데이트
            x_prev = x_point
            z_prev = z_point
            angle_prev = angle_to_point

        # 3. LiDAR (65개) - 전방 집중 필터링
        if lidar_data and len(lidar_data) > 0:
            if len(lidar_data) != LIDAR_NUM:
                # 리샘플링
                indices = np.linspace(0, len(lidar_data) - 1, LIDAR_NUM)
                lidar_resampled = [lidar_data[int(i)] for i in indices]
            else:
                lidar_resampled = lidar_data

            # 전방 ±90도 필터링 (장애물 회피에 집중)
            # LiDAR는 360도를 65개로 균등 분할
            # 전방 = -90도 ~ +90도 (270도 ~ 90도 범위)
            lidar_filtered = []
            for i, distance in enumerate(lidar_resampled):
                # 각도 계산 (0도 = 전방)
                angle_deg = (i / LIDAR_NUM) * 360.0

                # 전방 ±90도 범위인지 확인
                # 270도 ~ 360도 또는 0도 ~ 90도
                is_front = (angle_deg >= 270.0) or (angle_deg <= 90.0)

                if is_front:
                    # 전방: 실제 거리 사용
                    lidar_filtered.append(distance)
                else:
                    # 후방/측후방: MAX 거리로 설정 (장애물 없음)
                    lidar_filtered.append(LIDAR_RANGE)

            # distance_score_near 적용 (core.zip 방식)
            lidar_normalized = [distance_score_near(d) for d in lidar_filtered]
        else:
            lidar_normalized = [0.0] * LIDAR_NUM

        obs_list.extend(lidar_normalized)

        # 디버깅: LiDAR 통계
        if debug_log:
            lidar_arr = np.array(lidar_normalized)
            print(f"   LiDAR (전방 ±90도 필터링): min={lidar_arr.min():.4f}, max={lidar_arr.max():.4f}, mean={lidar_arr.mean():.4f}")
            # 전방 범위 카운트
            front_count = sum(1 for i in range(LIDAR_NUM) if ((i / LIDAR_NUM) * 360.0 >= 270.0) or ((i / LIDAR_NUM) * 360.0 <= 90.0))
            print(f"   전방 범위: {front_count}/{LIDAR_NUM}개 포인트 활성화 (후방은 MAX 거리로 설정)")

        observation = np.array(obs_list, dtype=np.float32)

        # 디버깅: 최종 관측값 통계
        if debug_log:
            print(f"   최종 observation: shape={observation.shape}, min={observation.min():.4f}, max={observation.max():.4f}")
            print(f"   처음 11개 값 (speed + goal features 1-2): {observation[:11]}")

        # 검증
        expected_dim = 1 + (5 * LOOKAHEAD_POINTS) + LIDAR_NUM  # 1 + 20 + 65 = 86
        if observation.shape[0] != expected_dim:
            print(f"⚠️ [86d-core] 차원 오류: {observation.shape} (목표: {expected_dim})")
            print(f"   Speed: 1, Goal: {5 * LOOKAHEAD_POINTS}, LiDAR: {LIDAR_NUM}")
            return None

        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"⚠️ [86d-core] NaN/Inf 포함")
            return None

        return observation

    def convert_to_command(self, action: Optional[Dict[str, float]]) -> Optional[Dict]:
        """
        행동을 탱크 제어 명령으로 변환

        Args:
            action: {"steering": float, "speed": float}

        Returns:
            {"moveWS": {...}, "moveAD": {...}, "fire": False}
        """
        if action is None:
            return None

        steering = action["steering"]
        speed = action["speed"]

        # 조향 명령
        if abs(steering) < 0.05:
            steer_dir = ""
            steer_weight = 0.0
        else:
            steer_dir = "D" if steering > 0 else "A"
            steer_weight = abs(steering)

        # 속도 명령
        if speed > 0.05:
            ws_cmd = "W"
            ws_weight = speed
        else:
            ws_cmd = "STOP"
            ws_weight = 0.0

        return {
            "moveWS": {"command": ws_cmd, "weight": round(ws_weight, 2)},
            "moveAD": {"command": steer_dir, "weight": round(steer_weight, 2)},
            "fire": False
        }

    def get_stats(self) -> Dict:
        """통계 반환"""
        return {
            "model_source": self.model_source,
            "model_loaded": self.model_loaded,
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "success_rate": (self.success_count / self.call_count * 100)
                if self.call_count > 0 else 0
        }


class UnifiedHybridPPOPlanner:
    """
    Unified PPO + Potential Field 하이브리드 플래너

    우선순위:
    1. Unified PPO (withobs_model 또는 ppo.zip)
    2. Potential Field (폴백)
    """

    def __init__(self, config, state_manager):
        self.config = config
        self.state = state_manager

        # Unified PPO 플래너
        self.ppo_planner = UnifiedPPOPlanner(config, state_manager)

        # Potential Field 폴백 (사용 안함 - 간소화)
        # from planners.working_rl_planner import WorkingRLPlanner
        # self.fallback_planner = WorkingRLPlanner(config, state_manager)
        self.fallback_planner = None

        # 모드 설정
        if self.ppo_planner.is_available():
            self.mode = "unified_ppo"
            model_src = self.ppo_planner.model_source
            print(f"🎓🚀 Unified PPO 모드 활성화! (모델: {model_src})")
        else:
            self.mode = "none"
            print(f"⚠️ Unified PPO 없음 - RL 플래너 사용 불가")

        self.call_count = 0
        self.fallback_count = 0

    def is_available(self) -> bool:
        """항상 사용 가능 (폴백 있음)"""
        return True

    def get_action(self,
                   lidar_data: List[float],
                   curr_x: float,
                   curr_z: float,
                   curr_yaw: float,
                   goal_x: float,
                   goal_z: float,
                   curr_velocity: float = 0.0) -> Optional[Dict[str, float]]:
        """행동 결정"""
        self.call_count += 1

        # Unified PPO 모드
        if self.mode == "unified_ppo":
            action = self.ppo_planner.get_action(
                lidar_data, curr_x, curr_z, curr_yaw,
                goal_x, goal_z, curr_velocity
            )

            if action is not None:
                return action

            # PPO 실패 → 폴백 없음 (None 반환)
            self.fallback_count += 1
            # 첫 10번은 항상 출력
            if self.fallback_count <= 10 or self.call_count % 20 == 1:
                print(f"⚠️ [Unified Hybrid] PPO 실패 #{self.fallback_count}")
                print(f"   PPO 모델 상태: loaded={self.ppo_planner.model_loaded}, source={self.ppo_planner.model_source}")
            return None

        # 모드가 "none"인 경우
        return None

    def convert_to_command(self, action: Optional[Dict[str, float]]) -> Optional[Dict]:
        """행동 → 명령 변환"""
        return self.ppo_planner.convert_to_command(action)

    def get_stats(self) -> Dict:
        """통합 통계 반환"""
        stats = {
            "mode": self.mode,
            "total_calls": self.call_count,
            "fallback_count": self.fallback_count,
            "fallback_rate": (self.fallback_count / self.call_count * 100)
                if self.call_count > 0 else 0
        }

        if self.mode == "unified_ppo":
            stats["ppo"] = self.ppo_planner.get_stats()

        return stats
