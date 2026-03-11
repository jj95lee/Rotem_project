# 🚀 Tank Navigation with Reinforcement Learning

Unity 시뮬레이터 기반 전차 자율주행 시스템  
**A* 경로 계획 + RL 국소 제어** 하이브리드 방식

---

## 📁 프로젝트 구조

```
tank_rl/
├── rl_environment.py      # Gymnasium 환경 (학습용)
├── rl_controller.py       # RL 컨트롤러 (추론용)
├── train_rl.py            # 학습 스크립트
├── server_rl.py           # Flask 서버 (시뮬레이터 연동)
└── env_data/              # 학습 환경 데이터
  ├── ob_v2.json           # 장애물 데이터
  ├── height_map.npy       # 높이 맵
  └── slope_costmap.npy    # 경사도 맵
└── models/                # 학습된 모델 저장 위치
    └── tank_nav_final.zip
```

---

## 🧠 시스템 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Unity 시뮬레이터                          │
└─────────────────────┬───────────────────────────────────────┘
                      │ REST API (~5Hz)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Server (server_rl.py)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [A*플래너]  ──waypoints──►  [RL 컨트롤러] ──명령──►         │
│       │                            │                        │
│   전역 경로 생성              국소 제어 결정                  │
│   (장애물 회피)              (부드러운 조향)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### RL 에이전트 역할

| 구분 | A* 플래너 | RL 에이전트 |
|------|-----------|-------------|
| 역할 | 전역 경로 생성 | 국소 제어 |
| 입력 | 시작점, 목적지, 장애물 | 현재 상태, 타겟, 라이다 |
| 출력 | waypoint 리스트 | W/A/D/STOP 명령 |
| 특징 | 최적 경로 보장 | 부드러운 움직임, 상황 적응 |

---

## 📊 Observation Space (20차원)

```python
observation = [
    heading_error / 180,      # 타겟 방향 오차 (-1 ~ 1)
    target_distance / 50,     # 타겟까지 거리 (0 ~ 1)
    current_speed / max_speed,# 현재 속도 (0 ~ 1)
    goal_distance / 300,      # 목표까지 거리 (0 ~ 1)
    lidar_ray_0 / 30,         # 라이다 0° (0 ~ 1)
    lidar_ray_1 / 30,         # 라이다 22.5°
    ...                       # 총 16개 방향
    lidar_ray_15 / 30,        # 라이다 337.5°
]
```

---

## 🎯 Action Space (6개 이산 행동)

| Action | 설명 | moveWS | moveAD |
|--------|------|--------|--------|
| 0 | 직진 (빠름) | W, 0.5 | - |
| 1 | 전진 + 좌회전 | W, 0.35 | A, 0.5 |
| 2 | 전진 + 우회전 | W, 0.35 | D, 0.5 |
| 3 | 제자리 좌회전 | STOP | A, 0.8 |
| 4 | 제자리 우회전 | STOP | D, 0.8 |
| 5 | 정지 | STOP | - |

---

## 🏆 Reward Function

```python
def calculate_reward():
    reward = 0
    
    # 1. 목표 도달 (+1000)
    if reached_goal:
        return 1000
    
    # 2. 충돌 (-500)
    if collision:
        return -500
    
    # 3. 목표 접근 (+10 per meter)
    reward += distance_delta * 10
    
    # 4. 시간 패널티 (-0.1 per step)
    reward -= 0.1
    
    # 5. 방향 정렬 보너스 (+0.5)
    if abs(heading_error) < 15:
        reward += 0.5
    
    # 6. 속도 보너스 (+0.3)
    if speed > 1.0:
        reward += 0.3
    
    return reward
```

---

## ⚙️ 하이퍼파라미터

### 학습 설정 (train_rl.py)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--timesteps` | 500,000 | 총 학습 스텝 |
| `--n-envs` | 4 | 병렬 환경 수 |
| `--lr` | 3e-4 | 학습률 |
| `--batch-size` | 64 | 배치 크기 |
| `--gamma` | 0.99 | 할인율 |

### 환경 설정 (rl_environment.py)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `goal_threshold` | 8.0m | 목표 도달 판정 거리 |
| `max_episode_steps` | 1500 | 최대 에피소드 길이 (300초) |
| `lidar_num_rays` | 16 | 가상 라이다 레이 수 |
| `lidar_max_range` | 30m | 라이다 최대 거리 |

---

## 📚 참고

- [Stable-Baselines3 문서](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 문서](https://gymnasium.farama.org/)
- [PPO 논문](https://arxiv.org/abs/1707.06347)
