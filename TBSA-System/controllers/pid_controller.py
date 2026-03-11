"""
PID 제어기 - 비례-적분-미분 제어

[개요]
PID(Proportional-Integral-Derivative) 제어기는 오차를 기반으로
제어 출력을 계산하는 피드백 제어 알고리즘입니다.

[제어 요소]
- P (비례): 현재 오차에 비례하는 제어
- I (적분): 누적 오차를 보정 (정상 상태 오차 제거)
- D (미분): 오차 변화율 억제 (오버슈트 방지)

[사용처]
- hybrid_controller.py의 조향 제어
- 목표 방향과 현재 방향의 차이를 보정

[제어 공식]
output = Kp * error + Ki * ∫error*dt + Kd * d(error)/dt
"""

import time


class PIDController:
    """
    PID 제어기 클래스
    
    Attributes:
        kp: 비례 게인 (Proportional Gain)
        ki: 적분 게인 (Integral Gain)
        kd: 미분 게인 (Derivative Gain)
        prev_error: 이전 시점 오차
        integral: 누적 오차 (적분 항)
        last_time: 마지막 계산 시각
    """
    
    def __init__(self, kp: float, ki: float, kd: float):
        """
        PID 제어기 초기화
        
        Args:
            kp: 비례 게인 (일반적으로 0.01 ~ 0.1)
            ki: 적분 게인 (일반적으로 0.0001 ~ 0.001)
            kd: 미분 게인 (일반적으로 0.01 ~ 0.1)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # 제어기 상태 변수
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
    
    def reset(self):
        """
        제어기 상태 초기화
        
        목적지 변경 시 또는 제어 재시작 시 호출
        누적된 오차와 이전 값을 모두 리셋
        """
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
    
    def compute(self, error: float) -> float:
        """
        PID 제어 출력 계산
        
        Args:
            error: 현재 오차 (목표값 - 현재값)
                  예: 각도 차이, 거리 차이 등
        
        Returns:
            float: 제어 출력 (-1.0 ~ 1.0 범위로 클램핑)
        
        Note:
            - dt(시간 간격)는 자동으로 계산됨
            - 적분 항은 wind-up 방지를 위해 ±10.0으로 제한
            - 최종 출력은 ±1.0으로 제한
        """
        current_time = time.time()
        
        # 시간 간격(dt) 계산
        if self.last_time is None:
            dt = 0.1  # 초기값
        else:
            dt = current_time - self.last_time
            
            # dt 유효성 검사
            if dt <= 0.0:
                dt = 0.001  # 최소값
            if dt > 1.0:
                dt = 0.1  # 최대값 (비정상적인 지연 방지)
        
        # P항: 비례 제어 (현재 오차에 비례)
        p_term = self.kp * error
        
        # I항: 적분 제어 (누적 오차 보정)
        self.integral += error * dt
        # Integral wind-up 방지 (적분 항이 무한정 커지는 것 방지)
        self.integral = max(min(self.integral, 10.0), -10.0)
        i_term = self.ki * self.integral
        
        # D항: 미분 제어 (오차 변화율 억제)
        d_term = self.kd * (error - self.prev_error) / dt
        
        # 다음 계산을 위한 상태 업데이트
        self.prev_error = error
        self.last_time = current_time
        
        # 최종 출력 계산 및 클램핑
        output = p_term + i_term + d_term
        output = max(min(output, 1.0), -1.0)
        
        return output