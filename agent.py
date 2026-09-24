import numpy as np


class Agent:
    def __init__(self):
        """모델과 가중치를 로드합니다. 초기화 제한 시간은 10초입니다."""
        pass

    def reset(self, observation):
        """트랙 시작 시 호출됩니다. 필요한 경우 이전 트랙의 내부 상태를 초기화하세요."""
        pass

    def act(self, observation) -> np.ndarray:
        """관측: float32 (4, 84, 84), 값 범위 [0, 1].
        반환: [steer, gas, brake], 범위 [-1, 1], [0, 1], [0, 1]."""

        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
