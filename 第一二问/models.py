"""问题一、二共用的常速度卡尔曼状态模型。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cv_matrices(dt: float, acceleration_variance: float,
                observation_variance: float):
    """建立三维位置—速度状态的离散运动矩阵。

    状态为 [x, y, z, vx, vy, vz]。过程噪声来自训练记录的运动变化尺度；
    observation_variance 是记录残差尺度，不宣称为独立传感器噪声真值。
    """
    identity = np.eye(3)
    zero = np.zeros((3, 3))
    transition = np.block([[identity, dt * identity], [zero, identity]])
    acceleration = np.vstack((0.5 * dt * dt * identity, dt * identity))
    process_noise = max(acceleration_variance, 1e-10) * acceleration @ acceleration.T
    observation = np.hstack((identity, zero))
    measurement_noise = max(observation_variance, 1e-10) * identity
    return transition, process_noise, observation, measurement_noise


@dataclass
class CVKalman:
    """六维常速度卡尔曼滤波器，保存状态均值与协方差。"""

    state: np.ndarray
    covariance: np.ndarray
    accel_var: float
    measurement_var: float

    def predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """向前预测一个时间间隔，返回位置均值和位置协方差。"""
        transition, process_noise, _, _ = cv_matrices(
            dt, self.accel_var, self.measurement_var)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        return self.state[:3].copy(), self.covariance[:3, :3].copy()

    def update(self, observation: np.ndarray) -> None:
        """只在观测已实际返回后更新状态，并使用 Joseph 形式保持数值稳定。"""
        _, _, observation_matrix, measurement_noise = cv_matrices(
            0.0, self.accel_var, self.measurement_var)
        residual = np.asarray(observation, dtype=float) - observation_matrix @ self.state
        residual_covariance = observation_matrix @ self.covariance @ observation_matrix.T + measurement_noise
        gain = np.linalg.solve(residual_covariance,
                               observation_matrix @ self.covariance).T
        self.state = self.state + gain @ residual
        identity = np.eye(6)
        correction = identity - gain @ observation_matrix
        self.covariance = (correction @ self.covariance @ correction.T
                           + gain @ measurement_noise @ gain.T)
