"""所有增强算法的统一抽象（传统/深度/级联/第三方适配器都实现它）。"""
from abc import ABC, abstractmethod

import torch


class Enhancer(ABC):
    """约定：单声道、float32、16kHz、波形进波形出。

    任何算法内部使用频谱/相位/多通道细节，都不允许泄漏到接口外。
    """

    name: str = "base"

    @abstractmethod
    def enhance(self, noisy: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """noisy: (B, T) 取值 [-1, 1]；返回同形状的增强波形。"""

    def load(self, ckpt_path: str) -> None:
        """加载权重；传统算法无需实现。"""

    def num_params(self) -> int:
        return 0
