# 项目公用工具函数
# 供 recall、rough_rank、fine_rank 各模块共同使用，避免重复定义
import torch


def get_device():
    # 优先使用 GPU（CUDA），不可用时回退到 CPU
    # torch.cuda.is_available() 会检测当前机器是否有 NVIDIA GPU 且驱动正常
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
