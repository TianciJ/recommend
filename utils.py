# 项目公用工具函数
# 供 recall、rough_rank、fine_rank 各模块共同使用，避免重复定义
import torch
from time import perf_counter


def get_device():
    # 优先使用 GPU（CUDA），不可用时回退到 CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def elapsed_ms(start_time):
    # 返回从 start_time 到现在经过的毫秒数
    return (perf_counter() - start_time) * 1000


def move_batch_to_device(batch, device):
    # 把 batch 中所有 tensor 移到指定 device，非 tensor 字段保持原样
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_checkpoint(model_path, device):
    # weights_only=False 兼容含 feature_info 字典的 checkpoint 格式
    return torch.load(model_path, map_location=device, weights_only=False)
