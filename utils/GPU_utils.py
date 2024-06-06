import os
import pynvml
import time
import gc

import torch


def get_gpu_metrics(handle) -> dict:
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        'timestamp': time.time(),
        'gpu_memory_total': mem_info.total,
        'gpu_memory_used': mem_info.used,
        'gpu_utilization': utilization.gpu,
        'memory_utilization': utilization.memory
    }


def clear_gpu_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def get_nvml_gpu_id(torch_gpu_id: int) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
        return ids[torch_gpu_id]
    else:
        return torch_gpu_id


def initialize_correct_pynvml_device():
    pynvml.nvmlInit()
    torch_gpu_id = torch.cuda.current_device()
    nvml_gpu_id = get_nvml_gpu_id(torch_gpu_id)
    return pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_id)
