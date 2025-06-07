import torch
from typing import Literal, Optional
from loguru import logger


def get_device(force_device: Optional[Literal["cpu", "mps", "cuda"]] = None) -> Literal["cpu", "mps", "cuda"]:
    if force_device:
        device = force_device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = "cpu"
    logger.info(f"Using {device}")

    return device

