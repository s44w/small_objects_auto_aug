from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Configure reproducibility for Python, NumPy and (if available) PyTorch.

    This is intentionally lightweight for MVP and can be expanded later.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

