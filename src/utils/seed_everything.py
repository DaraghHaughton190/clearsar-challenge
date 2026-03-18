"""
Utility to seed all random number generators to ensure reproducibility
"""

import random
import numpy as np
import torch


def seed_everything(seed: int = 96) -> None:
      """
      Seed Python, numpu and Pytorch (both CPU and CUDA) for
      reprodubicility.
      """
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False