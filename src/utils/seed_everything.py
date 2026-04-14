"""
Utility to seed all random number generators to ensure reproducibility
"""

import random
import numpy as np
import torch


def seed_everything(seed: int = 96, deterministic: bool = True) -> None:
      """
      Seed Python, numpy and Pytorch (both CPU and CUDA) for
      reprodubicility.
      """
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = deterministic
      
      # benchmark is the natural counterporart of .deterministic
      # this way we let cuDNN find the fastest algorithm when determinism
      # is not enforced
      torch.backends.cudnn.benchmark = not deterministic