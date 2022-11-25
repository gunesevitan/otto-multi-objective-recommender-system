import os
import random
import numpy as np
import torch


def set_seed(seed, deterministic_cudnn=False):

    """
    Set random seed for reproducible results

    Parameters
    ----------
    seed: int
        Random seed

    deterministic_cudnn: bool
        Whether to set deterministic cuDNN or not
    """

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
