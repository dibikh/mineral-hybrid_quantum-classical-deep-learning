import os
import random
import numpy as np

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_torch_seed(seed: int = 42):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def set_tf_seed(seed: int = 42):
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def set_all_seeds(seed: int = 42):
    set_global_seed(seed)
    set_torch_seed(seed)
    set_tf_seed(seed)
