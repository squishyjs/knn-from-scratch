import time
import numpy as np
import random
from typing import Optional

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

class Timer:
    def __init__(self):
        self.start_time: Optional[float] = None
    def __enter__(self):
        self.start_time = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self.start_time
