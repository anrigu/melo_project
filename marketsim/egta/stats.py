import math
import numpy as np
from typing import Sequence

MIN_SAMPLES = 10


def hoeffding_upper_bound(mean: float, n: int, alpha: float = .05) -> float:
    return math.sqrt(math.log(2 / alpha) / (2 * n))

def statistical_test(sample_a: Sequence[float],
                     sample_b: Sequence[float],
                     alpha: float = 0.05) -> bool:
    if len(sample_a) == 0 or len(sample_b) == 0:
        return False
    mx = max(max(sample_a), max(sample_b))
    mn = min(min(sample_a), min(sample_b))
    span = max(1e-12, mx - mn)
    diff = np.mean(sample_b) - np.mean(sample_a)
    bound = span * hoeffding_upper_bound(0.0,
                                         min(len(sample_a), len(sample_b)),
                                         alpha)
    
    
    return diff - bound > 0
