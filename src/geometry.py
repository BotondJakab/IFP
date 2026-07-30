from dataclasses import dataclass
import numpy as np


@dataclass
class Plane:
    """
    Represents a plane in 3D.

    Plane equation:
        ax + by + cz + d = 0
    """

    a: float
    b: float
    c: float
    d: float

    point: np.ndarray
    normal: np.ndarray