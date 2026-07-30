import numpy as np

from geometry import Plane


def point_plane_distance(points, plane):
    """
    Computes signed distances from points to a plane.

    Parameters
    ----------
    points : ndarray (N x 3)
        3D points

    plane : Plane
        Plane object

    Returns
    -------
    distances : ndarray (N,)
    """

    numerator = (
        points[:,0] * plane.a +
        points[:,1] * plane.b +
        points[:,2] * plane.c +
        plane.d
    )

    denominator = np.sqrt(
        plane.a**2 +
        plane.b**2 +
        plane.c**2
    )

    distances = numerator / denominator

    return distances