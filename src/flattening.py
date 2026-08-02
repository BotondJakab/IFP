import numpy as np

def flatten_cell(points, plane, threshold):
    """
    Flatten points close to a fitted plane.

    Points farther than threshold are preserved.
     
    Parameters
    ----------
    points : ndarray (N,3)
        Original points

    plane :
        Plane object with:
        - point
        - normal

    Returns
    -------
    ndarray
        Projected points
    """

    normal = plane.normal
    plane_point = plane.point

    flattened = []

    for p in points:

        distance = np.dot(
            p - plane_point,
            normal
        )


        if abs(distance) < threshold:

            # project to plane
            new_point = (
                p - distance * normal
            )

        else:

            # keep feature
            new_point = p


        flattened.append(new_point)


    return np.array(flattened)