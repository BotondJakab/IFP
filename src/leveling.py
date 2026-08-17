import numpy as np
from scipy.spatial.transform import Rotation


def level_cell(points, plane):
    """
    Rigidly transforms a cell so that its fitted plane
    becomes the global XY plane (z = 0), while preserving
    the cell's original XY position.

    Parameters
    ----------
    points : ndarray (N, 3)
        Points belonging to the cell.

    plane :
        Fitted plane with:
        - plane.point
        - plane.normal

    Returns
    -------
    ndarray (N, 3)
        Leveled points.
    """

    normal = np.asarray(
        plane.normal,
        dtype=float
    )

    plane_point = np.asarray(
        plane.point,
        dtype=float
    )

    # PCA normals have arbitrary direction.
    # Make sure the normal points roughly upward.
    if normal[2] < 0:
        normal = -normal

    # Normalize
    normal = normal / np.linalg.norm(normal)

    target_normal = np.array([
        0.0,
        0.0,
        1.0
    ])

    # Calculate rotation axis
    rotation_axis = np.cross(
        normal,
        target_normal
    )

    # Calculate rotation angle
    rotation_angle = np.arccos(
        np.clip(
            np.dot(
                normal,
                target_normal
            ),
            -1.0,
            1.0
        )
    )

    axis_length = np.linalg.norm(
        rotation_axis
    )

    if axis_length > 1e-10:

        rotation_axis = (
            rotation_axis /
            axis_length
        )

        rotation_vector = (
            rotation_axis *
            rotation_angle
        )

        rotation = Rotation.from_rotvec(
            rotation_vector
        )

    else:

        rotation = Rotation.identity()

    # --------------------------------------------------
    # Rotate points around the fitted plane point.
    # --------------------------------------------------

    relative_points = (
        points - plane_point
    )

    rotated_points = rotation.apply(
        relative_points
    )

    # --------------------------------------------------
    # Put the plane point back at its original XY
    # position, but at z = 0.
    # --------------------------------------------------

    target_plane_point = np.array([
        plane_point[0],
        plane_point[1],
        0.0
    ])

    leveled_points = (
        rotated_points +
        target_plane_point
    )

    return leveled_points

def level_mesh(vertices, cells, cell_planes):
    """
    Levels every occupied cell using its fitted PCA plane.

    Parameters
    ----------
    vertices : ndarray (N, 3)
        Flattened mesh vertices.

    cells : dict
        Mapping from cell coordinates to vertex indices.

    cell_planes : dict
        Fitted plane for each processed cell.

    Returns
    -------
    new_vertices : ndarray (N, 3)
        Leveled vertex positions.
    """

    new_vertices = vertices.copy()

    for cell, indices in cells.items():

        # If this cell did not receive a plane,
        # leave its vertices unchanged.
        if cell not in cell_planes:
            continue

        plane = cell_planes[cell]

        cell_points = vertices[indices]

        leveled_points = level_cell(
            cell_points,
            plane
        )

        new_vertices[indices] = leveled_points

    return new_vertices