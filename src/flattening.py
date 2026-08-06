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


        if (abs(distance) < threshold) or (distance > 0):

            # project to plane
            new_point = (
                p - distance * normal
            )

        else:

            # keep feature
            new_point = p


        flattened.append(new_point)


    return np.array(flattened)

def flatten_mesh(vertices, cells, fit_plane_function, threshold):
    """
    Flattens all occupied cells independently.

    Parameters
    ----------
    vertices : ndarray (N, 3)
        Original mesh vertices.

    cells : dict
        Mapping from cell coordinates to vertex indices.

    fit_plane_function : function
        Function that takes cell points and returns a fitted plane.

    threshold : float
        Maximum absolute distance from the plane for a point
        to be flattened.

    Returns
    -------
    new_vertices : ndarray (N, 3)
        Modified copy of the original vertices.

    cell_planes : dict
        Fitted plane for each processed cell.

    statistics : dict
        Information about the processing.
    """

    # Never modify the original vertex array
    new_vertices = vertices.copy()

    cell_planes = {}

    processed_cells = 0
    skipped_cells = 0
    flattened_vertices = 0
    preserved_vertices = 0

    MIN_POINTS = 100

    for cell, indices in cells.items():

        # Skip cells with too few points
        if len(indices) < MIN_POINTS:
            skipped_cells += 1
            continue

        # Get points belonging to this cell
        cell_points = vertices[indices]

        # Fit PCA plane
        plane = fit_plane_function(cell_points)

        cell_planes[cell] = plane

        # Flatten this cell
        flattened_points = flatten_cell(
            cell_points,
            plane,
            threshold
        )

        # Count what happened
        changed = np.linalg.norm(
            flattened_points - cell_points,
            axis=1
        )

        flattened_vertices += np.count_nonzero(
            changed > 1e-10
        )

        preserved_vertices += np.count_nonzero(
            changed <= 1e-10
        )

        # Put the modified points back into the global array
        new_vertices[indices] = flattened_points

        processed_cells += 1

    statistics = {
        "processed_cells": processed_cells,
        "skipped_cells": skipped_cells,
        "flattened_vertices": flattened_vertices,
        "preserved_vertices": preserved_vertices,
    }

    return new_vertices, cell_planes, statistics