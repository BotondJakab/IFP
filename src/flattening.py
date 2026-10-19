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


        if (abs(distance) < threshold) or (distance > 0):# or (abs(distance)>0.01):

            # project to plane
            new_point = (
                p - distance * normal
            )

        else:

            # keep feature
            new_point = p


        flattened.append(new_point)


    return np.array(flattened)

def flatten_mesh_old(vertices, cells, fit_plane_function, threshold):
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


def flatten_mesh_less_old(vertices, cells, fit_plane_function, threshold):
    """
    WORKING VERSION WITHOUT UI
    Flattens all occupied cells independently.

    Cells with enough vertices get their own PCA plane.
    Cells with too few vertices use the plane of the
    nearest valid cell.

    Parameters
    ----------
    vertices : ndarray (N, 3)
        Original mesh vertices.

    cells : dict
        Mapping from cell coordinates to vertex indices.

    fit_plane_function : function
        Function that takes cell points and returns a fitted plane.

    threshold : float
        Maximum distance from the plane for a point
        to be flattened.

    Returns
    -------
    new_vertices : ndarray (N, 3)
        Modified copy of the original vertices.

    cell_planes : dict
        Plane used by every occupied cell.

    statistics : dict
        Information about the processing.
    """

    # Never modify the original vertex array
    new_vertices = vertices.copy()

    MIN_POINTS = 100

    # --------------------------------------------------
    # STEP 1:
    # Fit planes only for cells with enough points.
    # --------------------------------------------------

    valid_cell_planes = {}

    skipped_cells = 0

    for cell, indices in cells.items():

        if len(indices) < MIN_POINTS:
            skipped_cells += 1
            continue

        cell_points = vertices[indices]

        plane = fit_plane_function(
            cell_points
        )

        valid_cell_planes[cell] = plane

    # --------------------------------------------------
    # STEP 2:
    # Give every occupied cell a plane.
    #
    # Valid cells use their own plane.
    # Sparse cells use the nearest valid cell's plane.
    # --------------------------------------------------

    cell_planes = {}

    inherited_cells = 0

    for cell in cells:

        # Cell has enough points -> use own plane
        if cell in valid_cell_planes:

            cell_planes[cell] = (
                valid_cell_planes[cell]
            )

            continue

        # Otherwise find nearest valid cell
        nearest_cell = min(
            valid_cell_planes.keys(),
            key=lambda other_cell:
                (cell[0] - other_cell[0]) ** 2
                +
                (cell[1] - other_cell[1]) ** 2
        )

        cell_planes[cell] = (
            valid_cell_planes[nearest_cell]
        )

        inherited_cells += 1

    # --------------------------------------------------
    # STEP 3:
    # Flatten every cell using its assigned plane.
    # --------------------------------------------------

    processed_cells = 0
    flattened_vertices = 0
    preserved_vertices = 0

    for cell, indices in cells.items():

        cell_points = vertices[indices]

        plane = cell_planes[cell]

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

        # Put modified points back into global array
        new_vertices[indices] = flattened_points

        processed_cells += 1

    statistics = {
        "processed_cells": processed_cells,
        "skipped_cells": skipped_cells,
        "inherited_cells": inherited_cells,
        "flattened_vertices": flattened_vertices,
        "preserved_vertices": preserved_vertices,
    }

    return (
        new_vertices,
        cell_planes,
        statistics
    )


def flatten_mesh(
    vertices,
    cells,
    fit_plane_function,
    threshold,
    progress_callback=None,
):
    """
    Flatten all occupied cells independently.

    Optionally report progress through progress_callback.

    The callback receives:
        stage, cell, current, total, status

    Stages:
        "fitting_planes"
        "assigning_planes"
        "flattening"

    Status:
        "stage_started"
        "cell_started"
        "cell_completed"
        "stage_completed"
    """

    # Never modify the original vertex array
    new_vertices = vertices.copy()

    MIN_POINTS = 100

    # Send progress information only when a callback is provided
    def report(stage, cell, current, total, status):
        if progress_callback is not None:
            progress_callback(
                stage, cell, current, total, status
            )

    # --------------------------------------------------
    # STEP 1: Fit planes to cells with enough points
    # --------------------------------------------------

    valid_cell_planes = {}

    skipped_cells = 0
    total_cells = len(cells)

    report(
        "fitting_planes", None, 0, total_cells,
        "stage_started"
    )

    for current, (cell, indices) in enumerate(
        cells.items(), start=1
    ):
        report(
            "fitting_planes", cell, current, total_cells,
            "cell_started"
        )

        if len(indices) < MIN_POINTS:
            skipped_cells += 1
        else:
            cell_points = vertices[indices]

            plane = fit_plane_function(cell_points)

            valid_cell_planes[cell] = plane

        report(
            "fitting_planes", cell, current, total_cells,
            "cell_completed"
        )

    report(
        "fitting_planes", None, total_cells, total_cells,
        "stage_completed"
    )

    # --------------------------------------------------
    # STEP 2: Assign a plane to every occupied cell
    # --------------------------------------------------

    cell_planes = {}
    inherited_cells = 0

    report(
        "assigning_planes", None, 0, total_cells,
        "stage_started"
    )

    for current, cell in enumerate(cells, start=1):
        report(
            "assigning_planes", cell, current, total_cells,
            "cell_started"
        )

        if cell in valid_cell_planes:
            cell_planes[cell] = valid_cell_planes[cell]

        else:
            if not valid_cell_planes:
                raise ValueError(
                    "Cannot flatten the mesh: no occupied cell "
                    "has enough points to fit a plane. "
                    "Consider reducing MIN_POINTS."
                )

            nearest_cell = min(
                valid_cell_planes.keys(),
                key=lambda other_cell:
                    (cell[0] - other_cell[0]) ** 2
                    + (cell[1] - other_cell[1]) ** 2
            )

            cell_planes[cell] = valid_cell_planes[nearest_cell]
            inherited_cells += 1

        report(
            "assigning_planes", cell, current, total_cells,
            "cell_completed"
        )

    report(
        "assigning_planes", None, total_cells, total_cells,
        "stage_completed"
    )

    # --------------------------------------------------
    # STEP 3: Flatten every occupied cell
    # --------------------------------------------------

    processed_cells = 0
    flattened_vertices = 0
    preserved_vertices = 0

    report(
        "flattening", None, 0, total_cells,
        "stage_started"
    )

    for current, (cell, indices) in enumerate(
        cells.items(), start=1
    ):
        report(
            "flattening", cell, current, total_cells,
            "cell_started"
        )

        cell_points = vertices[indices]
        plane = cell_planes[cell]

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

        # Put modified points back into the global array
        new_vertices[indices] = flattened_points

        processed_cells += 1

        report(
            "flattening", cell, current, total_cells,
            "cell_completed"
        )

    report(
        "flattening", None, total_cells, total_cells,
        "stage_completed"
    )

    statistics = {
        "processed_cells": processed_cells,
        "skipped_cells": skipped_cells,
        "inherited_cells": inherited_cells,
        "flattened_vertices": flattened_vertices,
        "preserved_vertices": preserved_vertices,
    }

    return new_vertices, cell_planes, statistics
