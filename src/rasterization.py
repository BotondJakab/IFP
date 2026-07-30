import numpy as np


def create_grid(vertices, cell_size):

    x_min = vertices[:,0].min()
    x_max = vertices[:,0].max()

    y_min = vertices[:,1].min()
    y_max = vertices[:,1].max()


    width = x_max - x_min
    height = y_max - y_min


    nx = int(np.ceil(width / cell_size))
    ny = int(np.ceil(height / cell_size))


    return {
        "x_min": x_min,
        "y_min": y_min,
        "cell_size": cell_size,
        "nx": nx,
        "ny": ny
    }

def assign_vertices(vertices, grid):

    cell_vertices = {}


    for index, vertex in enumerate(vertices):

        x, y, z = vertex


        cx = int(
            (x - grid["x_min"])
            / grid["cell_size"]
        )

        cy = int(
            (y - grid["y_min"])
            / grid["cell_size"]
        )


        cell = (cx, cy)


        if cell not in cell_vertices:
            cell_vertices[cell] = []

        cell_vertices[cell].append(index)


    return cell_vertices

def get_cell_points(vertices, cells, cell):
    """
    Returns the vertices belonging to a specific cell.

    Parameters
    ----------
    vertices : ndarray (N x 3)

    cells : dict

    cell : tuple(int, int)

    Returns
    -------
    ndarray
        Vertices inside the requested cell.
    """

    indices = cells[cell]

    return vertices[indices]