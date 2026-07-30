import open3d as o3d
import numpy as np


def show_mesh(mesh):

    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Original Mesh"
    )


def create_grid_lines(grid, z):

    lines = []

    x_min = grid["x_min"]
    y_min = grid["y_min"]
    cell_size = grid["cell_size"]

    nx = grid["nx"]
    ny = grid["ny"]


    # vertical lines (constant x)
    for i in range(nx + 1):

        x = x_min + i * cell_size

        lines.append([
            [x, y_min, z],
            [x, y_min + ny * cell_size, z]
        ])


    # horizontal lines (constant y)
    for j in range(ny + 1):

        y = y_min + j * cell_size

        lines.append([
            [x_min, y, z],
            [x_min + nx * cell_size, y, z]
        ])


    points = []
    line_indices = []

    for line in lines:

        start = len(points)

        points.append(line[0])
        points.append(line[1])

        line_indices.append(
            [start, start + 1]
        )


    grid_lines = o3d.geometry.LineSet()

    grid_lines.points = o3d.utility.Vector3dVector(points)

    grid_lines.lines = o3d.utility.Vector2iVector(line_indices)

    return grid_lines

