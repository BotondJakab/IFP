from mesh_loader import load_mesh
from visualization import show_mesh, create_grid_lines
from rasterization import create_grid, assign_vertices
from plane_fitting import fit_plane_pca

# Load mesh
mesh, vertices, faces = load_mesh(
    "data/Worms_188.ply"
)


# Basic information
print("Number of vertices:", len(vertices))
print("Number of faces:", len(faces))


# Create XY grid
grid = create_grid(
    vertices,
    cell_size=60
)


print("Grid size:")
print(
    grid["nx"],
    "x",
    grid["ny"],
    "cells"
)

grid_lines = create_grid_lines(
    grid,
    z=vertices[:,2].max()
)


# Assign vertices to cells
cells = assign_vertices(
    vertices,
    grid
)

# Pick one cell (the first occupied one) 
"""
first_cell = next(iter(cells))

print("Testing cell:", first_cell)

indices = cells[first_cell]
cell_points = vertices[indices]

fit_plane_pca(cell_points)"""

test_cell = max(
    cells,
    key=lambda cell: len(cells[cell])
)

print("Testing cell:", test_cell)

indices = cells[test_cell]
cell_points = vertices[indices]



from plane_operations import point_plane_distance


plane = fit_plane_pca(cell_points)


distances = point_plane_distance(
    cell_points,
    plane
)


print("Distance statistics:")
print("min:", distances.min())
print("max:", distances.max())
print("mean:", distances.mean())
print("std:", distances.std())


print("Number of occupied cells:", len(cells))


# Show some example cells
for cell, ids in list(cells.items())[:5]:
    print(
        cell,
        "contains",
        len(ids),
        "vertices"
    )


# Finally visualize mesh
#show_mesh(mesh)

import open3d as o3d

o3d.visualization.draw_geometries(
    [
        mesh,
        grid_lines
    ],
    window_name="Mesh with grid"
)