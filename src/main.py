import numpy as np
import open3d as o3d

from mesh_loader import load_mesh
from visualization import (
    show_mesh,
    show_cell,
    create_grid_lines,
    show_cell_with_plane,
)
from rasterization import (
    create_grid,
    assign_vertices,
    get_cell_points,
)
from plane_fitting import fit_plane_pca
from plane_operations import point_plane_distance
from flattening import flatten_cell, flatten_mesh
from leveling import (
    level_cell,
    level_mesh,
)
from filter_artifacts import filter_grid_artifacts


# Load mesh
mesh, vertices, faces = load_mesh(
    "data/Worms_19.ply"
    #"data/BleiplatteSpeyer2.ply"
)


# Basic information
print("Number of vertices:", len(vertices))
print("Number of faces:", len(faces))


# Create XY grid
grid = create_grid(
    vertices,
    cell_size=60
)

shifted_grid = create_grid(
    vertices,
    cell_size=60,
    offset=30
)
threshold=1.0

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

new_vertices, cell_planes, statistics = flatten_mesh(
    vertices,
    cells,
    fit_plane_pca,
    threshold
)

leveled_vertices = level_mesh(
    new_vertices,
    cells,
    cell_planes
)

#shifted

shifted_grid_lines = create_grid_lines(
    shifted_grid,
    z=vertices[:,2].max()
)

shifted_cells =assign_vertices(
    vertices,
    shifted_grid
)

shifted_new_vertices, shifted_cell_planes, statistics = flatten_mesh(
    vertices,
    shifted_cells,
    fit_plane_pca,
    threshold
)

shifted_leveled_vertices = level_mesh(
    shifted_new_vertices,
    shifted_cells,
    shifted_cell_planes
)

filtered_vertices = filter_grid_artifacts(
    leveled_vertices,
    shifted_leveled_vertices,
    threshold
)



print()
print("Flattening results:")
print(
    "Processed cells:",
    statistics["processed_cells"]
)
print(
    "Skipped cells:",
    statistics["skipped_cells"]
)
print(
    "Flattened vertices:",
    statistics["flattened_vertices"]
)
print(
    "Preserved vertices:",
    statistics["preserved_vertices"]
)

"""
new_mesh = o3d.geometry.TriangleMesh()

new_mesh.vertices = o3d.utility.Vector3dVector(
    new_vertices
)

new_mesh.triangles = o3d.utility.Vector3iVector(
    faces
)

new_mesh.compute_vertex_normals()

o3d.visualization.draw_geometries(
    [new_mesh],
    window_name="Flattened Mesh"
)

##shifted
shifted_new_mesh = o3d.geometry.TriangleMesh()

shifted_new_mesh.vertices = o3d.utility.Vector3dVector(
    shifted_new_vertices
)

shifted_new_mesh.triangles = o3d.utility.Vector3iVector(
    faces
)

shifted_new_mesh.compute_vertex_normals()

o3d.visualization.draw_geometries(
    [shifted_new_mesh],
    window_name="shifted Flattened Mesh"
)
"""
leveled_mesh = o3d.geometry.TriangleMesh()

leveled_mesh.vertices = o3d.utility.Vector3dVector(
    leveled_vertices
)

leveled_mesh.triangles = o3d.utility.Vector3iVector(
    faces
)

leveled_mesh.compute_vertex_normals()

o3d.visualization.draw_geometries(
    [leveled_mesh],
    window_name="Leveled Mesh"
)


filtered_mesh = o3d.geometry.TriangleMesh()

filtered_mesh.vertices = o3d.utility.Vector3dVector(
    filtered_vertices
)

filtered_mesh.triangles = o3d.utility.Vector3iVector(
    faces
)

filtered_mesh.compute_vertex_normals()

o3d.visualization.draw_geometries(
    [filtered_mesh],
    window_name="Filtered Mesh"
)

# Finally visualize mesh
#show_mesh(mesh)

"""
o3d.visualization.draw_geometries(
    [
        mesh,
        grid_lines
    ],
    window_name="Mesh with grid"
)

"""

#=====================================================================

# Pick one cell (the first occupied one) 
"""
first_cell = next(iter(cells))

print("Testing cell:", first_cell)

indices = cells[first_cell]
cell_points = vertices[indices]

fit_plane_pca(cell_points)"""

"""
test_cell = (6, 8) #max(cells, key=lambda cell: len(cells[cell]))
test_cells = [
    (6,8),   # flat surface with carvings
    (7,8),   # more complex geometry
]

print("Testing cell:", test_cell)

cell_points = get_cell_points(
    vertices,
    cells,
    test_cell
)

plane = fit_plane_pca(cell_points)

distances = point_plane_distance(
    cell_points,
    plane
)

flattened_points = flatten_cell(
    cell_points,
    plane,
    threshold=1.2
)

leveled_points = level_cell(
    flattened_points,
    plane
) 

#show_cell_with_plane(leveled_points, np.zeros(len(flattened_points)),plane)
#show_cell(cell_points)
show_cell_with_plane(cell_points, distances, plane)
show_cell_with_plane(flattened_points, distances,plane)

"""
"""
print("Distance statistics:")
print("min:", distances.min())
print("max:", distances.max())
print("mean:", distances.mean())
print("std:", distances.std())

print("Number of occupied cells:", len(cells))
"""


#===============================================================================