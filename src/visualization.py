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

def show_cell(points):
    """
    Displays a single raster cell as a point cloud.
    """

    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries(
        [point_cloud],
        window_name="Raster Cell"
    )

def create_plane_mesh(plane, size=80):
    """
    Creates a square mesh representing a fitted plane.
    """

    center = plane.point
    normal = plane.normal


    # Create two vectors perpendicular to the normal
    # First choose an arbitrary vector
    if abs(normal[0]) < 0.9:
        helper = np.array([1, 0, 0])
    else:
        helper = np.array([0, 1, 0])


    # First tangent direction
    tangent1 = np.cross(normal, helper)
    tangent1 = tangent1 / np.linalg.norm(tangent1)


    # Second tangent direction
    tangent2 = np.cross(normal, tangent1)
    tangent2 = tangent2 / np.linalg.norm(tangent2)


    # Create square corners
    p1 = center + size/2 * tangent1 + size/2 * tangent2
    p2 = center - size/2 * tangent1 + size/2 * tangent2
    p3 = center - size/2 * tangent1 - size/2 * tangent2
    p4 = center + size/2 * tangent1 - size/2 * tangent2


    vertices = np.array([
        p1,
        p2,
        p3,
        p4
    ])


    triangles = np.array([
        [0,1,2],
        [0,2,3],

        # backside
        [2,1,0],
        [3,2,0]
    ])


    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    mesh.compute_vertex_normals()


    return mesh

def show_cell_with_plane(points, plane):

    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(points)


    plane_mesh = create_plane_mesh(
        plane,
        size=80
    )

    #plane_mesh.paint_uniform_color([0.8,0.8,0.8])

    o3d.visualization.draw_geometries(
        [
            point_cloud,
            plane_mesh
        ],
        window_name="Cell with fitted plane"
    )