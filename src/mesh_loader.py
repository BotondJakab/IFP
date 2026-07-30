import open3d as o3d
import numpy as np


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)

    if not mesh.has_vertices():
        raise Exception("Mesh contains no vertices")

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    return mesh, vertices, faces