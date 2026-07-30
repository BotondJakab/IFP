import numpy as np

from geometry import Plane


def fit_plane_pca(points):
    """
    Fits a plane to a set of 3D points using PCA.

    Parameters
    ----------
    points : ndarray (N x 3)

    Returns
    -------
    Plane
    """
    # Step 1: Compute the centroid
    centroid = np.mean(points, axis=0)

    print("Centroid:")
    print(centroid)


    # Step 2: Center the points
    centered_points = points - centroid

    #print("First original point:")
    #print(points[0])

    #print("First centered point:")
    #print(centered_points[0])

    print("Centered mean:")
    print(np.mean(centered_points, axis=0))


    # Step 3: Compute covariance matrix
    covariance = np.cov(
        centered_points,
        rowvar=False
    )

    print("Covariance matrix:")
    print(covariance)


    # Step 4: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    print("Eigenvalues:")
    print(eigenvalues)

    print("Eigenvectors:")
    print(eigenvectors)


    # Step 5: Extract normal vector
    normal = eigenvectors[:, 0]

    # Make sure normal points roughly outward (+Z)
    if normal[2] < 0:
        normal = -normal

    # Plane equation:
    # ax + by + cz + d = 0

    a, b, c = normal

    d = -np.dot(normal, centroid)


    plane = Plane(
        a=a,
        b=b,
        c=c,
        d=d,
        point=centroid,
        normal=normal
    )

    print("Plane normal:")
    print(plane.normal)

    print("Plane equation:")
    print(
        f"{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f}=0"
    )

    return plane
