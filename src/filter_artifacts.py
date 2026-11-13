import numpy as np

def filter_grid_artifacts(
    leveled_vertices,
    shifted_leveled_vertices,
    threshold
):
    # A feature must be below the reference plane in BOTH results
    stable_features = (
        (leveled_vertices[:, 2] < -threshold) &
        (shifted_leveled_vertices[:, 2] < -threshold)
    )

    # Start with the first result
    filtered_vertices = leveled_vertices.copy()

    # Remove features that were not detected in both runs
    filtered_vertices[~stable_features, 2] = 0.0

    # Some useful statistics
    stable_count = np.sum(stable_features)
    removed_count = len(stable_features) - stable_count

    print("Grid artifact filtering:")
    print(f"Stable feature vertices: {stable_count}")
    print(f"Flattened vertices:      {removed_count}")

    return filtered_vertices