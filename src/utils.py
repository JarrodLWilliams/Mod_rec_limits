import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


def filtered_voronoi_edges(points_A, points_B):
    # Combine points
    points = np.vstack([points_A, points_B])

    # Label them: 0 = A, 1 = B
    labels = np.array([0]*len(points_A) + [1]*len(points_B))

    vor = Voronoi(points)

    filtered_edges = []

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        # Skip infinite ridges
        if v1 == -1 or v2 == -1:
            continue

        # Keep only edges between DIFFERENT sets
        if labels[p1] != labels[p2]:
            vtx1 = vor.vertices[v1]
            vtx2 = vor.vertices[v2]
            filtered_edges.append((vtx1, vtx2))

    return vor, filtered_edges



def plot_filtered_voronoi(points_A, points_B):
    vor, edges = filtered_voronoi_edges(points_A, points_B)

    plt.figure()

    # Plot original points
    plt.scatter(points_A[:, 0], points_A[:, 1], color='blue', label='A', marker="*")
    plt.scatter(points_B[:, 0], points_B[:, 1], color='red', label='B', marker="+")

    # Plot filtered edges
    for v1, v2 in edges:
        plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-')

    plt.legend()
    plt.title("Filtered Voronoi (A–B boundaries only)")
    plt.gca().set_aspect('equal')
    plt.show()