import math

import cv2


def cluster_xmeans(data_points, max_clusters=100):
    """Clusters data points into an unspecified number of clusters.

    The number of clusters is arrived at via Bayesian Information Criterion (BIC),
    which privileges the amount of variance explained by a clustering, while penalizing
    the number of clusters. We start with one cluster and split it
    until BIC stops improving. Uses kmeans for clustering.

    Args:
        data_points: List of data points.
        max_clusters: Integer, a hard limit on the maximum number of clusters.
            Can be None for no limit.

    Returns:
        Array of labels for each data point.
    """
    num_clusters = 1
    if max_clusters is None:
        max_clusters = math.inf

    best_labels, best_centers = cluster_kmeans(data_points, num_clusters)
    best_clustering_score = _score_clustering(data_points, best_labels, best_centers)
    while num_clusters < len(data_points) and num_clusters <= max_clusters:
        num_clusters += 1
        labels, centers = cluster_kmeans(data_points, num_clusters)
        clustering_score = _score_clustering(data_points, labels, centers)
        if clustering_score < best_clustering_score:
            best_labels = labels
            best_clustering_score = clustering_score
        else:
            # Results are no longer improving with additional clusters.
            break
    return best_labels


def cluster_kmeans(data_points, num_clusters, max_iterations=100, max_accuracy=0.25):
    """Clusters data points into a given number of clusters.

     Uses kmeans.

    Args:
        data_points: List of data points.
        num_clusters: Integer.
        max_iterations: Integer, how many iterations kmeans is allowed to run.
        max_accuracy: Float, stop kmeans when the impact of a marginal iteration
            falls below this threshold.

    Returns:
        Array of labels for each data point, array of centers for each cluster.
    """
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        max_iterations,
        max_accuracy)
    _, labels, centers = cv2.kmeans(
        data=data_points, K=num_clusters, bestLabels=None, criteria=criteria,
        attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS)
    return [l[0] for l in labels], centers


def _score_clustering(data_points, labels, centers):
    """Calculates a score for how efficiently a given clustering models the data.

    The score is trying to weigh model fit against model complexity.

    Args:
        data_points: List of data points.
        labels: List of integer data point labels, in corresponding order
            to the data_points argument.
        centers: List of cluster centers.

    Returns:
        float, score, lower is better.
    """
    num_obs = data_points.shape[0]
    num_clusters = len(centers)
    total_sse = _get_total_cluster_sse(data_points, labels, centers)
    if total_sse <= 0:
        # A set of duplicate points will trigger this.
        return 0

    # Bayes Information Criterion
    return num_obs * math.log(total_sse / num_obs) + num_clusters * math.log(num_obs)


def _get_total_cluster_sse(data_points, labels, centers):
    """Calculates sum of all within-cluster sum of squared errors.

    Args:
        data_points: List of data points.
        labels: List of integer data point labels, in corresponding order
            to the data_points argument.
        centers: List of cluster centers.

    Returns:
        float, sum of squared error.
    """
    center_to_group_data_points = {}
    for i, point in enumerate(data_points):
        group_center = tuple(centers[labels[i]])
        center_to_group_data_points.setdefault(group_center, []).append(point)
    within_group_sse = 0
    for center, group_points in center_to_group_data_points.items():
        within_group_sse += _get_cluster_sse(group_points, center)
    return within_group_sse


def _get_cluster_sse(data_points, center):
    """Calculates sum of squared error in a single cluster.

    Args:
        data_points: List of data points.
        center: Cluster center.

    Returns:
        float, sum of squared error.
    """
    variance = 0
    for point in data_points:
        variance += _get_distance_between_points(
            tuple(center), tuple(point)) ** 2
    return variance


def _get_distance_between_points(point_a, point_b):
    """Gets distance between two points of any dimensionality.

    Args:
        point_a: Tuple of any size.
        point_b: Tuple of any size.

    Returns:
        float, distance.
    """
    num_dimensions = len(point_a)
    return math.sqrt(sum([abs(point_a[i] - point_b[i]) ** 2
                          for i in range(num_dimensions)]))
