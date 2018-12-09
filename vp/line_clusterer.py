import numpy as np

from vp import clusterer
from vp import geom_tools


def cluster_lines_by_direction(lines, max_clusters=100):
    """Clusters lines according to their direction.

    Args:
        lines: List of lines specified as (x1, y1, x2, y2), i.e. two points on the line.
        max_clusters: Integer, a hard limit on the maximum number of clusters.
            Can be None for no limit.

    Returns:
        Array of labels for each line.
    """
    line_angles = np.float32(np.asarray([[geom_tools.get_line_angle(l)] for l in lines]))
    return clusterer.cluster_xmeans(line_angles, max_clusters=max_clusters)


