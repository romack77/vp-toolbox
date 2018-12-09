import os

import numpy as np
import scipy.io

from vp import geom_tools
import dataset
from print_progress import print_progress

IMAGE_EXTENSION = 'jpg'
# Pixel dimensions of the images in the dataset.
IMAGE_DIMS = (640, 480)


def load_dataset(dataset_path, show_progress_bar=True):
    """Loads the York Urban vanishing point dataset.

    Args:
        dataset_path: String, path to a directory.
        show_progress_bar: Boolean, whether to print a progress bar.

    Returns:
        Dataset instance.
    """
    image_paths = []
    image_gt_vps = []
    image_gt_segments = []
    # Images paths are <base>/<base>.jpg.
    # VP data paths are <base>/<base>LinesAndVP.mat.
    entries = os.listdir(dataset_path)
    if show_progress_bar:
        entries = print_progress(entries)
    for entry in entries:
        if not os.path.isdir(os.path.join(dataset_path, entry)):
            continue
        image_file_name = os.path.join(entry, '%s.%s' % (entry, IMAGE_EXTENSION))
        image_paths.append(os.path.join(dataset_path, image_file_name))
        gt_data = scipy.io.loadmat(os.path.join(
            dataset_path, entry, '%sLinesAndVP.mat' % entry))
        gt_segments = [[], [], []]
        # Lines are in a flattened array of (x, y) point tuples.
        lines = gt_data['lines']
        # Labels are 1-3, one per line.
        vp_labels = gt_data['vp_association']
        for i in range(len(lines) // 2):
            gt_segments[int(vp_labels[i]) - 1].append([
                lines[i * 2][0],
                lines[i * 2][1],
                lines[i * 2 + 1][0],
                lines[i * 2 + 1][1]])
        gt_vps = []
        for vp_segments in gt_segments:
            if len(vp_segments) > 0:
                vx, vy = geom_tools.find_point_cluster_average(
                    geom_tools.find_all_intersections(vp_segments))
                if not np.isnan(vx) and not np.isnan(vy):
                    gt_vps.append((vx, vy))
        image_gt_segments.append(gt_segments)
        image_gt_vps.append(gt_vps)
    return dataset.Dataset(image_paths, IMAGE_DIMS, image_gt_vps, image_gt_segments)
