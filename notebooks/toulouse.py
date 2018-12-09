import json
import os

from vp import geom_tools

import dataset
from print_progress import print_progress

IMAGE_EXTENSION = 'jpg'
# Pixel dimensions of the images in the dataset.
IMAGE_DIMS = (1920, 1080)


def load_dataset(dataset_path, show_progress_bar=True):
    """Loads the Toulouse vanishing point dataset.

    Args:
        dataset_path: String, path to a directory.
        show_progress_bar: Boolean, whether to print a progress bar.

    Returns:
        Dataset instance.
    """
    image_paths = []
    image_gt_segments = []
    image_gt_vps = []
    entries = os.listdir(dataset_path)
    if show_progress_bar:
        entries = print_progress(entries)
    for file_name in entries:
        file_parts = file_name.rsplit('.', 1)
        if len(file_parts) < 2:
            continue
        file_base_name, extension = file_parts
        if extension != IMAGE_EXTENSION:
            continue
        image_paths.append(os.path.join(dataset_path, file_name))
        with open(os.path.join(
                dataset_path, '%s.txt' % file_base_name), 'r') as f:
            gt_segments = json.loads(f.read())['segments']
        gt_vps = []
        for vp_segments in gt_segments:
            gt_vps.append(geom_tools.find_point_cluster_average(
                geom_tools.find_all_intersections(
                    vp_segments)))
        image_gt_segments.append(gt_segments)
        image_gt_vps.append(gt_vps)
    return dataset.Dataset(image_paths, IMAGE_DIMS, image_gt_vps, image_gt_segments)
