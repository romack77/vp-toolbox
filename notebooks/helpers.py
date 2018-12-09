import itertools
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

from vp import draw_tools
from vp import horizon_finder
from vp import scoring
from print_progress import print_progress

HORIZON_LINE_COLOR = (255, 255, 0)
BAD_LINES_COLOR = (0, 55, 0)


def batch_detect_vps_and_score(dataset, detection_func, show_progress_bar=True):
    """Finds vanishing points and detection error vs ground truth.

    Args:
        dataset: Dataset instance. Contains image and ground truth info.
        detection_func: Function taking a cv2 image, and returning
            a tuple of:
                Dict of VP tuple to list of constituent lines.
                List of outlier lines. Lines are in [x1, y1, x2, y2] format.
        show_progress_bar: Boolean, whether to print a progress bar.

    Returns:
        VPResults instance.
    """
    image_vp_to_lines = []
    image_horizon_params = []
    image_bad_lines = []
    horizon_errors = []
    num_model_errors = []
    vp_direction_errors = []
    location_errors = []
    detection_times = []
    principal_point = (dataset.image_dims[0] // 2, dataset.image_dims[1] // 2)

    image_paths = dataset.image_paths
    if show_progress_bar:
        image_paths = print_progress(dataset.image_paths)
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        detection_time_secs = time.time()
        vp_to_lines, bad_lines = detection_func(image)
        detection_time_secs = time.time() - detection_time_secs
        detection_times.append(detection_time_secs)
        horizon_params = horizon_finder.find_horizon(vp_to_lines.keys(), principal_point)
        image_horizon_params.append(horizon_params)

        image_vp_to_lines.append(vp_to_lines)
        image_bad_lines.append(bad_lines)
        horizon_errors.append(scoring.horizon_error(
            dataset.image_gt_horizon[i], horizon_params, dataset.image_dims))
        num_model_errors.append(scoring.num_model_detection_error(
            dataset.image_gt_vps[i], vp_to_lines.keys()))
        vp_direction_errors.append(scoring.vp_direction_error(
            dataset.image_gt_vps[i], vp_to_lines.keys(), dataset.image_dims))
        location_errors.append(scoring.location_accuracy_error(
            dataset.image_gt_vps[i], vp_to_lines.keys()))
    return VPResults(
        image_vp_to_lines, image_horizon_params, image_bad_lines,
        horizon_errors, num_model_errors, vp_direction_errors,
        location_errors, detection_times)


def show_results_summary(dataset, results):
    """Plots and prints a summary of detection results.

    Args:
        dataset: Dataset instance.
        results: VPResults instance.
    """
    print('Horizon detection error is the pixel distance between the detected '
          'and actual horizon, divided by the image height.')
    auc = histogram_cumulative(
        results.horizon_errors, range=(0, .25), title='Horizon error cumulative')
    print(('\t%0.2f%% of results were within a quarter-image distance of the '
           'actual horizon.') % (auc * 100))

    print('\nThis shows the horizon error if we always guessed a flat, '
          'image-center horizon, i.e., no algorithm at all.')
    fixed_horizon_error = [
        scoring.horizon_error(
            gt, (0, dataset.image_dims[1] // 2), dataset.image_dims)
        for gt in dataset.image_gt_horizon]
    auc = histogram_cumulative(
        fixed_horizon_error, range=(0, .25), title='Fixed-horizon error cumulative')
    print(('\t%0.2f%% of the images had a horizon closer to the center of the image '
           'than the edges.') % (auc * 100))

    vp_direction_errors = list(itertools.chain(*results.vp_direction_errors))
    non_empty_vp_direction_errors = [e for e in vp_direction_errors if e is not None]
    num_missed = len(vp_direction_errors) - len(non_empty_vp_direction_errors)
    dir_err_or_inf = [e if e is not None else np.inf for e in vp_direction_errors]
    print('\nVP direction error is the difference in angle from image center to the '
          'actual and detected VPs.')
    auc = histogram_cumulative(
        [e if e is not None else np.inf for e in vp_direction_errors], range=(0, 5),
        title='VP direction error cumulative')
    print(('\t%0.2f%% of GT VPs were detected within 5 degrees, or %0.2f%% '
           'if we omit the %0.2f%% of GT VPs that were completely undetected.') % (
        auc * 100,
        (len(vp_direction_errors) * auc) / len(non_empty_vp_direction_errors) * 100,
        num_missed / float(len(vp_direction_errors)) * 100))

    histogram_cumulative(
        dir_err_or_inf, range=(0, 180), title='VP direction error cumulative')

    histogram(results.num_model_errors, bins=[i + 0.5 for i in range(-4, 4)],
              title='Num model error (<0 fewer than GT, >0 more than GT)')
    histogram(results.detection_times, title='Detection time (secs)')


def visualize_vp_detection_results(dataset, vp_results, mask_indices=None):
    """Displays plots illustrating detection results.

    A side by side plot of ground truth and detection results.

    Args:
        dataset: Dataset instance.
        vp_results: VPResults instance.
        mask_indices: If supplied, only these indices of image_paths will
            be displayed.
    """
    print('Left side is ground truth, right side is detection results.')
    if mask_indices is None:
        mask_indices = range(len(dataset.image_paths))
    for i in mask_indices:
        # Build an image with ground truth information.
        image = cv2.imread(dataset.image_paths[i])
        gt_image = _build_results_image(
            image,
            dataset.image_gt_vps[i],
            dataset.image_gt_segments[i],
            [],
            dataset.image_gt_horizon[i])

        # Build an image with detection results.
        working_image = _build_results_image(
            image,
            vp_results.image_vp_to_lines[i].keys(),
            vp_results.image_vp_to_lines[i].values(),
            vp_results.image_bad_lines[i],
            vp_results.image_horizon_params[i])

        fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        ax[0].imshow(gt_image)
        ax[1].imshow(working_image)
        plt.show()
        print('Horizon error: %s. VP angle error (degrees): %s. Line counts: %s' % (
            vp_results.horizon_errors[i],
            ', '.join(['%0.2f' % e if e is not None else 'NA'
                       for e in vp_results.vp_direction_errors[i]]),
            [len(vs) for vs in vp_results.image_vp_to_lines[i].values()]))


def _build_results_image(image, vps, segments, bad_segments, horizon_params):
    """Overlays VP info onto an image.

    Args:
        image: cv2 image instance.
        vps: List of vanishing point tuples.
        segments: List of contributing line segments, grouped by vanishing point.
            Lines are in [x1, y1, x2, y2] format.
        bad_segments: Outlier segments.
        horizon_params: Tuple of (slope, intercept) floats.

    Returns:
        cv2 image instance.
    """
    image = image.copy()
    im_height, im_width, _ = image.shape
    if len(bad_segments) > 0:
        draw_tools.draw_lines(bad_segments, image, color=BAD_LINES_COLOR)
    draw_tools.draw_line_groups(segments, image)
    for i, vp in enumerate(vps):
        color = draw_tools.DRAW_PALETTE[i]
        draw_tools.draw_fitted_point(vp, image, color=color, size=15)
    if horizon_params is not None:
        h_slope, h_intercept = horizon_params
        horizon_line = [0, h_intercept, im_width, h_slope * im_width + h_intercept]
        if 0 <= horizon_line[1] <= im_height and 0 <= horizon_line[3] <= im_height:
            draw_tools.draw_lines([horizon_line], image, color=HORIZON_LINE_COLOR)
    return image


class VPResults(object):
    """Stores VP detection results for a batch of images."""

    def __init__(
            self, image_vp_to_lines, image_horizon_params, image_bad_lines,
            horizon_errors, num_model_errors, vp_direction_errors,
            location_errors, detection_times):
        """Constructor.

        Args:
            image_vp_to_lines: List of dicts of VP tuple to constituent line set.
            image_horizon_params: List of (slope, intercept) tuples, or None if
                detection failed for that image.
            image_bad_lines: List of outlier lines, one per image.
            horizon_errors: List of float horizon errors. See scoring.horizon_error.
            num_model_errors: List of integer num model errors.
            vp_direction_errors: List of lists of VP angle errors.
            location_errors: List of float average location errors.
            detection_times: List of float processing time in seconds.
        """
        self.image_vp_to_lines = image_vp_to_lines
        self.image_horizon_params = image_horizon_params
        self.image_bad_lines = image_bad_lines
        self.horizon_errors = horizon_errors
        self.num_model_errors = num_model_errors
        self.vp_direction_errors = vp_direction_errors
        self.location_errors = location_errors
        self.detection_times = detection_times

    def with_mask(self, indices):
        """Returns a new VPResults for a masked subset of images.

        Args:
            indices: Set of integer indices to keep.

        Returns:
            VPResults instance.
        """
        return VPResults(
            [self.image_vp_to_lines[i] for i in indices],
            [self.image_horizon_params[i] for i in indices],
            [self.image_bad_lines[i] for i in indices],
            [self.horizon_errors[i] for i in indices],
            [self.num_model_errors[i] for i in indices],
            [self.vp_direction_errors[i] for i in indices],
            [self.location_errors[i] for i in indices],
            [self.detection_times[i] for i in indices])


def histogram(data, bins=None, title=None, y_thresh=True):
    """Displays a standardized histogram.

    Args:
        data: List of values.
        bins: List of bin edges. Auto-detected if None.
        title: Optional string title.
        y_thresh: Optional bool or float y value at which to plot a
            threshold line. True will use the number of data points.
    """
    if bins is None:
        counts, bins = np.histogram(np.array(data))
        plt.ylim(ymax=max(counts), ymin=0)
    else:
        plt.ylim(ymax=len(data), ymin=0)
    plt.hist(data, bins=bins)
    if title is not None:
        plt.title(title)
    if y_thresh is True:
        plt.axhline(y=len(data), linewidth=1)
    elif y_thresh is not False:
        plt.axhline(y=y_thresh, linewidth=1)
    plt.show()
    print('\tMedian: %0.2f. Max: %0.2f' % (
        np.median(data), max(data)))


def histogram_cumulative(data, range, title=None, y_thresh=True):
    """Displays a standardized histogram.

    Args:
        data: List of values.
        range: Tuple, range of values to display.
        title: Optional string title.
        y_thresh: Optional bool or float y value at which to plot a
            threshold line. True will use the number of data points.

    Returns:
        Float, ratio of results that fell within the given range..
    """
    counts, bins = np.histogram(np.array(data), range=range)
    plt.hist(data, bins=bins, cumulative=True, histtype='step')
    if title is not None:
        plt.title(title)
    if y_thresh is True:
        plt.axhline(y=len(data), linewidth=1)
    elif y_thresh is not False:
        plt.axhline(y=y_thresh, linewidth=1)
    plt.show()
    print('\tMedian: %0.2f.' % np.median(data))
    return sum(counts) / len(data)
