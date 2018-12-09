import math

import cv2
import numpy as np

import ransac

from vp import line_detector
from vp import vp_ransac


def find_vanishing_points_in_image(
        image,
        line_detection_options=None,
        ransac_options=None,
        x_ransac_options=None):
    """Estimates the vanishing points in an image.

    There are no accuracy guarantees on the number of vanishing points
    returned (it may be under or over ground truth, or even 0), or their
    estimated locations.

    Args:
        image: OpenCV image.
        line_detection_options: LineDetectionOptions instance. If None,
            a default instance is used.
        ransac_options: RansacOptions instance. If None, a default instance
            is used.
        x_ransac_options: An XRansacOptions instance. If supplied, X-RANSAC
            is used to choose VP models instead of J-linkage.

    Returns:
        Tuple of:
            1) Dict of vanishing point tuple to line set.
            2) List of lines that were rejected by all models, if any.
        Both items may be empty if a valid model was never found.
    """
    if line_detection_options is None:
        line_detection_options = LineDetectionOptions()
    if ransac_options is None:
        ransac_options = RansacOptions()

    lines = _detect_lines(image, line_detection_options)

    if x_ransac_options is None:
        return _find_vanishing_points_j_linkage(lines, ransac_options)
    else:
        image_height, image_width, _ = image.shape
        image_diagonal_length = round(
            math.sqrt(image_height ** 2 + image_width ** 2))
        return _find_vanishing_points_x_ransac(
            lines, ransac_options, x_ransac_options, image_diagonal_length)


class LineDetectionOptions(object):

    def __init__(self, min_edge_length=0.055, min_edge_precision=0.05):
        # These values are in terms of a ratio of the pixels in the
        # diagonal of the image.
        self.min_edge_length = min_edge_length
        self.min_edge_precision = min_edge_precision


class RansacOptions(object):
    """Stores options for the X-RANSAC extension."""

    def __init__(
            self,
            model=None,
            num_sample_points=2,
            inlier_threshold=3,
            min_inliers=4,
            outlier_rate=0.73,
            desired_success_rate=0.999,
            max_iterations=10000,
            random_seed=0):
        if model is None:
            model = vp_ransac.SegmentVPModel()
        self.model = model
        self.num_sample_points = num_sample_points
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers
        self.outlier_rate = outlier_rate
        self.desired_success_rate = desired_success_rate
        self.max_iterations = max_iterations
        self.random_seed = random_seed


class XRansacOptions(object):
    """Stores options for the X-RANSAC extension."""

    def __init__(
            self,
            min_peak_samples=10,
            residual_histogram_num_bins=25,
            min_prominence=5):
        self.min_peak_samples = min_peak_samples
        self.residual_histogram_num_bins = residual_histogram_num_bins
        self.min_prominence = min_prominence


def _detect_lines(image, options):
    """Finds lines in an image.

    Args:
        image: OpenCV image.
        options: LineDetectionOptions instance.

    Returns:
        List of lines in [x1, y1, x2, y2] format.
    """
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = line_detector.lsd_lines(
        image,
        min_line_length=options.min_edge_length,
        min_precision=options.min_edge_precision)
    return lines


def _find_vanishing_points_x_ransac(
        lines, ransac_options, x_ransac_options):
    """Runs XRansac on a given line set to detect vanishing points.

    Args:
        lines: List of lines specified as (x1, y1, x2, y2),
            i.e. two points on the line.
        ransac_options: RansacOptions instance.
        x_ransac_options: XRansacOptions instance.

    Returns:
        Tuple of:
            1) Dict of vanishing point tuple to line set.
            2) List of lines that were rejected by all models, if any.
        Both items may be empty if a valid model was never found.
    """
    stop_iterations = ransac.calculate_xransac_iterations(
        ransac_options.num_sample_points,
        ransac_options.outlier_rate,
        ransac_options.desired_success_rate,
        min_peak_samples=x_ransac_options.min_peak_samples)
    ransac_inst = ransac.XRansac(
        ransac_options.model,
        num_sample_points=ransac_options.num_sample_points,
        min_inliers=ransac_options.min_inliers,
        inlier_threshold=ransac_options.inlier_threshold,
        stop_iterations=stop_iterations,
        random_seed=ransac_options.random_seed,
        residual_histogram_num_bins=x_ransac_options.residual_histogram_num_bins,
        min_prominence=x_ransac_options.min_prominence)
    results = ransac_inst.run(np.array(lines))
    if not results or len(results.get_model_results()) == 0:
        return {}, lines
    vp_to_inliers = {r.fit: r.inliers for r in results.get_model_results()}
    return vp_to_inliers, results.get_global_outliers()


def _find_vanishing_points_j_linkage(lines, options):
    """Runs J-linkage on a given line set to detect vanishing points.

    Args:
        lines: List of lines specified as (x1, y1, x2, y2),
            i.e. two points on the line.
        options: RansacOptions instance.

    Returns:
        Tuple of:
            1) Dict of vanishing point tuple to line set.
            2) List of lines that were rejected by all models, if any.
        Both items may be empty if a valid model was never found.
    """
    stop_iterations = ransac.calculate_ransac_iterations(
        options.num_sample_points,
        options.outlier_rate,
        options.desired_success_rate)
    j_link = ransac.JLinkage(
        options.model,
        num_sample_points=options.num_sample_points,
        min_inliers=options.min_inliers,
        inlier_threshold=options.inlier_threshold,
        stop_iterations=stop_iterations,
        random_seed=options.random_seed)
    results = j_link.run(np.array(lines))
    if not results or len(results.get_model_results()) == 0:
        return {}, lines
    vp_to_inliers = {r.fit: r.inliers for r in results.get_model_results()}
    return vp_to_inliers, results.get_global_outliers()
