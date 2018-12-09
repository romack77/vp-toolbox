import math

import cv2
import numpy as np

from vp import geom_tools


def enhance_edges(image):
    """Pre-processing step to enhance edges.

    Args:
        image: OpenCV image.

    Returns:
        Image, filtered for edge detection.
    """
    working_image = image.copy()
    working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
    # Blur away fine details.
    working_image = cv2.GaussianBlur(working_image, (5, 5), 0)
    return working_image


def lsd_lines(source_image, min_line_length=0.0375, max_line_length=1, min_precision=0):
    """LSD algorithm for line detection.

    Args:
        source_image: An OpenCV Image.
        min_line_length: Minimum line size. Specified as a percentage of the
            source image diagonal (0-1).
        max_line_length: Maximum line size. Specified as a percentage of the
            source image diagonal (0-1).
        min_precision: Minimum precision of detections.

    Returns:
        Array of line endpoints tuples (x1, y1, x2, y2).
    """
    height, width = source_image.shape[:2]
    diagonal = math.sqrt(height ** 2 + width ** 2)
    min_line_length = min_line_length * diagonal
    max_line_length = max_line_length * diagonal

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines, rect_widths, precisions, false_alarms = detector.detect(source_image)
    line_lengths = [geom_tools.get_line_length(l[0]) for l in lines]
    return [l[0] for (i, l) in enumerate(lines)
            if max_line_length > line_lengths[i] > min_line_length and
            precisions[i] > min_precision]


def hough_lines(source_image, min_points=0.075, min_line_length=0.2, max_line_gap=0.2):
    """Hough line detection.

    Args:
        source_image: An OpenCV Image.
        min_points: Float, minimum number of points that must be detected on a line.
            Specified as a percentage of the source image diagonal (0-1).
        min_line_length: Minimum line size. Specified as a percentage of the
            source image diagonal (0-1).
        max_line_gap: Maximum gap between segments of the same line. Specified as
            a percentage of the source image diagonal (0-1).
    Returns:
        Array of line endpoints tuples (x1, y1, x2, y2).
    """
    height, width = source_image.shape[:2]
    size = math.sqrt(height ** 2 + width ** 2)
    lines = cv2.HoughLinesP(
        source_image,
        1,
        np.pi / 180,
        math.ceil(min_points * size),
        minLineLength=math.ceil(min_line_length * size),
        maxLineGap=math.ceil(max_line_gap * size))
    return [l[0] for l in lines] if lines is not None else []


def canny_edges(image, thresholding_sigma=0.33):
    """Detects edges in an image.

    Uses Canny edge detection method. Bases thresholding on
    image statistics combined with the provided sigma.

    Args:
        image: An OpenCV Image.
        thresholding_sigma: Float. Higher values will detect more edges.

    Returns:
        Image with edges marked.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - thresholding_sigma) * v))
    upper = int(min(255, (1.0 + thresholding_sigma) * v))
    edged_image = cv2.Canny(image, lower, upper)
    return edged_image
