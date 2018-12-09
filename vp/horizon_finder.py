import math

import numpy as np

from vp import geom_tools


def find_horizon(vanishing_points, principal_point, vertical_vanishing_point=None):
    """Detects the horizon line.

    Args:
        vanishing_points: Set of vanishing point tuples.
        principal_point: Tuple, camera center on the projection.
        vertical_vanishing_point: Vertical vanishing point tuple.
            Supply if known, otherwise this will be calculated from
            the vanishing point set.

    Returns:
        Horizon line in (slope, intercept) format.
    """
    if len(vanishing_points) == 0 and vertical_vanishing_point is None:
        # In total failure cases, assume a flat, image-center horizon.
        return 0, principal_point[1]

    if vertical_vanishing_point is None:
        vertical_vanishing_point = choose_vertical_vanishing_point(
            vanishing_points, principal_point)

    if vertical_vanishing_point is not None:
        # Horizon line will be perpendicular to the line from the principal point to
        # the vertical vanishing point.
        vertical_vp_line = (
            principal_point[0], principal_point[1],
            vertical_vanishing_point[0], vertical_vanishing_point[1])
        try:
            # A perpendicular line has a negative reciprocal slope.
            horizon_line_slope = -1 / geom_tools.get_line_slope(vertical_vp_line)
        except ZeroDivisionError:
            horizon_line_slope = 0
        horizontal_vps = set(vanishing_points).difference({vertical_vanishing_point})
    else:
        # If we can't find a vertical vanishing point, assume a flat horizon line.
        horizon_line_slope = 0
        horizontal_vps = set(vanishing_points)

    if not horizontal_vps:
        # If we only found a vertical VP, used the predicted slope and
        # assume an image-center horizon.
        return horizon_line_slope, principal_point[1]

    intercept = _find_best_intercept(horizontal_vps, horizon_line_slope)
    if np.isnan(intercept):
        intercept = principal_point[1]
    return horizon_line_slope, intercept


def choose_vertical_vanishing_point(vanishing_points, principal_point):
    """Looks for a vertical vanishing point.

    This looks for vertically oriented vanishing points,
    and returns the most distant of them.

    Args:
        vanishing_points: Set of vanishing points.
        principal_point: Tuple, camera center on the projection.

    Returns:
        Point tuple, or None if detection fails.
    """
    best_vertical_vp = None
    best_vertical_vp_mag = None
    for vp in vanishing_points:
        vx, vy = (vp[0] - principal_point[0], vp[1] - principal_point[1])
        angle = math.degrees(math.atan2(vy, vx))
        if 45 <= angle <= 135 or -135 <= angle <= -45:
            magnitude = math.hypot(vx, vy)
            if magnitude > principal_point[1] * 2 and (
                    best_vertical_vp is None or magnitude > best_vertical_vp_mag):
                best_vertical_vp = vp
                best_vertical_vp_mag = magnitude
    return best_vertical_vp


def _find_best_intercept(points, slope):
    """Fits an intercept to points given a fixed slope.

    Args:
        points: List of point tuples.
        slope: Float, line slope value.

    Returns:
        Float, intercept value.
    """
    # TODO: Use a weighted least squares fit, where the weight of each
    # detected horizontal vanishing point equals the number of corresponding
    # lines, OR the trace of covariance.
    coeffs = np.vstack(np.ones(len(points)))
    ys = np.array([y - slope * x for (x, y) in points], dtype=np.float64)
    return np.linalg.lstsq(coeffs, ys, rcond=None)[0][0]
