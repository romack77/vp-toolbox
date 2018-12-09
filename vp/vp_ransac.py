import numpy as np

import ransac
from vp import geom_tools

    
class SegmentVPModel(ransac.Model):
    """Fits a vanishing point to a set of lines.

    The error measure is segment_midpoint_vp_error().
    """

    def fit(self, data):
        best_vp_point = choose_best_vp_by_max_error(data)
        if best_vp_point is None:
            raise ransac.DegenerateModelException()
        return best_vp_point

    def predict(self, xs, fit):
        return None

    def get_residuals(self, data, best_vp_point):
        residuals = np.array([
            segment_midpoint_vp_error(segment, best_vp_point)
            for segment in data])
        return residuals


def choose_best_vp_by_max_error(lines):
    """Finds the vanishing point that minimizes error among lines.

    Candidates vps are intersections of the lines.
    The error measure is segment_midpoint_vp_error().

    Returns:
        Vanishing point tuple, or None if detection fails.
    """
    intersections = geom_tools.find_all_intersections(lines)
    best_vp = None
    best_vp_error = None
    for point in intersections:
        error = max([segment_midpoint_vp_error(l, point) for l in lines])
        if best_vp is None or error < best_vp_error:
            best_vp = point
            best_vp_error = error
    return best_vp


def segment_midpoint_vp_error(segment, vp):
    """Calculates an error term for a segment's assignment to a vanishing point.

    This uses the distance from a segment endpoint to a line passing through
    the vanishing point and the segment's midpoint.

    Args:
        segment: Segment tuple in (x1, y1, x2, y2) format.
        vp: Tuple of vanishing point, i.e. (vx, vy).

    Returns:
        Float, error amount.
    """
    sx1, sy1, sx2, sy2 = segment
    vx, vy = vp
    cx, cy = geom_tools.get_midpoint((sx1, sy1), (sx2, sy2))
    vp_midpoint_line = (cx, cy, vx, vy)
    return geom_tools.point_to_line_dist((sx1, sy1), vp_midpoint_line)
