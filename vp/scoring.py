import math

from vp import geom_tools


def horizon_error(ground_truth_horizon, detected_horizon, image_dims):
    """Calculates error in a detected horizon.

    This measures the max distance between the detected horizon line and
    the ground truth horizon line, within the image's x-axis, and
    normalized by image height.

    Args:
        ground_truth_horizon: Tuple with (slope, intercept) for the GT horizon line.
        detected_horizon: Tuple with (slope, intercept) for the detected horizon line.
        image_dims: Tuple of integers, (width, height) of the image, in pixels.

    Returns:
        Float, or None if a horizon is missing altogether.
    """
    if ground_truth_horizon is None or detected_horizon is None:
        return None

    def gt(x):
        return ground_truth_horizon[0] * x + ground_truth_horizon[1]

    def dt(x):
        return detected_horizon[0] * x + detected_horizon[1]

    width, height = image_dims
    return max(abs(gt(0) - dt(0)), abs(gt(width) - dt(width))) / height


def vp_direction_error(ground_truth_vps, detected_vps, image_dims):
    """Measures error in direction from center of detected vanishing points.

    Each detected VP is matched with its closest unclaimed ground truth VP.

    Args:
        ground_truth_vps: List of ground truth VP point tuples.
        detected_vps: List of detected VP point tuples.
        image_dims: Tuple of integers, (width, height) of the image, in pixels.

    Returns:
        List with float degrees of error for each ground truth VP.
        Error is None for missing VPs.
    """
    principal_point = (image_dims[0] // 2, image_dims[1] // 2)
    point_pair_dists = []
    for gt_vp in ground_truth_vps:
        for dt_vp in detected_vps:
            gt_angle = geom_tools.get_line_angle((
                principal_point[0], principal_point[1], gt_vp[0], gt_vp[1]))
            dt_angle = geom_tools.get_line_angle((
                principal_point[0], principal_point[1], dt_vp[0], dt_vp[1]))
            angle_diff = 180 - abs(abs(gt_angle - dt_angle) - 180)
            point_pair_dists.append((angle_diff, gt_vp, dt_vp))

    point_pair_dists = sorted(point_pair_dists, key=lambda k: k[0])

    gt_vp_to_error = {}
    seen_dt_vps = set()
    for distance, gt_vp, dt_vp in point_pair_dists:
        if gt_vp in gt_vp_to_error or dt_vp in seen_dt_vps:
            continue
        gt_vp_to_error[gt_vp] = distance
        seen_dt_vps.add(dt_vp)

    return [gt_vp_to_error.get(gt, None) for gt in ground_truth_vps]


def location_accuracy_error(ground_truth_vps, detected_vps):
    """Measures average error in the location of detected vanishing points.

    "Missed" or "extra" VPs do not count against the score.
    Based on log distance of detected vp from ground truth vp.

    Args:
        ground_truth_vps: List of ground truth VP point tuples.
        detected_vps: List of detected VP point tuples.

    Returns:
        Float, error.
    """
    if len(ground_truth_vps) == 0 or len(detected_vps) == 0:
        return 0

    point_pair_dists = []
    for gt_vp in ground_truth_vps:
        for dt_vp in detected_vps:
            distance = geom_tools.point_to_point_dist(gt_vp, dt_vp)
            point_pair_dists.append((distance, gt_vp, dt_vp))

    sorted(point_pair_dists, key=lambda k: k[0])

    seen_gt_vps = set()
    seen_dt_vps = set()
    total_error = 0
    for distance, gt_vp, dt_vp in point_pair_dists:
        if gt_vp in seen_gt_vps or dt_vp in seen_dt_vps:
            continue
        seen_gt_vps.add(gt_vp)
        seen_dt_vps.add(dt_vp)
        if distance > 0:
            total_error += math.log(distance)

    return total_error / min(len(detected_vps), len(ground_truth_vps))


def num_model_detection_error(ground_truth_vps, detected_vps):
    """Measures error in the number of detected vanishing points.

    Returns:
        Integer, positive when there are too many VPs, negative
            when there are too few.
    """
    return len(detected_vps) - len(ground_truth_vps)
