import cv2

from vp import geom_tools


def border_image(image, left_border, top_border, color=(255, 255, 255)):
    """Applies a border to an image.

    Args:
        image: OpenCV image.
        left_border: Size, in pixels, of each left/right border strip.
        top_border: Size, in pixels, of each top/bottom border strip.
        color: Color tuple for the border.

    Returns:
        New bordered image.
    """
    if left_border < 0 or top_border < 0:
        raise ValueError('Negative borders are unsupported.')
    return cv2.copyMakeBorder(
        image,
        int(top_border), int(top_border), int(left_border), int(left_border),
        cv2.BORDER_CONSTANT, value=color)


def border_image_to_accommodate_points(
        image, points, color=(255, 255, 255), max_border_size=1000):
    """Adds border to an image until the given points fit.

    Args:
        image: OpenCV image.
        points: List of data points.
        color: Color tuple for the border.
        max_border_size: Optional limit on the max pixel size of the border.

    Returns:
        Tuple of:
            1) new bordered image.
            2) The original data points, shifted to match the image's translation
                to the center of the new bordered image.
            3) Tuple of shift factor. Any points calculated against the original
                image must apply this translation to maintain their position relative
                to the image.
    """
    if not points:
        return image, points, None
    image_height, image_width = image.shape[:2]
    left_border, top_border = _calculate_border_size(
        image_width, image_height, points, max_border_size=max_border_size)
    new_image = border_image(image, left_border, top_border, color=color)
    shifted_points = [(x + left_border, y + top_border)
                      for (x, y) in points]
    return new_image, shifted_points, (left_border, top_border)


def _calculate_border_size(image_width, image_height, points, max_border_size=None):
    """Finds a border size that will contain the given points.

    Args:
        image_width: Integer image width.
        image_height: Integer image height.
        points: List of point tuples.
        max_border_size: Optional, constrain the border to this size.

    Returns:
        Tuple of left/right border size, top/bottom border size.
        Size of one side of the border (e.g. top, not top + bottom).
    """
    if not points:
        return 0, 0
    min_bounds, max_bounds = geom_tools.find_bounding_points(
        points + [(0, 0), (image_width, image_height)])

    desired_width = max_bounds[0] - min_bounds[0]
    desired_height = max_bounds[1] - min_bounds[1]

    width_adjustment = max(desired_width - image_width, 0)
    height_adjustment = max(desired_height - image_height, 0)

    if max_border_size is not None:
        width_adjustment = min(width_adjustment, max_border_size * 2)
        height_adjustment = min(height_adjustment, max_border_size * 2)
    return width_adjustment / 2, height_adjustment / 2
