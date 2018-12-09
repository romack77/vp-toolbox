import cv2
from vp import geom_tools

# Based on "The Best of Metro Colors" https://www.color-hex.com/color-palette/861
DRAW_PALETTE = [
    (0, 171, 169), (255, 0, 151), (162, 0, 255), (27, 161, 226), (240, 150, 9),
    (0, 102, 101), (153, 0, 90), (97, 0, 153), (16, 96, 135), (144, 90, 5),
    (102, 204, 203), (255, 102, 192), (199, 102, 255), (118, 198, 237), (246, 192, 107),
]


def draw_lines(lines, dest_image, color=(0, 0, 255), thickness=2):
    for x1, y1, x2, y2 in lines:
        cv2.line(dest_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def draw_points(points, dest_image, color=(0, 255, 0), size=1):
    for point in points:
        cv2.circle(dest_image, (int(point[0]), int(point[1])), size, color, -1)


def draw_line_groups(line_groups, dest_image, color_options=DRAW_PALETTE):
    for i, lines in enumerate(line_groups):
        if i > len(color_options):
            print("Warning: There are fewer color options than groups.")
            color = (0, 0, 0)
        else:
            color = DRAW_PALETTE[i]
        draw_lines(lines, dest_image, color=color)


def draw_fitted_point(point, dest_image, color=(0, 255, 0), size=1):
    """Draws a point on an image, or its border if it would be "off screen".

    Args:
        point: Point tuple.
        dest_image: OpenCV image.
        color: RGB color tuple.
        size: Integer, point radius.
    """
    x, y = point
    h, w = dest_image.shape[:2]
    if not (0 <= x <= w and 0 <= y <= h):
        # Image bounds exceeded - find the correct point on the border.
        angle = geom_tools.get_line_angle((w / 2, h / 2, x, y))
        rect = [(0, 0), (w, h)]
        point = geom_tools.find_point_on_rect_border(rect, angle)
    draw_points([point], dest_image, color=color, size=size)
