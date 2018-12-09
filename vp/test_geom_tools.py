import unittest

import numpy as np

from vp import geom_tools


class TestFindIntersection(unittest.TestCase):

    def test_intersection(self):
        self.assertEqual(
            geom_tools.find_intersection((0, 0, 1, 1), (0, 10, 10, 0)),
            (5, 5))
        x, y = geom_tools.find_intersection((80, 159, 403, 346), (63, 80, 390, 276))
        self.assertEqual(round(x), 3446)
        self.assertEqual(round(y), 2108)

    def test_parallel(self):
        self.assertEqual(
            geom_tools.find_intersection((0, 0, 1, 1), (0, 2, 1, 3)),
            None)

    def test_overlapping(self):
        self.assertEqual(
            geom_tools.find_intersection((0, 0, 1, 1), (0, 0, 1, 1)),
            None)
        self.assertEqual(
            geom_tools.find_intersection((0, 0, 1, 1), (2, 2, 3, 3)),
            None)


class TestFindAllIntersections(unittest.TestCase):

    def test_happy(self):
        intersections = geom_tools.find_all_intersections(
            [(0, 0, 1, 1), (0, 10, 10, 0), (0, 2, 1, 3)])
        self.assertIn((5, 5), intersections)
        self.assertIn((4, 6), intersections)
        self.assertEqual(len(intersections), 2)

    def test_no_lines(self):
        self.assertEqual(geom_tools.find_all_intersections([]), [])

    def test_no_intersections(self):
        self.assertEqual(
            geom_tools.find_all_intersections([(0, 0, 1, 1), (0, 2, 1, 3)]),
            [])


class TestPointToLineDistance(unittest.TestCase):

    def test_happy(self):
        distance = geom_tools.point_to_line_dist((0, 5), (0, 0, 1, 1))
        self.assertEqual(round(distance, 2), 3.54)

    def test_zero_distance(self):
        self.assertEqual(geom_tools.point_to_line_dist((2, 2), (0, 0, 1, 1)), 0)


class TestPointToPointDistance(unittest.TestCase):

    def test_happy(self):
        self.assertEqual(geom_tools.point_to_point_dist((0, 0), (6, 8)), 10)

    def test_vertical(self):
        self.assertEqual(geom_tools.point_to_point_dist((0, 0), (0, 1)), 1)

    def test_horizontal(self):
        self.assertEqual(geom_tools.point_to_point_dist((0, 0), (1, 0)), 1)

    def test_negative(self):
        self.assertEqual(geom_tools.point_to_point_dist((-1, -1), (-7, -9)), 10)

    def test_zero_distance(self):
        self.assertEqual(geom_tools.point_to_point_dist((1, 1), (1, 1)), 0)


class TestGetMidpoint(unittest.TestCase):

    def test_happy(self):
        self.assertEqual(
            geom_tools.get_midpoint((0, 0), (2, 2)),
            (1, 1))
        self.assertEqual(
            geom_tools.get_midpoint((0, 0), (1, 1)),
            (0.5, 0.5))
        self.assertEqual(
            geom_tools.get_midpoint((0, 0), (5, 1)),
            (2.5, 0.5))

    def test_negative(self):
        self.assertEqual(
            geom_tools.get_midpoint((-1, -1), (-3, -3)),
            (-2, -2))

    def test_zero_distance(self):
        self.assertEqual(
            geom_tools.get_midpoint((1, 1), (1, 1)),
            (1, 1))


class TestGetLineAngle(unittest.TestCase):

    def test_vertical_line(self):
        self.assertEqual(
            geom_tools.get_line_angle((0, 0, 0, 1)),
            90)
        self.assertEqual(
            geom_tools.get_line_angle((0, 0, 0, 10)),
            90)
        self.assertEqual(
            geom_tools.get_line_angle((10, 10, 10, 100)),
            90)
        self.assertEqual(
            geom_tools.get_line_angle((0, 0, 0, -1)),
            270)

    def test_horizontal_line(self):
        self.assertEqual(
            geom_tools.get_line_angle((0, 0, 1, 0)),
            0)
        self.assertEqual(
            geom_tools.get_line_angle((0, 0, 10, 0)),
            0)
        self.assertEqual(
            geom_tools.get_line_angle((10, 10, 100, 10)),
            0)
        self.assertEqual(
            geom_tools.get_line_angle((0, 0, -1, 0)),
            180)

    def test_angled_line(self):
        self.assertEqual(
            geom_tools.get_line_angle((0, 0, 1, 1)),
            45)
        self.assertEqual(
            geom_tools.get_line_angle((1, 1, 10, 10)),
            45)
        self.assertEqual(
            geom_tools.get_line_angle((0, 0, -1, -1)),
            225)


class TestGetLineSlope(unittest.TestCase):

    def test_happy(self):
        self.assertEqual(
            geom_tools.get_line_slope((0, 0, 1, 1)),
            1)
        self.assertEqual(
            geom_tools.get_line_slope((0, 0, 10, 10)),
            1)
        self.assertEqual(
            geom_tools.get_line_slope((2, 2, 3, 3)),
            1)
        self.assertEqual(
            geom_tools.get_line_slope((0, 0, 1, 2)),
            2)

    def test_vertical_line(self):
        with self.assertRaises(ZeroDivisionError):
            geom_tools.get_line_slope((0, 0, 0, 1))

    def test_horizontal_line(self):
        self.assertEqual(
            geom_tools.get_line_slope((0, 0, 1, 0)),
            0)


class TestFindBoundingPoints(unittest.TestCase):

    def test_no_points(self):
        self.assertEqual(
            geom_tools.find_bounding_points([]),
            (None, None))

    def test_points(self):
        self.assertEqual(
            geom_tools.find_bounding_points([(0, 0), (100, 100)]),
            ([0, 0], [100, 100]))
        self.assertEqual(
            geom_tools.find_bounding_points([(100, 0), (0, 100)]),
            ([0, 0], [100, 100]))

    def test_negative_points(self):
        self.assertEqual(
            geom_tools.find_bounding_points([(-100, -100), (-200, -200)]),
            ([-200, -200], [-100, -100]))


class TestFindPointOnRectBorder(unittest.TestCase):

    def test_rect_corners(self):
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 2)], 45),
            (4, 2))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 2)], 135),
            (0, 2))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 2)], 225),
            (0, 0))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 2)], 315),
            (4, 0))

    def test_rect_sides(self):
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 2)], 90),
            (2, 2))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 2)], 180),
            (0, 1))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 2)], 270),
            (2, 0))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 2)], 360),
            (4, 1))

    def test_square_corners(self):
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 4)], 45),
            (4, 4))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 4)], 135),
            (0, 4))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 4)], 225),
            (0, 0))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 4)], 315),
            (4, 0))

    def test_square_sides(self):
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 4)], 90),
            (2, 4))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 4)], 180),
            (0, 2))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 4)], 270),
            (2, 0))
        self.assertEqual(
            geom_tools.find_point_on_rect_border([(0, 0), (4, 4)], 360),
            (4, 2))


class TestFindPointClusterAverage(unittest.TestCase):

    def test_happy(self):
        self.assertEqual(
            geom_tools.find_point_cluster_average(
                [(0, 0), (0, 10), (10, 10), (10, 0)]),
            (5, 5))

    def test_no_points(self):
        self.assertEqual(
            geom_tools.find_point_cluster_average([]),
            None)


class TestFindLargestIntersectionCluster(unittest.TestCase):

    def test_no_lines(self):
        self.assertEqual(
            geom_tools.find_largest_intersection_cluster([]),
            [])

    def test_no_intersections(self):
        self.assertEqual(
            geom_tools.find_largest_intersection_cluster(
                [(0, 0, 1, 1)]),
            [])

    def test_one_intersection(self):
        cluster = geom_tools.find_largest_intersection_cluster(
            [(0, 0, 1, 1), (0, 10, 10, 0)])
        self.assertIn((5, 5), cluster)
        self.assertEqual(len(cluster), 1)
