import unittest

from vp import horizon_finder


class TestFindHorizon(unittest.TestCase):

    def test_happy(self):
        self.assertEqual(
            horizon_finder.find_horizon(
                {(0, 0), (10, 0), (0, 100)}, (0, 0)),
            (0, 0))
        slope, intercept = horizon_finder.find_horizon(
            {(0, 10), (10, 10), (0, 100)}, (0, 0))
        self.assertAlmostEqual(slope, 0)
        self.assertAlmostEqual(intercept, 10)
        slope, intercept = horizon_finder.find_horizon(
                {(0, 0), (1, 1), (0, 100)}, (0, 0))
        self.assertAlmostEqual(slope, 0)
        self.assertAlmostEqual(intercept, 0.5)

    def test_no_vps(self):
        self.assertEqual(
            horizon_finder.find_horizon(set([]), (0, 0)), (0, 0))

    def test_no_vertical_vp(self):
        self.assertEqual(
            horizon_finder.find_horizon(
                {(0, 0), (10, 0)}, (0, 0)),
            (0, 0))


class TestChooseVerticalVanishingPoint(unittest.TestCase):

    def test_happy(self):
        self.assertEqual(
            horizon_finder.choose_vertical_vanishing_point([
                (1, 0), (10, 1), (1, 1)], (0, 0)),
            (1, 1))
        self.assertEqual(
            horizon_finder.choose_vertical_vanishing_point([
                (1, 0), (10, 1), (1, 1), (0, 30)], (0, 0)),
            (0, 30))

    def test_negative_vps(self):
        self.assertEqual(
            horizon_finder.choose_vertical_vanishing_point([
                (1, 0), (10, 1), (1, 1), (0, -30)], (0, 0)),
            (0, -30))

    def test_no_vertical_vps(self):
        self.assertIsNone(
            horizon_finder.choose_vertical_vanishing_point([
                (1, 0), (10, 1)], (0, 0)))

    def test_no_vps(self):
        self.assertIsNone(
            horizon_finder.choose_vertical_vanishing_point([], (0, 0)))

    def test_principal_point_adjustment(self):
        self.assertIsNone(
            horizon_finder.choose_vertical_vanishing_point([
                (0, 100), (100, 100)], (100, 100)))
        self.assertEqual(
            horizon_finder.choose_vertical_vanishing_point([
                (100, 100), (101, 101), (100, 90)], (100, 10)),
            (101, 101))


class TestFindBestIntercept(unittest.TestCase):

    def test_happy(self):
        self.assertEqual(
            horizon_finder._find_best_intercept([
                (0, 10), (1, 11), (2, 12)], 1),
            10)
        self.assertEqual(
            horizon_finder._find_best_intercept([
                (0, 10), (1, 11), (2, 12), (3, 20)], 1),
            11.75)

    def test_no_points(self):
        with self.assertRaises(ValueError):
            horizon_finder._find_best_intercept([], 1)

    def test_zero_slope(self):
        self.assertAlmostEqual(
            horizon_finder._find_best_intercept([
                (1, 0), (2, 5), (3, 10)], 0),
            5)
