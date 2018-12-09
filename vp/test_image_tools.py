import unittest

from vp import image_tools


class TestCalculateBorderSize(unittest.TestCase):

    def test_no_border(self):
        self.assertEqual(
            image_tools._calculate_border_size(100, 100, [(0, 0), (100, 100)]),
            (0, 0))

    def test_no_points(self):
        self.assertEqual(
            image_tools._calculate_border_size(100, 100, []),
            (0, 0))

    def test_border(self):
        self.assertEqual(
            image_tools._calculate_border_size(100, 100, [(150, 300)]),
            (25, 100))

    def test_border_non_square_image(self):
        self.assertEqual(
            image_tools._calculate_border_size(100, 200, [(150, 300)]),
            (25, 50))

    def test_max_border_not_hit(self):
        self.assertEqual(
            image_tools._calculate_border_size(
                100, 100, [(200, 200)], max_border_size=100),
            (50, 50))

    def test_max_border_hit(self):
        self.assertEqual(
            image_tools._calculate_border_size(
                100, 100, [(200, 200)], max_border_size=25),
            (25, 25))

    def test_max_border_zero(self):
        self.assertEqual(
            image_tools._calculate_border_size(
                100, 100, [(200, 200)], max_border_size=0),
            (0, 0))

    def test_negative_points(self):
        self.assertEqual(
            image_tools._calculate_border_size(
                100, 100, [(-100, -100), (200, 200)]),
            (100, 100))
        self.assertEqual(
            image_tools._calculate_border_size(
                100, 100, [(-100, 300)]),
            (50, 100))
