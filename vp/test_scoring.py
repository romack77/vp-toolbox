import unittest

from vp import scoring


class TestHorizonError(unittest.TestCase):

    def test_perfect_fit(self):
        self.assertEqual(
            scoring.horizon_error((1, 1), (1, 1), (100, 100)),
            0)
        self.assertEqual(
            scoring.horizon_error((2, 8), (2, 8), (100, 100)),
            0)

    def test_uneven(self):
        self.assertEqual(
            scoring.horizon_error((1, 0), (2, 0), (100, 100)),
            1)
        self.assertEqual(
            scoring.horizon_error((-1, 100), (2, 0), (100, 100)),
            2)

    def test_height_normalized(self):
        self.assertEqual(
            scoring.horizon_error((1, 0), (2, 0), (100, 50)),
            2)
        self.assertEqual(
            scoring.horizon_error((1, 0), (2, 0), (100, 10)),
            10)

    def test_out_of_image_bounds(self):
        self.assertEqual(
            scoring.horizon_error((1, 0), (1, 500), (100, 100)),
            5)

    def test_perpendicular(self):
        self.assertEqual(
            scoring.horizon_error((1, 0), (-1, 0), (100, 100)),
            2)


class TestVPDirectionError(unittest.TestCase):

    def test_same_vps(self):
        self.assertEqual(
            scoring.vp_direction_error(
                [(1, 1), (-2, -2)], [(1, 1), (-2, -2)], (200, 200)),
            [0, 0])

    def test_same_direction(self):
        self.assertEqual(
            scoring.vp_direction_error([(1, 1)], [(3, 3)], (200, 200)),
            [0])

    def no_vps(self):
        self.assertEqual(
            scoring.vp_direction_error([], [], (200, 200)),
            [])
        self.assertEqual(
            scoring.vp_direction_error([], [(1, 1)], (200, 200)),
            [])
        self.assertEqual(
            scoring.vp_direction_error([(1, 1)], [], (200, 200)),
            [None])

    def test_image_dims(self):
        self.assertEqual(
            scoring.vp_direction_error([(1, 1)], [(2, 2)], (200, 200)),
            [0])
        self.assertEqual(
            scoring.vp_direction_error([(1, 1)], [(2, 2)], (0, 0)),
            [0])
        self.assertEqual(
            round(scoring.vp_direction_error([
                (1, 1)], [(2, 2)], (200, 100))[0], 2),
            0.24)

    def test_error(self):
        self.assertEqual(
            scoring.vp_direction_error([
                (200, 200)], [(100, 200)], (200, 200)),
            [45])
        first, second = scoring.vp_direction_error([
                (200, 200), (100, 500)], [(200, 400), (100, 600)], (200, 200))
        self.assertEqual(round(first, 2), 26.57)
        self.assertEqual(second, 0)


class TestLocationAccuracyError(unittest.TestCase):

    def test_happy(self):
        ground_truth_vps = [(0, 0), (1, 1), (2, 2)]
        self.assertEqual(
            scoring.location_accuracy_error(
                ground_truth_vps,
                ground_truth_vps),
            0)
        self.assertEqual(
            scoring.location_accuracy_error(
                ground_truth_vps,
                [(0, 0)]),
            0)
        self.assertEqual(
            scoring.location_accuracy_error(
                ground_truth_vps,
                ground_truth_vps + [(10, 10)]),
            0)
        self.assertEqual(
            round(scoring.location_accuracy_error(
                ground_truth_vps,
                [(10, 0)]), 1),
            2.3)
        self.assertEqual(
            round(scoring.location_accuracy_error(
                ground_truth_vps,
                [(10, 0), (11, 1)]), 1),
            2.3)
        self.assertEqual(
            round(scoring.location_accuracy_error(
                ground_truth_vps,
                [(1000, 0)]), 1),
            6.9)


class TestNumModelDetectionError(unittest.TestCase):

    def test_happy(self):
        ground_truth_vps = [(0, 0), (1, 1), (2, 2)]
        self.assertEqual(
            scoring.num_model_detection_error(
                ground_truth_vps,
                ground_truth_vps),
            0)

        self.assertEqual(
            scoring.num_model_detection_error(
                ground_truth_vps,
                [(10, 10)]),
            -2)

        self.assertEqual(
            scoring.num_model_detection_error(
                ground_truth_vps,
                ground_truth_vps + [(3, 3)]),
            1)
