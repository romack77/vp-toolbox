from vp import horizon_finder


class Dataset(object):
    """Stores image dataset information."""

    def __init__(self, image_paths, image_dims, image_gt_vps, image_gt_segments,
                 image_gt_vertical_vp=None, image_gt_horizon=None):
        """Constructor.

        Args:
            image_paths: List of string image paths.
            image_dims: Tuple of (width, height) of images, in pixels.
            image_gt_vps: List of lists of ground truth VP point tuples,
                one per image.
            image_gt_segments: List of lists of ground truth line segments,
                (x1, y1, x2, y2) per VP and per image.
            image_gt_vertical_vp: Optional list with the vertical VP tuple for
                each image, or None. Calculated if not supplied.
            image_gt_horizon: Optional list with the vertical VP tuple for
                each image, or None. Calculated if not supplied.
        """
        self.image_paths = image_paths
        self.image_dims = image_dims
        self.image_gt_vps = image_gt_vps
        self.image_gt_segments = image_gt_segments
        self.image_gt_vertical_vp = image_gt_vertical_vp
        self.image_gt_horizon = image_gt_horizon
        principal_point = (self.image_dims[0] // 2, self.image_dims[1] // 2)

        if self.image_gt_vertical_vp is None:
            self.image_gt_vertical_vp = []
            for gt_vps in image_gt_vps:
                vertical_vp = horizon_finder.choose_vertical_vanishing_point(
                    gt_vps, principal_point)
                self.image_gt_vertical_vp.append(vertical_vp)

        if self.image_gt_horizon is None:
            self.image_gt_horizon = []
            for i, gt_vps in enumerate(image_gt_vps):
                vertical_vp = self.image_gt_vertical_vp[i]
                self.image_gt_horizon.append(horizon_finder.find_horizon(
                    gt_vps, principal_point, vertical_vanishing_point=vertical_vp))

    def with_mask(self, indices):
        """Returns a new Dataset with a masked subset of images.

        Args:
            indices: Set of integer indices to keep.

        Returns:
            Dataset instance.
        """
        return Dataset(
            [self.image_paths[i] for i in indices],
            self.image_dims,
            [self.image_gt_vps[i] for i in indices],
            [self.image_gt_segments[i] for i in indices],
            [self.image_gt_vertical_vp[i] for i in indices],
            [self.image_gt_horizon[i] for i in indices])
