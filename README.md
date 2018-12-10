# VP Toolbox

Vanishing point detection toolbox
([Wikipedia: Vanishing Points](https://en.wikipedia.org/wiki/Vanishing_point)).
This uses some well known methods to find vanishing points and horizon lines.
These have a variety of uses, especially when a 3D understanding of a 2D
scene is required. The accuracy is not perfect and the speed is not
real time (or optimised).
 
This also provides a decent toolbox to
experiment with, including helpers to load datasets, compare methods, and
score results. Loaders are included for the York Urban dataset [1] and
the Toulouse dataset [2].

## The demo methods
#### Synopsis of approach
The most accurate method shown uses line detection (LSD), followed by RANSAC
to find vanishing point hypotheses for those lines, and finally
J-linkage [6] to cluster the lines which had similar responses to the
hypotheses. The consistency measure, which defines the quality of a VP
for a given line, is the distance of a segment's endpoint to a line
connecting its midpoint with a hypothetical VP, as described in [4].

#### Key assumptions
This makes no "Manhattan world assumption"
(under which images would be assumed to have 3 dominant vanishing points on a
regular grid). Instead, a RANSAC variant ([6] or [3]) tries to detect the
underlying number of models. There is no orthogonality restriction or
correction on VPs. There is also no reliance on known camera parameters.

#### Results

Detection is scored against the popular York Urban dataset. It comprises 102
low resolution (640 x 480) images of strongly architectural environments,
as well as ground truth vanishing points [1].
It presents a relatively easy, and fast, dataset for this task. Parameters
were tuned against the first ten images.

Detection is scored with a common measure of horizon error as well as angular
deviation of VPs, both described in [5]. On the York Urban dataset, the best
results were 74% of horizons detected within a quarter-image distance, and 74%
of ground truth VPs detected within 5 degrees. These generous measures were
reported as 74-94% for horizons in a comparison of 6 approaches in [5]. [5]
also reported finding 99% of ground truth VPs within 5 degrees.

The comparison is not rigorously controlled but may provide a rough benchmark.
It is also flawed because these approaches often make a Manhattan
assumption, or select VPs that maximize orthogonality, which is not done here.

Median detection time was 45 seconds, and max was a whopping 28 minutes.
Though this is unoptimized, parameter selection does already offer
some control over speed. For instance, reducing the number of lines detected,
or the number of RANSAC iterations, reduces the amount of work.


Results can be seen in this IPython notebook:
[score_dataset.ipynb](https://github.com/romack77/vp-toolbox/blob/master/notebooks/score_dataset.ipynb).

## Usage

```
import cv2
from vp import vp_finder

image = cv2.imread('path/to/my/image.jpg')
vp_to_lines, outlier_lines = vp_finder.find_vanishing_points_in_image(image)
for vp in vp_to_lines:
    print('Vanishing point found at: (%s, %s)' % (vp[0], vp[1])
```
Richer examples can be found in this IPython notebook:
[score_dataset.ipynb](https://github.com/romack77/vp-toolbox/blob/master/notebooks/score_dataset.ipynb).

## Installation

Requires Python 3 and Make.

To install:
```
make
```

To run tests:
```
make test
```

## License

This project is licensed under the MIT License - see the
[LICENSE.txt](LICENSE.txt) file for details.

## References

For the York Urban dataset:

1\. Denis P., Elder J. H. & Estrada F. (2008).
[Efficient Edge-Based Methods for Estimating Manhattan Frames in Urban Imagery.](http://elderlab.apps01.yorku.ca/wp-content/uploads/2017/02/DenisElderEstradaECCV08.pdf)
European Conference on Computer Vision, 2 (5303), 197-210.

For the Toulouse dataset:

2\. V. Angladon, S. Gasparini and V. Charvillat (2015).
[The Toulouse Vanishing Points Dataset.](https://hal.archives-ouvertes.fr/hal-01130447v1)
Proceedings of the 6th ACM Multimedia Systems Conference (MMSys '15), Mar 2015,
Portland, OR, United States. 2015.
[With the associated slides of the oral presentation.](http://ubee.enseeiht.fr/tvpd/slides/src/)

For the multi-model RANSAC variant:

3\. Zhang, W., KoseckÃ¡, J.:
[Nonparametric estimation of multiple structures with outliers.](Nonparametric estimation of multiple structures with outliers.)
In: Vidal, R., Heyden, A., Ma, Y. (eds.) WDV 2006. LNCS, vol. 4358, pp. 60-74.
Springer, Heidelberg (2006)

For the Segment-VP consistency measure:

4\. H. Wildenauer and M. Vincze.
[Vanishing point detection in complex man-made worlds.](https://ieeexplore.ieee.org/abstract/document/4362845)
In ICIAP, 2007.

For an empirical comparison of several approaches:

5\. Kluger F., Ackermann H., Yang M.Y., Rosenhahn B. (2017)
[Deep Learning for Vanishing Point Detection Using an Inverse Gnomonic Projection.](https://link.springer.com/chapter/10.1007/978-3-319-66709-6_2)
In: Roth V., Vetter T. (eds) Pattern Recognition. GCPR 2017.
Lecture Notes in Computer Science, vol 10496. Springer, Cham

For the J-linkage variant:

6\. Toldo, R., & Fusiello, A. (2008, October).
[Robust multiple structures estimation with j-linkage.](https://link.springer.com/chapter/10.1007/978-3-540-88682-2_41)
In European conference on computer vision (pp. 537-547). Springer, Berlin, Heidelberg.

A very similar approach:

7\.
Tardif, J. P. (2009, September).
[Non-iterative approach for fast and accurate vanishing point detection.](https://ieeexplore.ieee.org/abstract/document/5459328)
In Computer Vision, 2009 IEEE 12th International Conference on (pp. 1250-1257). IEEE.
