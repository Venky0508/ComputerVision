import cv2 as cv
import numpy as np
# from typing import Optional


class HoughLineDetector:
    def __init__(
        self,
        image_shape: tuple[int, int],
        min_angle: float = 0,
        max_angle: float = np.pi,
        angle_spacing: float = np.pi / 180,
        offset_spacing: float = 2.0,
    ):
        h, w = image_shape

        # We'll use the center of the image as our "origin" for the coordinate system for lines.
        self.origin_xy = np.array([w / 2, h / 2])

        # Largest possible offset is the distance from the origin to the corner of the image.
        max_offset = np.sqrt(h**2 + w**2) / 2

        num_offsets = int(np.ceil(max_offset / (offset_spacing or 2)))
        num_angles = int(np.ceil(np.pi / angle_spacing))

        # Create a coordinate system of offsets (rho) and angles (theta) for the parameter space.
        self.offsets = np.linspace(-max_offset, max_offset, num_offsets)
        # We don't want to include the same angle twice (e.g. both 0 and pi denote vertical lines),
        # so we'll create num_angles+1 values and exclude the last one.
        self.angles = np.linspace(0, np.pi, num_angles + 1)[:num_angles]

        self._min_angle = min_angle
        self._max_angle = max_angle

        # Precompute a 2xnum_angles array of cosines and sines of the angles.
        self._cos_sin = np.stack([np.cos(self.angles), np.sin(self.angles)], axis=0)

        # Initialize self.accumulator to be a 2D array of zeros with the same shape as the
        # parameter space. The value in accumulator[i,j] represents the 'votes' for a line with
        # rho = offsets[i] and theta = angles[j].
        self.accumulator = np.zeros(shape=(len(self.offsets), len(self.angles)), dtype=float)

    def clear(self):
        self.accumulator = np.zeros(shape=(len(self.offsets), len(self.angles)), dtype=float)

    def add_points_xy(self, xy: np.ndarray):
        """Given n points in the image's coordinate system as an array of shape (n, 2), increment
        the accumulator for all lines that pass through each point.

        The main idea is to loop over all (x,y) points and increment self.accumulator[i,j] for all
        lines that pass through the point (x,y).
        """
        # First, adjust the coordinate system such that xy are relative to origin_xy.
        xy = xy - self.origin_xy

        # We'll calculate a value for 'rho' once for each point (n) and for each theta (m). This
        # will be done using vectorized operations for speed. Using the fact that xy is (n,2) and
        # self._cos_sin is (2,m), we can do a matrix multiplication to get an (n,m) array of rho
        # values (based on the normal form of the line equation: rho = x*cos(theta) + y*sin(theta)).
        rhos = xy @ self._cos_sin

        # The equation for a line in normal form is already implemented in the 'rhos = ...' line
        # above. It provides a value for 'rho' once for each (x,y) point and for each angle. You
        # just need to figure out where and by how much to increment the accumulator. *This will
        # be the main time bottleneck* and will greatly benefit from vectorization.
        # raise NotImplementedError("your code here")
        # Normalize the rho values to be in range [0, len(self.offsets)-1].
        rho_indices = np.clip(
            np.round(
                ((rhos - self.offsets.min()) / (self.offsets.ptp()) * (len(self.offsets) - 1))
            ),
            0, len(self.offsets) - 1).astype(int)

        # Increment the accumulator for each point and angle
        for angle_idx, _ in enumerate(self.angles):
            for rho in rho_indices[:, angle_idx]:
                self.accumulator[rho, angle_idx] += 1

    def non_maximal_suppression(self, angle_range: float, offset_range: float) -> np.ndarray:
        """Do non-maximal suppression on the accumulator array. This means that for each cell in
        the accumulator which represents some line, compare it to other 'neighboring' lines. Two
        lines are considered 'neighbors' if their difference in angle is < angle_range and their
        difference in offset is < offset_range.

        Note that this requires a bit of care where 'angles' wrap around. To handle this,
        we'll use a temporary wrap-around padding on the accumulator.
        """
        h, w = self.accumulator.shape

        angle_spacing = self.angles[1] - self.angles[0]
        offset_spacing = self.offsets[1] - self.offsets[0]
        pad_angle = int(np.ceil(angle_range / angle_spacing))
        pad_offset = int(np.ceil(offset_range / offset_spacing))

        angle_mask = np.logical_and(self._min_angle <= self.angles, self.angles <= self._max_angle)
        padded_accumulator = self.accumulator * angle_mask[None, :]

        padded_accumulator = np.pad(
            padded_accumulator, ((pad_offset, pad_offset), (0, 0)), mode="constant"
        )
        padded_accumulator = np.pad(
            padded_accumulator, ((0, 0), (pad_angle, pad_angle)), mode="wrap"
        )

        # Create a stack of accumulators of size (num_offsets, num_angles, w_offset*w_angle),
        # where w_offset=2*pad_offset+1 and w_angle=2*pad_angle+1 are the window sizes. The stack
        # is such that stack[i, j, :] contains the values of the neighbors of (i, j) in the
        # original array.
        stack = np.stack(
            [
                padded_accumulator[i : i + h, j : j + w]
                for i in range(2 * pad_offset + 1)
                for j in range(2 * pad_angle + 1)
            ],
            axis=0,
        )

        # Find the max value along the last axis of the stack, and compare it to the original
        return np.where(self.accumulator == stack.max(axis=0), self.accumulator, 0)

    def get_lines(self, threshold: float, nms_angle_range: float, nms_offset_range: float):
        """Return a list of lines (rho, theta) which have a vote count above the threshold and
        are a local maximum in the accumulator space.

        Steps:
        1. non-maximal suppression of the accumulator
        2. normalization so that votes are in the range [0, 1]
        3. thresholding
        4. Return all (rho, theta) lines that are above threshold and are local maxima
        """
        votes = self.non_maximal_suppression(nms_angle_range, nms_offset_range)
        # raise NotImplementedError("your code here")
        # # Normalize the votes to range [0, 1]
        # max_vote = np.max(votes)
        # if max_vote > 0:
        #     votes /= np.max(votes)
        # # Threshold the votes
        # votes[votes < threshold] = 0
        # # Get indices of lines above threshold and are local maxima
        # ii, jj = np.where(votes > 0)
        # # Return lines (rho, theta)
        # return np.stack([self.offsets[ii], self.angles[jj]], axis=-1)
        # Normalize votes to range [0, 1]
        max_vote = votes.max()
        if max_vote > 0:
            votes /= votes.max()

        # Apply threshold
        idx = np.where(votes > threshold)

        # Get rho and theta values for lines above the threshold
        rhos = self.offsets[idx[0]]
        thetas = self.angles[idx[1]]

        # Zip them together and return as a list of tuples
        lines = list(zip(rhos, thetas))
        return lines

    def line_to_p(self, offset, angle):
        """Convert a line (rho, theta) to the point (x, y) in the image coordinate system on the
        line closest to the origin.
        """
        return np.array([np.cos(angle), np.sin(angle)]) * offset + self.origin_xy

    def line_to_xy_endpoints(self, offset, angle):
        """Convert a line (offset, angle) to a pair of points (x1, y1), (x2, y2) which are the
        endpoints of the line.
        """
        length = float(self.offsets.max() - self.offsets.min())
        p = self.line_to_p(offset, angle)
        v = np.array([np.sin(angle), -np.cos(angle)])
        return p + v * length / 2, p - v * length / 2


def main(
    image: np.ndarray,
    canny_blur: int,
    canny_thresholds: tuple[float, float],
    max_edges: int,
    min_angle: float,
    max_angle: float,
    angle_spacing: float,
    offset_spacing: float,
    accumulator_threshold: float,
    nms_angle_range: float,
    nms_offset_range: float
) -> np.ndarray:
    annotated_image = image.copy()

    # Convert to grayscale.
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image before running Canny edge detection.
    image = cv.GaussianBlur(image, (canny_blur, canny_blur), 0)

    # Run Canny edge detection.
    edges = cv.Canny(image, *canny_thresholds)

    # Create a HoughLineDetector object.
    hough = HoughLineDetector(image.shape[:2], min_angle, max_angle, angle_spacing, offset_spacing)

    # Iterate over the edges and add each edge to the HoughLineDetector.
    yx = np.argwhere(edges > 0)
    if len(yx) > max_edges:
        yx = yx[np.random.choice(len(yx), max_edges, replace=False)]
    hough.add_points_xy(yx[:, ::-1])

    # Get the lines from the HoughLineDetector.
    lines = hough.get_lines(accumulator_threshold, nms_angle_range, nms_offset_range)

    # Draw the lines on the original image.
    for offset, angle in lines:
        p1, p2 = hough.line_to_xy_endpoints(offset, angle)
        cv.line(
            annotated_image,
            tuple(p1.astype(int)),
            tuple(p2.astype(int)),
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

    return annotated_image
