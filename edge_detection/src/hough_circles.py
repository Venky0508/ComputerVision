import cv2 as cv
import numpy as np
# import os
from typing import Optional


class HoughCircleDetector:
    def __init__(
        self,
        image_shape: tuple[int, int],
        radius: float,
        spacing: float = 5.0,
        soft_vote_sigma: Optional[float] = None,
    ):
        h, w = image_shape

        # Default: circle spacing is once every 5 pixels
        resolution_x = int(np.ceil(w / spacing))
        resolution_y = int(np.ceil(h / spacing))

        # Create a grid of parameters (x and y centers of circles)
        self.center_x = np.linspace(0, w, resolution_x + 1)
        self.center_x = (self.center_x[:-1] + self.center_x[1:]) / 2
        self.center_y = np.linspace(0, h, resolution_y + 1)
        self.center_y = (self.center_y[:-1] + self.center_y[1:]) / 2
        self.radius = radius

        # Precompute all pairs of xy as a (2, |y|, |x|) array
        self._center_xy = np.array(np.meshgrid(self.center_x, self.center_y))

        if soft_vote_sigma is None:
            # Sensible default for 'sigma' is the smallest spacing between circles
            self.sigma = (
                min(self.center_x[1] - self.center_x[0], self.center_y[1] - self.center_y[0]) / 2
            )
        else:
            self.sigma = soft_vote_sigma

        # Initialize self.accumulator to be a 2D array of zeros with the same shape as the
        # parameter space.
        self.accumulator = np.zeros(shape=(len(self.center_y), len(self.center_x)), dtype=float)

    def clear(self):
        self.accumulator = np.zeros(shape=(len(self.center_y), len(self.center_x)), dtype=float)

    def add_points_xy(self, xy: np.ndarray):
        """Given n points in the image's coordinate system as an array of shape (n, 2), increment
        the accumulator for all circles that pass through each point.
        """
        # First, calculate the distance from each point to each circle's center. Each of dx and dy
        # has shape (n, |center_y|, |center_x|), such that dx[i,j,k] is the difference between the
        # x-coordinate of point i and the x-coordinate of the circle in the [j,k] position of the
        # accumulator.
        dx = xy[:, 0][:, np.newaxis, np.newaxis] - self._center_xy[0][np.newaxis, :, :]
        dy = xy[:, 1][:, np.newaxis, np.newaxis] - self._center_xy[1][np.newaxis, :, :]
        r = np.sqrt(dy**2 + dx**2)
        dr = np.abs(r - self.radius)
        self.accumulator = self.accumulator + np.sum(np.exp(-(dr**2) / self.sigma**2 / 2), axis=0)

    def non_maximal_suppression(self, overlap: float):
        """Do non-maximal suppression on the accumulator array. This means that for each cell in
        the accumulator which represents some circle, compare it to other 'neighboring' circles. Two
        circles are considered 'neighbors' if their difference in centers is less than 'overlap'
        times the radius. If a cell is not the maximum in its neighborhood, set it to zero.
        """
        h, w = self.accumulator.shape
        x_spacing = self.center_x[1] - self.center_x[0]
        y_spacing = self.center_y[1] - self.center_y[0]
        pad_x = int(np.ceil(overlap * self.radius / x_spacing))
        pad_y = int(np.ceil(overlap * self.radius / y_spacing))

        padded_accumulator = np.pad(
            self.accumulator, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant"
        )

        # Create a stack of accumulators of size (|center_y|, |center_x|, w_x*w_y),
        # where w_x=2*pad_x+1 and w_y=2*pad_y+1, such that stack[i, j, :] contains the values of
        # the neighbors of (i, j) in the original array.
        stack = np.stack(
            [
                padded_accumulator[i : i + h, j : j + w]
                for i in range(2 * pad_y + 1)
                for j in range(2 * pad_x + 1)
            ],
            axis=0,
        )

        # Find the max value along the last axis of the stack, and compare it to the original
        return np.where(self.accumulator == stack.max(axis=0), self.accumulator, 0)

    def get_circles(self, threshold: float, nms_overlap: float) -> np.ndarray:
        """Return a list of circles (cx, cy) which have a vote count above the threshold and are
        a local maximum in the accumulator space.

        Steps:
        1. non-maximal suppression of the accumulator
        2. normalization so that votes are in the range [0, 1]
        3. thresholding
        4. Return all (cx, cy) circle centers that are above threshold and are local maxima
        """
        votes = self.non_maximal_suppression(nms_overlap)
        # raise NotImplementedError("your code here")
        # Normalize the votes within the accumulator
        normalized_votes = votes / np.max(votes)

        # Apply a threshold to the normalized votes
        indices_above_threshold = np.where(normalized_votes >= threshold)

        # Extract the circle centers based on the indices
        circle_centers = []
        for idx in zip(*indices_above_threshold):
            center_x = self.center_x[idx[1]]
            center_y = self.center_y[idx[0]]
            circle_centers.append((center_x, center_y))

        return np.array(circle_centers)


def main(
    image: np.ndarray,
    canny_blur: int,
    canny_thresholds: tuple[float, float],
    accumulator_threshold: float,
    nms_overlap: float,
    radius: float,
    spacing: float,
    max_edges: int,
) -> np.ndarray:
    annotated_image = image.copy()

    # Convert to grayscale.
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image before running Canny edge detection.
    image = cv.GaussianBlur(image, (canny_blur, canny_blur), 0)

    # Run Canny edge detection.
    edges = cv.Canny(image, *canny_thresholds)

    # Create a HoughCircleDetector object.
    hough = HoughCircleDetector(image.shape[:2], radius, spacing)

    # Add the edge points to the HoughCircleDetector.
    yx = np.argwhere(edges > 0)
    if len(yx) > max_edges:
        yx = yx[np.random.choice(len(yx), max_edges, replace=False)]
    hough.add_points_xy(yx[:, ::-1])

    # Get the circles from the HoughCircleDetector.
    circles = hough.get_circles(accumulator_threshold, nms_overlap)

    # Draw the circles on the original image.
    for cx, cy in circles:
        cv.circle(
            annotated_image, (int(cx), int(cy)), int(radius + 0.5), (0, 0, 255), 2, cv.LINE_AA
        )

    return annotated_image
