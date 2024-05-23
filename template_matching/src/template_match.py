import cv2 as cv
import numpy as np
import os


# NOTE: the values you set here are the values we will import and use when testing your code
_DEFAULT_THRESHOLD = 0.74
_DEFAULT_NMS_WINDOW = 7


def non_maximal_suppression_2d(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Performs non-maximal suppression on the given values matrix. That is, anywhere that
    values[i,j] is a *local maximum* within a (window x window) box, it is kept, and anywhere
    that it is not a local maximum it is 'suppressed' (set to zero).

    The original values matrix is not modified, but a new matrix is returned.
    """
    # raise NotImplementedError("Your code here.")
    out = np.zeros_like(values)
    height, width = values.shape

    # Iterate through the input values matrix
    for y in range(height):
        for x in range(width):
            value = values[y, x]

            # Define the neighborhood bounds
            y_start = max(0, y - window // 2)
            y_end = min(height, y + window // 2 + 1)
            x_start = max(0, x - window // 2)
            x_end = min(width, x + window // 2 + 1)

            # Check if the current value is the maximum in its neighborhood
            if value == np.max(values[y_start:y_end, x_start:x_end]):
                out[y, x] = value

    return out


def apply_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Applies a threshold to the given values matrix. That is, anywhere that values[i,j] is greater
    than the threshold, it is kept with the same value, and anywhere that it is below the threshold
    it is set to zero.

    The original values matrix is not modified, but a new matrix is returned.
    """
    # raise NotImplementedError("Your code here.")
    # Create a copy of the input values matrix
    threshold_values = values.copy()
    # Apply thresholding
    if threshold:
        threshold_values[threshold_values < threshold] = 0
    return threshold_values


def find_objects_by_template_matching(
    image: np.ndarray, template: np.ndarray, threshold: float, nms_window: int
) -> list[tuple[int, int]]:
    """Finds copies of the given template in the given image by template-matching. Returns a list of
    (x, y) coordinates of the top-left corner of each match. The main steps of this function are:

    1. Use cv.matchTemplate to get a score map. This map is a 2D array where score[i,j] gives a
       measure of how well the template matches the image at position (i,j). Depending on the choice
       of 'method' in cv.matchTemplate, the score can be positive or negative, and the best match
       can be either the maximum or minimum value in the score map.
    2. Normalize the score map so that the best match is 1 and the worst match is 0
    3. Apply the threshold to the score map to throw away any values below the threshold (i.e. set
       pixels to zero if their score is below the threshold). Use a call to apply_threshold for
       this.
    3. Use non-maximal suppression to keep only local maxima in a (nms_window x nms_window) window
       (i.e. set pixels to zero if they are not the maximum value among their neighbors). Use a call
       to non_maximal_suppression_2d for this.
    4. Use np.where() to find all remaining nonzero pixels --> these are our matches.
    """
    # raise NotImplementedError("Your code here.")
    # Step 1: Use cv.matchTemplate to get a score map
    score_map = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)

    # Step 2: Normalize the score map
    score_map_normalized = (score_map - score_map.min()) / (score_map.max() - score_map.min())

    # Step 3: Apply threshold to the score map
    threshold_map = apply_threshold(score_map_normalized, threshold)

    # Step 4: Perform non-maximal suppression
    suppressed_map = non_maximal_suppression_2d(threshold_map, window=nms_window)

    # Step 5: Find coordinates of remaining non-zero pixels
    y, x = np.where(suppressed_map)

    # Convert coordinates to (x, y) tuples
    matches = []
    for i in range(len(x)):
        matches.append((x[i], y[i]))

    return matches


def visualize_matches(scene: np.ndarray, obj_hw: tuple[int, int], xy: list[tuple[int, int]]):
    """Visualizes the matches found by find_objects_by_template_matching."""
    count = len(xy)
    h, w = obj_hw
    for x, y in xy:
        cv.rectangle(scene, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # Add text in the bottom left corner by using x=10 and y=the height of the scene - 20 pixels
    cv.putText(
        scene,
        f"Found {count} matches",
        (10, scene.shape[0] - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv.imshow("Matches", scene)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to the scene", required=True)
    parser.add_argument("--template", help="Path to the template", required=True)
    parser.add_argument(
        "--threshold", help="Threshold for matches", type=float, default=_DEFAULT_THRESHOLD
    )
    parser.add_argument(
        "--nms-window",
        help="Window size for non-maximal suppression",
        type=int,
        default=_DEFAULT_NMS_WINDOW,
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not os.path.exists(args.template):
        raise FileNotFoundError(f"Image not found: {args.template}")

    if args.nms_window % 2 == 0:
        raise ValueError("The window size must be odd.")

    if args.nms_window < 1:
        raise ValueError("The window size must be greater than or equal to 1.")

    scene = cv.imread(args.image)
    object1 = cv.imread(args.template)
    xy = find_objects_by_template_matching(scene, object1, args.threshold, args.nms_window)

    visualize_matches(scene, object1.shape[:2], xy)