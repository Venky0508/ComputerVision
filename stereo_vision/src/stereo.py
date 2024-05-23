import cv2 as cv
import numpy as np
from utils import ceil_16, fix16_to_float32, label_image
from pathlib import Path


def my_sad_disparity_map(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int,
    max_disparity: int,
) -> np.ndarray:
    """Compute a disparity value for each pixel in img1 using the SAD metric.

    :param img1: left image
    :param img2: right image
    :param window_size: size of the window used in template matching
    :param max_disparity: maximum disparity value to search for
    :return: disparity map of same width and height as img1, in float32.
    """
    img_shape = img1.shape
    img1, img2 = np.atleast_3d(img1) / 255.0, np.atleast_3d(img2) / 255.0
    assert img1.shape == img2.shape, "Images must have the same shape."

    padded_img2 = np.pad(img2, [(0, 0), (max_disparity + 1, 0), (0, 0)], mode='constant')

    # Initialize the SAD array
    sad = np.zeros((img1.shape[0], img1.shape[1], max_disparity + 1), dtype=np.float32)

    # Loop through all disparities
    for d in range(max_disparity + 1):
        # Shift img2 by disparity d
        if d == 0:
            shifted_img2 = padded_img2[:, max_disparity + 1:, :]
        else:
            shifted_img2 = padded_img2[:, (max_disparity + 1) - d:-d, :]
        # Calculate SAD
        diff = img1 - shifted_img2
        abs_val = np.abs(diff)
        windows = cv.filter2D(abs_val, -1, np.ones((window_size, window_size), np.float32))
        if len(img_shape) != 2:
            sad[:, :, d] = np.sum(windows, axis=2)
        else:
            sad[:, :, d] = windows

    # Find the disparity index with minimum SAD for each pixel
    disparity_map = np.argmin(sad, axis=-1)
    return disparity_map.astype(np.float32)


def my_leaderboard_disparity_map(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int,
    max_disparity: int,
) -> np.ndarray:
    """Compute a disparity value for each pixel in img1 using the SAD metric.

    :param img1: left image
    :param img2: right image
    :param window_size: size of the window used in template matching
    :param max_disparity: maximum disparity value to search for
    :return: disparity map of same width and height as img1, in float32.
    """
    # Default: just redirect to my_sad_disparity_map.
    # IF YOU WANT TO PARTICIPATE IN THIS LEADERBOARD CHALLENGE, IMPLEMENT THIS FUNCTION USING
    # NUMPY AND OPENCV PRIMITIVES.
    return my_sad_disparity_map(img1, img2, window_size, max_disparity)


def rms_error(
    true_disparity: np.ndarray, disparity: np.ndarray, zero_penalty: float = 1.0
) -> float:
    """Compute the root mean squared error between the true disparity and the estimated disparity.

    A disparity value of less than or equal to zero is considered a special case of "unknown"
    disparity levels. Where true_disparity is zero, values in 'disparity' are ignored. Where
    disparity is zero but true_disparity is not, the error is penalized by zero_penalty.

    :param true_disparity: true disparity map
    :param disparity: estimated disparity map
    :return: root mean squared error.
    """
    # raise NotImplementedError("Your code here (3 lines in the answer key)")
    valid_mask = true_disparity > 0
    error = np.zeros_like(true_disparity, dtype=np.float32)
    error[valid_mask] = (true_disparity[valid_mask] - disparity[valid_mask]) ** 2
    error[(disparity <= 0) & valid_mask] = zero_penalty ** 2
    rms = np.sqrt(np.mean(error[valid_mask]))
    return rms


def percent_match(
    true_disparity: np.ndarray,
    disparity: np.ndarray,
    zero_weight: float = 0.5,
    threshold: float = 1.0,
) -> float:
    """Compute the percentage of pixels where the estimated disparity is within ± threshold
    pixels of the true disparity.

    A disparity value of less than or equal to zero is considered a special case of "unknown"
    disparity levels. Where true_disparity is zero, values in 'disparity' are ignored. Where
    disparity is zero but true_disparity is not, the error is weighted by zero_weight.

    :param true_disparity: true disparity map
    :param disparity: estimated disparity map
    :return: root mean squared error.
    """
    is_error = (np.abs(true_disparity - disparity) > threshold).astype(float)
    is_error[disparity <= 0] = zero_weight
    return 1 - np.mean(is_error[true_disparity > 0])


def main(scene_folder: Path, window_size: int, scale: float):
    # Load images
    img1 = cv.imread(str(scene_folder / "view1.png"), cv.IMREAD_COLOR)
    img2 = cv.imread(str(scene_folder / "view5.png"), cv.IMREAD_COLOR)

    if (scene_folder / "disp1.png").exists():
        true_disparity = cv.imread(str(scene_folder / "disp1.png"), cv.IMREAD_GRAYSCALE)

        # Per the Middlebury stereo data docs, disparity maps are stored relative to
        # full-resolution images and need to be scaled. But, values where true_disparity is zero
        # are considered unknown and should remain at zero.
        true_disparity[true_disparity > 0] = true_disparity[true_disparity > 0] * scale
    else:
        # Unknown disparity everywhere
        true_disparity = np.zeros_like(img1[..., 0])

    # Heuristic: set max disparity to the smallest multiple of 16 that is larger than 1/8th the
    # image width. Note that StereoSGBM requires this to be a multiple of 16.
    max_disparity = ceil_16(img1.shape[1] / 8)

    # Compute disparity (custom implementation of SAD)
    my_disparity_1 = my_sad_disparity_map(
        img1, img2, max_disparity=max_disparity, window_size=window_size
    )

    # Compute disparity (my leaderboard submission)
    # my_disparity_2 = my_leaderboard_disparity_map(
    #     img1, img2, max_disparity=max_disparity, window_size=window_size
    # )
    my_disparity_2 = my_disparity_1

    # Compute disparity (OpenCV).
    cv_stereo_matcher = cv.StereoSGBM.create(
        minDisparity=0,
        numDisparities=max_disparity,
        mode=cv.STEREO_SGBM_MODE_HH,
        blockSize=window_size,
    )

    # Per the docs, the disparity map returned by a StereoSGBM object is in 16-bit fixed-point
    # decimal format with 4 fractional bits. This is not natively handled by numpy, so it ends up
    # looking like an array with dtype np.int16.
    cv_disparity = fix16_to_float32(cv_stereo_matcher.compute(img1, img2), fractional=4)

    # Display
    disp_my_disparity_1 = np.tile((my_disparity_1[:, :, None]), (1, 1, 3)) / scale / 255
    disp_my_disparity_2 = np.tile((my_disparity_2[:, :, None]), (1, 1, 3)) / scale / 255
    disp_cv_disparity = np.tile((cv_disparity[:, :, None]), (1, 1, 3)) / scale / 255
    disp_true_disparity = np.tile((true_disparity[:, :, None]), (1, 1, 3)) / scale / 255
    blocks = np.vstack(
        [
            np.hstack(
                [
                    label_image(img1 / 255, "Left Image"),
                    label_image(img2 / 255, "Right Image"),
                    label_image(disp_true_disparity, "True Disparity Map"),
                ]
            ),
            np.hstack(
                [
                    label_image(disp_my_disparity_1, "My Disparity Map (SAD)"),
                    # label_image(img1 / 255, "Left Image"),
                    label_image(disp_my_disparity_2, "My Disparity Map (Leaderboard)"),
                    label_image(disp_cv_disparity, "OpenCV Disparity Map"),
                ]
            ),
        ]
    )

    cv.imshow("Comparison of Stereo Methods", blocks)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scene_folder",
        type=Path,
        help="Path to the scene folder which must at least contain view1.png, view5.png, and "
        "disp1.png.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=7,
        help="Size of the window used for computing the disparity map.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1 / 3,
        help="Scale factor for GT disparity. Defaults to 1/3 because this is the scale of the "
        "images provided with the assignment.",
    )
    args = parser.parse_args()
    main(args.scene_folder, args.window_size, args.scale)
