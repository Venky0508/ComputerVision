import cv2 as cv
import numpy as np


_DEFAULT_GAIN_BGR = (0.8, 1.0, 1.5)
_DEFAULT_ALPHA_LUT = 0.1


def adjust_gain_bgr(image: np.ndarray, scale: tuple[float, float, float]) -> np.ndarray:
    """Scale the B, G, and R channels of an image by the given scale factors. Input image shoul
    in uint8 format. Output image will also be in uint8 format. Values are rounded and clipped to
    ensure that they are in the range [0, 255].
    """
    # raise NotImplementedError("your code here")
    scaled_image = np.round(image * np.array(scale)).clip(0, 255).astype(np.uint8)
    return scaled_image


def create_lookup_table(image: np.ndarray, alpha: float) -> np.ndarray:
    """Using numpy, create a lookup table for partial histogram equalization. When alpha=0,
    the lookup table should act such that the image is unchanged. When alpha=1, the lookup table
    should act such that the image is fully equalized. Values of alpha in between should act as a
    partial equalization.
    Useful numpy functions: np.arange(), np.bincount(), np.cumsum()
    """
    # raise NotImplementedError("your code here")
    # Extract the L channel from the LAB image
    if alpha != 1:
        l_channel = image[:, :, 0]
    else:
        l_channel = image

    # Compute histogram using np.bincount
    hist = np.bincount(l_channel.flatten(), minlength=256)

    # Compute cumulative sum of histogram as float
    lut_cum_sum = np.cumsum(hist)

    # Normalize cumulative sum to range [0, 1]
    lut_cum_sum_normalized = (lut_cum_sum / lut_cum_sum.max()) * 255

    # Compute final lookup table
    if alpha < 1:
        lut = (1 - alpha) * np.arange(256) + alpha * lut_cum_sum_normalized
    else:
        lut = lut_cum_sum_normalized

    # Round and clip the values
    lut = np.round(lut).clip(0, 255)

    return lut.astype(np.uint8)


def enhance(
    input_image_path: str,
    output_image_path: str,
    gain_bgr: tuple[float, float, float],
    alpha_lut: float,
):
    """Main function for underwater image enhancement.

    Steps:
    1. apply scale factors to the B, G, and R channels of the image. chosen scale factors should
       do something like boost the amount of red in the image, and reduce the amount of blue and
       green, since water absorbs red light more than blue and green.
    2. convert the image to the LAB color space.
    3. create a lookup table that will partially equalize the L channel using the
       create_lookup_table function. Choose an alpha value that you think makes the image look good.
    4. apply the lookup table to the L channel using the cv.LUT function.
    5. convert the result back to the BGR color space.
    """
    image = cv.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not find image {input_image_path}")

    # Step 1: Apply scale factors to BGR channels
    scaled_image = np.round(image * np.array(gain_bgr)).clip(0, 255).astype(np.uint8)

    # Step 2: Convert the image to the LAB color space
    lab_image = cv.cvtColor(scaled_image, cv.COLOR_BGR2LAB)

    # Step 3: Create a lookup table for partial histogram equalization
    lut = create_lookup_table(lab_image, alpha_lut)

    # Step 4: Apply the lookup table to the L channel
    lab_image[:, :, 0] = cv.LUT(lab_image[:, :, 0], lut)

    # Ensure values are rounded and clipped in LAB space
    lab_image = np.round(lab_image).clip(0, 255).astype(np.uint8)

    # Step 5: Convert the result back to the BGR color space
    enhanced_image = cv.cvtColor(lab_image, cv.COLOR_LAB2BGR)

    # Ensure values are rounded and clipped in BGR space
    enhanced_image = np.round(enhanced_image).clip(0, 255).astype(np.uint8)

    # Save the enhanced image
    cv.imwrite(output_image_path, enhanced_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="input image")
    parser.add_argument("--output", required=True, type=str, help="output image")
    parser.add_argument(
        "--gain-b",
        required=False,
        type=float,
        default=_DEFAULT_GAIN_BGR[0],
        help="gain for blue channel",
    )
    parser.add_argument(
        "--gain-g",
        required=False,
        type=float,
        default=_DEFAULT_GAIN_BGR[1],
        help="gain for green channel",
    )
    parser.add_argument(
        "--gain-r",
        required=False,
        type=float,
        default=_DEFAULT_GAIN_BGR[2],
        help="gain for red channel",
    )
    parser.add_argument(
        "--alpha",
        required=False,
        type=float,
        default=_DEFAULT_ALPHA_LUT,
        help="alpha for partial histogram equalization",
    )
    args = parser.parse_args()

    enhance(args.input, args.output, (args.gain_b, args.gain_g, args.gain_r), args.alpha)
