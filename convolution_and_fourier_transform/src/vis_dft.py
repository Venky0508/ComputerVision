import cv2 as cv
import numpy as np
from pathlib import Path
import argparse


def fourier_magnitude_as_image(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Displays the log magnitude of the fourier transform of the image with the DC component in
    the center. Output is scaled to 0-255.

    magnitude is calculated as log(eps + |F(u, v)|) where F(u, v) is the fourier transform of the
    image. eps is a small value to avoid division by zero. The output will always be normalized to
    use the full range of 0-255.

    :param image: a single-channel image as a (h, w) array
    :param eps: small value to avoid division by zero
    :return: a grayscale image representing the (log) magnitude of the fourier transform.
    """
    # raise NotImplementedError("your code here")
    # Compute the 2D Fourier Transform of the image
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    # Compute the magnitude, apply log scaling
    mag = np.log(eps + np.abs(f_shift))

    # Normalize magnitude to the range [0, 255]
    magnitude_normalized = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    return magnitude_normalized


def fourier_phase_as_image(image: np.ndarray) -> np.ndarray:
    """Displays the phase of the fourier transform of the image with the DC component in the
    center, using HSV color space (saturation and value both set to max).

    :param image: a single-channel image as a (h, w) array
    :return: a BGR image representing the phase of the fourier transform with hue.
    """
    # Compute the Fourier transform of the image
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    # Compute the phase
    phase_spectrum = np.angle(f_shift)

    # Normalize phase to [0, 1] and then scale to [0, 180] for hue representation
    # 0 and pi radians map to 0 and 90 degrees, respectively
    # -pi radians also maps to 90 degrees
    # Linearly scale phase values: [0, pi] -> [0, 90] and [-pi, 0] -> [90, 180]
    # normalized_phase = (phase_spectrum + np.pi) / np.pi  # Normalize to [0, 1]
    # hue_array = normalized_phase * 90  # Scale to [0, 180] for hue
    hue_array = np.zeros_like(phase_spectrum)
    for row in range(len(phase_spectrum)):
        for col in range(len(phase_spectrum[row])):
            val = phase_spectrum[row][col]
            if 0 <= val <= np.pi:
                new_val = val / np.pi
                new_val = round(new_val * 90)
                hue_array[row][col] = new_val
            elif -np.pi <= val < 0:
                new_val = val / np.pi
                new_val = round((new_val + 2) * 90)
                hue_array[row][col] = new_val

    # Create HSV image, set Saturation and Value to max
    hsv_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    hsv_img[..., 0] = hue_array.astype(np.uint8)  # Hue
    hsv_img[..., 1] = 255  # Saturation
    hsv_img[..., 2] = 255  # Value

    # Convert HSV to BGR for display
    bgr_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)

    return bgr_img


def swap_magnitude_phase(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Swap the magnitude and phase of two images in the fourier domain. The magnitude is taken
    from image1 and the phase is taken from image2.

    :param image1: an image that will provide the magnitude part in the frequency domain
    :param image2: an image that will provide the phase part in the frequency domain
    :return: a new image with the magnitude of image1 and the phase of image2
    """
    f1 = np.fft.fft2(image1)
    f2 = np.fft.fft2(image2)
    mag = np.abs(f1)
    phase = np.angle(f2)
    combined = np.fft.ifft2(mag * np.exp(1j * phase)).real
    return np.clip(combined, 0, 255).astype(np.uint8)


def display(image: np.ndarray):
    """Main display function for a single image. Shows magnitude and phase.

    Caller must destroy windows.
    """
    cv.imshow("Original", image)
    cv.imshow("Magnitude", fourier_magnitude_as_image(image))
    cv.imshow("Phase", fourier_phase_as_image(image))
    cv.waitKey(0)


def display_swap(image1: np.ndarray, image2: np.ndarray):
    """Main display function for a pair of images. Shows swapped magnitude + phase.

    Caller must destroy windows.
    """
    combined = swap_magnitude_phase(image1, image2)
    cv.imshow("Image1", image1)
    cv.imshow("Image2", image2)
    cv.imshow("Combined (magnitude1, phase2)", combined)
    cv.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "images",
        help="One or more images (space-separated if more than one) to process",
        type=Path,
        nargs="+",
    )
    parser.add_argument("-o", "--output-dir", help="Output directory for images", type=Path)
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True)

    images = [cv.imread(str(im), cv.IMREAD_GRAYSCALE) for im in args.images]
    images = [im for im in images if im is not None]
    for im in images:
        display(im)
    cv.destroyAllWindows()

    for i, im1 in enumerate(images):
        for j, im2 in enumerate(images):
            if i == j:
                continue
            display_swap(im1, im2)
    cv.destroyAllWindows()
