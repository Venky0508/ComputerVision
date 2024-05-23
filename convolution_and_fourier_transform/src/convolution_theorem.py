import numpy as np
import cv2 as cv
from pathlib import Path
import argparse


def conv2D(image: np.ndarray, h: np.ndarray, **kwargs) -> np.ndarray:
    """Apply a *convolution* operation rather than a *correlation* operation. Using the fact that
    convolution is equivalent to correlation with a flipped kernel, and cv.filter2D implements
    correlation.

    :param image: The input image
    :param h: The kernel
    :param kwargs: Additional arguments to cv.filter2D
    :return: The result of convolving the image with the kernel
    """
    return cv.filter2D(image, -1, cv.flip(h, -1), **kwargs)


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """Convert an image from uint8 to float32, scaling the values to the range [0, 1].

    :param image: the uint8 image to be converted.
    """
    if np.issubdtype(image.dtype, np.floating):
        return image
    return image.astype("float32") / 255.0


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image from float32 to uint8, scaling the values to the range [0, 255].

    :param image: the float image to be converted.
    """
    if np.issubdtype(image.dtype, np.unsignedinteger):
        return image
    return np.clip(np.round(image * 255), 0, 255).astype("uint8")


def text_to_array(file: Path) -> np.ndarray:
    """Read a text file containing a matrix of numbers and return it as a numpy array. The file
    should contain one row per line, with the numbers separated by whitespace. Returned array will
    always have ndim=2 even if the file contains a row or column vector.

    :param file: The file to read
    """
    with open(file) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return np.array([[float(x) for x in line.split()] for line in lines])


def convolution_theorem(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Replicate the behavior of conv2D with borderType=cv.BORDER_CONSTANT, but do it in the
    frequency domain using the convolution theorem.
    """
    # raise NotImplementedError("your code here")
    # Find the size for padding, which is the original size plus filter size minus 1
    padded_shape = (image.shape[0] + filter.shape[0] - 1,
                    image.shape[1] + filter.shape[1] - 1)

    # Compute the Fourier Transform of the image and filter
    # Specifying s=padded_shape pads the FFT with zeros
    img_f = np.fft.fft2(image, s=padded_shape, axes=(0, 1))
    filter_f = np.fft.fft2(filter, s=padded_shape, axes=(0, 1))

    # Ensure the filter has the same number of channels as the image
    if image.ndim == 3:
        filter_f = filter_f[:, :, np.newaxis]

    # Multiply the Fourier Transforms
    img_filtered_f = img_f * filter_f

    # Inverse Fourier Transform to get back to spatial domain
    result = np.fft.ifft2(img_filtered_f, axes=(0, 1))
    result = np.real(result)  # Discard the imaginary part

    # Crop and center the result back to the original image size
    h_shift = filter.shape[0] // 2
    w_shift = filter.shape[1] // 2
    result = np.roll(result, -h_shift, axis=0)
    result = np.roll(result, -w_shift, axis=1)
    result = result[:image.shape[0], :image.shape[1]]

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("filter", type=Path)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image {args.image} does not exist")
    if not args.filter.exists():
        raise FileNotFoundError(f"Filter {args.filter} does not exist")

    image = uint8_to_float(cv.imread(str(args.image)))
    filter = uint8_to_float(text_to_array(args.filter))

    out1 = conv2D(image, filter, borderType=cv.BORDER_CONSTANT)
    out2 = convolution_theorem(image, filter)

    assert np.allclose(out1, out2, atol=1e-6), "Results do not match"
