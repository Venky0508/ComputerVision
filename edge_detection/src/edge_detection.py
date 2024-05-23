import cv2 as cv
import numpy as np


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


def separable_conv2D(image: np.ndarray, u: np.ndarray, v: np.ndarray, **kwargs) -> np.ndarray:
    """Apply a *separable convolution* operation. This is a special case of convolution where the
    kernel is separable, i.e. it can be expressed as the outer product of two vectors. This allows
    us to perform two 1D convolutions instead of one 2D convolution, which is faster.

    :param image: The input image
    :param u: The first 1D kernel
    :param v: The second 1D kernel
    :param kwargs: Additional arguments to cv.filter2D
    :return: The result of convolving the image with the separable kernel
    """
    # Ensure u is a column vector and v is a row vector
    u, v = u.reshape(-1, 1), v.reshape(1, -1)
    return conv2D(conv2D(image, u, **kwargs), v, **kwargs)


def filter_filter(filter1: np.ndarray, filter2: np.ndarray) -> np.ndarray:
    """Using the associative property of convolution, we can combine two filters into one. In other
    words, if filter3 = filter_filter(filter1, filter2), then conv2D(image, filter3) is equivalent
    to conv2D(conv2D(image, filter1), filter2).
    """
    assert (
        filter1.ndim == 2 and filter2.ndim == 2
    ), "Filters must have ndim=2 so that it's clear which axis is rows and which is columns"
    assert (s % 2 == 1 for s in filter1.shape + filter2.shape), "Only supports odd-sized filters"
    h2, w2 = filter2.shape[:2]
    # Combined filter will be slightly bigger than each of the first two. New size will be (h1 +
    # h2 - 1, w1 + w2 - 1) using zero-padding. (Replicating np.convolve(..., mode="full")
    # behavior but with OpenCV).
    pad_w, pad_h = (w2 - 1) // 2, (h2 - 1) // 2
    padded_filter1 = np.pad(filter1, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    return conv2D(padded_filter1, filter2, borderType=cv.BORDER_CONSTANT)


def my_edge_detect_1(
    image: np.ndarray, blur_size: int, low_threshold: float, high_threshold: float
):
    """
    Canny Edge Detection using OpenCV. Steps are (1) blur the image, (2) run Canny Edge Detection.

    :param image: The input image
    :param blur_size: width and height of the Gaussian blur kernel
    :param low_threshold: Low threshold for Canny Edge Detection
    :param high_threshold: High threshold for Canny Edge Detection
    :return: The edge detected image
    """
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(image, (blur_size, blur_size), 0)

    return cv.Canny(
        image=blurred, threshold1=low_threshold, threshold2=high_threshold, L2gradient=True
    )


def get_separated_deriv_of_gaussian_kernel(blur_size: int):
    """Construct the DoG filter by constructing its two separate 'u' and 'v' components. The result
    is a pair of 1D kernels, one for the u-component and one for the v-component. Both the Gaussian
    kernel and the Sobel kernel are separable, so the DoG filter is also separable. The DoG filter
    is constructed such that conv2D(conv2D(image, gauss), sobel) is equivalent to conv2D(image, dog)

    :param blur_size: width and height of the Gaussian blur kernel
    :return: A pair of 1D kernels, one for the u-component and one for the v-component. Each
        component is a column vector. The u-component is the derivative part and the v-component is
        the tent part.
    """
    # # raise NotImplementedError("your code here")
    # Generate 1D Gaussian kernel using OpenCV
    gaussian_kernel_1d = cv.getGaussianKernel(blur_size, -1)

    # Flatten the Gaussian kernel to get u and v components
    gaussian_kernel_v = gaussian_kernel_1d
    gaussian_kernel_u = gaussian_kernel_1d.T

    # Generate Sobel kernels using OpenCV
    sobel_kernel_x = cv.getDerivKernels(1, 0, 3)

    # Transpose the Sobel kernels to obtain correct convolution result
    sobel_kernel_u = sobel_kernel_x[0].T
    sobel_kernel_v = sobel_kernel_x[1]
    # Convolve Gaussian kernel with Sobel kernels to get the derivative of Gaussian filter
    dog_u = filter_filter(gaussian_kernel_u, sobel_kernel_u)
    dog_v = filter_filter(gaussian_kernel_v, sobel_kernel_v)

    return dog_u.T, dog_v


def my_edge_detect_2(
    image: np.ndarray, blur_size: int, low_threshold: float, high_threshold: float
):
    """
    Canny Edge Detection using OpenCV. Behavior is identical to blur_canny, but the approach is
    different. Here, we precompute dx and dy image gradients using a DoG filter.

    :param image: The input image
    :param blur_size: width and height of the Gaussian blur kernel
    :param low_threshold: Low threshold for Canny Edge Detection
    :param high_threshold: High threshold for Canny Edge Detection
    :return: The edge detected image
    """
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Get derivative of Gaussian kernels
    u, v = get_separated_deriv_of_gaussian_kernel(blur_size)

    if image.dtype != np.int16:
        image = image.astype(np.int16)
    # Compute gradients using separable convolution
    dx = separable_conv2D(image, v, u.T)
    dy = separable_conv2D(image, u, v.T)

    if dx.dtype != np.int16:
        dx = dx.astype(np.int16)
    if dy.dtype != np.int16:
        dy = dy.astype(np.int16)
    # Pass the image gradients to Canny Edge Detection
    res = cv.Canny(
        dx=dx, dy=dy, threshold1=low_threshold, threshold2=high_threshold, L2gradient=True
    )
    return res


def run_interactive(image: np.ndarray, blur: int, low_threshold: float, high_threshold: float):
    windows = ["Original", "Canny 1 (Gaussian Blur)", "Canny 2 (Derivative of Gaussians)"]
    trackbars = ["Blur//2", "Low Threshold", "High Threshold"]
    cv.namedWindow(windows[0], cv.WINDOW_NORMAL)
    cv.namedWindow(windows[1], cv.WINDOW_NORMAL)
    cv.namedWindow(windows[2], cv.WINDOW_NORMAL)

    def redraw():
        edges1 = my_edge_detect_1(
            image, blur_size=blur, low_threshold=low_threshold, high_threshold=high_threshold
        )

        edges2 = my_edge_detect_2(
            image, blur_size=blur, low_threshold=low_threshold, high_threshold=high_threshold
        )
        cv.imshow(windows[0], image)
        cv.imshow(windows[1], cv.normalize(edges1, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
        cv.imshow(windows[2], cv.normalize(edges2, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))

    def on_blur_set(value):
        nonlocal blur
        blur = 2 * value + 1
        redraw()

    def on_low_threshold_set(value):
        nonlocal low_threshold, high_threshold
        low_threshold = value
        if value >= high_threshold:
            cv.setTrackbarPos(trackbars[2], windows[0], value + 1)
        redraw()

    def on_high_threshold_set(value):
        nonlocal low_threshold, high_threshold
        high_threshold = value
        if value <= low_threshold:
            cv.setTrackbarPos(trackbars[1], windows[0], value - 1)
        redraw()

    cv.createTrackbar(trackbars[0], windows[0], blur // 2, 30, on_blur_set)
    cv.createTrackbar(trackbars[1], windows[0], int(low_threshold), 255, on_low_threshold_set)
    cv.createTrackbar(trackbars[2], windows[0], int(high_threshold), 255, on_high_threshold_set)

    while cv.waitKey(1) != ord("q"):
        ...

    cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Canny Edge Detection")
    parser.add_argument("image", help="Path to the image")
    parser.add_argument("-b", "--blur", type=int, default=5, help="Width of blur kernel")
    parser.add_argument(
        "-l",
        "--low-threshold",
        type=float,
        default=50.0,
        help="Low threshold for Canny Edge Detection",
    )
    parser.add_argument(
        "-u",
        "--upper-threshold",
        type=float,
        default=150.0,
        help="High threshold for Canny Edge Detection",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    else:
        image = cv.imread(args.image)
        if image is None:
            raise ValueError(f"Invalid image: {image}")

    if args.low_threshold >= args.upper_threshold:
        raise ValueError("Low threshold must be less than high threshold")

    if args.low_threshold < 0 or args.upper_threshold < 0:
        raise ValueError("Thresholds must be non-negative")

    if args.blur < 1:
        raise ValueError("Blur size must be at least 1")

    if args.blur % 2 == 0:
        raise ValueError("Blur size must be odd")

    run_interactive(image, args.blur, args.low_threshold, args.upper_threshold)
