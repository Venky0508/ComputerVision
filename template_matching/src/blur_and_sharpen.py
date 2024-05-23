import numpy as np
import cv2 as cv
import os


def create_separated_blur_or_sharpen_kernel(m: int, a: float) -> np.ndarray:
    """Creates a Mx1 kernel 'u' such that the outer-product of 'u' with itself is a blur or
    sharpen kernel 'h'. When a is +1, 'h' is a m-by-m Gaussian blur kernel. When a is 0,
    'h' is a m-by-m kernel such that convolving the image with it will have no effect. When a is
    -1, 'h' is a m-by-m sharpening kernel. The parameter 'a' thus ranges from -1 to +1 and
    controls the amount of sharpening (if negative) or amount of blurring (if positive).

    Useful: use cv.getGaussianKernel to create the Gaussian kernel.
    """
    if a == 0.0:
        delta = np.zeros((m, 1), dtype=np.float32)
        delta[m // 2, 0] = 1.0
        return delta

    elif a == +1:
        return cv.getGaussianKernel(m, 0)

    elif a == -1:
        blur_kernel = cv.getGaussianKernel(m, 0)
        delta = np.zeros((m, 1), dtype=np.float32)
        delta[m // 2, 0] = 1.0
        v = 2*delta - blur_kernel
        return v

    elif a < 0.0:
        blur_kernel = cv.getGaussianKernel(m, 0)
        delta = np.zeros((m, 1), dtype=np.float32)
        delta[m // 2, 0] = 1.0
        v = delta + a*(delta - blur_kernel)
        return v

    elif a > 0.0:
        blur_kernel = cv.getGaussianKernel(m, 0)
        delta = np.zeros((m, 1), dtype=np.float32)
        delta[m // 2, 0] = 1.0
        v = delta + a * (blur_kernel - delta)
        return v


def blur_or_sharpen(image: np.ndarray, m: int = 7, a: float = 0.0) -> np.ndarray:
    """Blurs or sharpens an image using a Gaussian kernel of size MxM. The parameter 'a' ranges
    from -1 to +1 and controls the amount of sharpening (if negative) or amount of blurring (if
    positive).
    """
    u = create_separated_blur_or_sharpen_kernel(m, a)
    return cv.sepFilter2D(image, -1, u, u, borderType=cv.BORDER_REPLICATE)


def run_live(m: int):
    """Runs a live demo of the blur_or_sharpen function."""
    camera = cv.VideoCapture(0)
    window_name = "Press Q to quit."
    cv.namedWindow(window_name)
    a = 0.0

    def on_trackbar(x):
        nonlocal a
        a = x / 50 - 1

    cv.createTrackbar("value of a [-1, +1]", window_name, 50, 100, on_trackbar)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Output will be hard to see if the image is high resolution. Scale it down if so.
        if frame.shape[0] > 800:
            frame = cv.resize(frame, (0, 0), fx=800/frame.shape[0], fy=800/frame.shape[0])

        result = blur_or_sharpen(frame, m, a)
        cv.imshow(window_name, result)
        if cv.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="The image to blur or sharpen.", default=None)
    parser.add_argument("--m", help="The size of the Gaussian kernel.", type=int, default=15)
    parser.add_argument("--a", help="The sharpening factor.", type=float, default=0.0)
    args = parser.parse_args()

    if args.m < 3:
        raise ValueError("The kernel size must be at least 3.")

    if args.m % 2 == 0:
        raise ValueError("The kernel size must be odd.")

    if args.image is None:
        run_live(args.m)
    else:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"The file '{args.image}' does not exist.")
        image = cv.imread(args.image)
        result = blur_or_sharpen(image, args.m, args.a)
        cv.imshow(f"Result for a={args.a}", result)
        cv.waitKey(0)
        cv.destroyAllWindows()
