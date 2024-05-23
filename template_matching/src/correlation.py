import numpy as np


def my_correlation(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Perform correlation of the given image with the given filter or kernel. This function must
    use numpy - you may not make any calls to OpenCV or scipy. The output should exactly mimic the
    behavior or cv.filter2D with borderType=cv.BORDER_REPLICATE. That is, the output should be the
    same size as the input image, and the output should be computed "as if" the image were padded
    with a border of pixels that replicate the edge values of the input image.
    """
    # By default, use float32 for the output, to avoid overflow when summing the products. Will
    # convert back to the input image's dtype at the end.
    out = np.zeros_like(image, dtype=np.float32)
    h, w = image.shape[:2]
    kernel_h, kernel_w = kernel.shape[:2]

    for y in range(h):
        for x in range(w):
            res = 0.0
            for kernel_y in range(kernel_h):
                for kernel_x in range(kernel_w):
                    img_y = min(max(y + kernel_y - kernel_h // 2, 0), h - 1)
                    img_x = min(max(x + kernel_x - kernel_w // 2, 0), w - 1)
                    res += image[img_y, img_x] * kernel[kernel_y, kernel_x]
            out[y, x] = res

    # Convert back to the input image's dtype, rounding to the nearest integer and clipping the
    # range if the input image's dtype is some kind of integer type
    if np.issubdtype(image.dtype, np.integer):
        out = np.clip(np.round(out), np.iinfo(image.dtype).min, np.iinfo(image.dtype).max)
    return out.astype(image.dtype)
