import numpy as np
import cv2 as cv  # noqa: F401


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with uint8 values in the range [0, 255] and
    return a copy of the image with data type float32 and values in the range [0, 1]
    """
    # raise NotImplementedError("your code here")
    # Ensure the image is in the correct data type
    image = image.astype(np.float32)

    # Scale the values to the range [0, 1]
    image /= 255.0

    return image


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with float32 values in the range [0, 1] and
    return a copy of the image with uint8 values in the range [0, 255]. Values outside the range
    should be clipped (i.e. a float of 1.1 should be converted to a uint8 of 255, and a float of
    -0.1 should be converted to a uint8 of 0).
    """
    # raise NotImplementedError("your code here")
    # Clip values to the range [0, 1]
    clipped_image = np.clip(image, 0, 1)

    # Scale values to the range [0, 255] and round to the nearest integer
    uint8_image = (clipped_image * 255 + 0.5).astype(np.uint8)

    return uint8_image


def crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image cropped to the
    given rectangle. Any part of the rectangle that falls outside the image should be considered
    black (i.e. 0 intensity in all channels).
    """
    # raise NotImplementedError("your code here")
    # Get the height and width of the original image
    original_height, original_width, _ = image.shape

    # Create a black canvas with the specified width and height
    cropped_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Calculate the region to copy from the original image
    x_start = max(-x, 0)
    y_start = max(-y, 0)
    x_end = min(original_width - x, w)
    y_end = min(original_height - y, h)

    # Calculate the region to paste into the cropped image
    copy_width = x_end - x_start
    copy_height = y_end - y_start
    paste_width = min(w, copy_width)
    paste_height = min(h, copy_height)

    # Copy the region from the original image to the cropped image
    cropped_image[y_start:y_start + paste_height, x_start:x_start + paste_width, :] = \
        image[y + y_start:y + y_end, x + x_start:x + x_end, :]

    return cropped_image


def scale_by_half_using_numpy(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image taking every
    other pixel in each row and column. For example, if the original image has shape (H, W, 3),
    the returned image should have shape (H // 2, W // 2, 3).
    """
    half_size_image = image[::2, ::2, :]

    return half_size_image


def scale_by_half_using_cv(image: np.ndarray) -> np.ndarray:
    """Using cv.resize, take an image and return a copy of the image scaled down by a factor of 2,
    mimicking the behavior of scale_by_half_using_numpy_slicing. Pay attention to the
    'interpolation' argument of cv.resize (see the OpenCV documentation for details).
    """
    # raise NotImplementedError("your code here")
    # Get the height and width of the image
    h, w = image.shape[0:2]
    
    # Resize the numpy-sliced image to match the behavior
    half_size_cv = cv.resize(image, (w // 2, h // 2), interpolation=cv.INTER_NEAREST)

    return half_size_cv


def horizontal_mirror_image(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image flipped
    horizontally (i.e. a mirror image). The behavior should match cv.flip(image, 1).
    """
    # raise NotImplementedError("your code here")
    horizontal_flipped = image[:, ::-1]
    return horizontal_flipped


def rotate_counterclockwise_90(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image rotated
    counterclockwise by 90 degrees. The behavior should match
    cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE).
    """
    # raise NotImplementedError("your code here")
    rotated_image = np.transpose(image, (1, 0, 2))[::-1, :]
    return rotated_image


def swap_b_r(image: np.ndarray) -> np.ndarray:
    """Given an OpenCV image in BGR channel format, return a copy of the image with the blue and red
    channels swapped. You may use any numpy or opencv functions you like.
    """
    # raise NotImplementedError("your code here")
    b, g, r = cv.split(image)
    swapped_image = cv.merge([r, g, b])
    return swapped_image


def blues(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the blue
    channel
    """
    # raise NotImplementedError("your code here")
    # Create a copy of the input image to avoid modifying the original
    blue_image = np.copy(image)

    # Set green and red channels to 0, leaving only the blue channel
    blue_image[:, :, 1] = 0   # Set green channel to 0
    blue_image[:, :, 2] = 0   # Set red channel to 0

    return blue_image


def greens(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the green
    channel
    """
    # raise NotImplementedError("your code here")
    # Create a copy of the input image to avoid modifying the original
    green_image = np.copy(image)

    # Set blue and red channels to 0, leaving only the green channel
    green_image[:, :, 0] = 0  # Set blue channel to 0
    green_image[:, :, 2] = 0  # Set red channel to 0

    return green_image


def reds(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the red
    channel
    """
    # raise NotImplementedError("your code here")
    # Create a copy of the input image to avoid modifying the original
    red_image = np.copy(image)

    # Set green and red channels to 0, leaving only the blue channel
    red_image[:, :, 0] = 0  # Set blue channel to 0
    red_image[:, :, 1] = 0  # Set green channel to 0

    return red_image


def scale_saturation(image: np.ndarray, scale: float) -> np.ndarray:
    """Take an OpenCV image in BGR channel format. Convert to HSV and multiply the saturation
    channel by the given scale factor, then convert back to BGR.
    """
    # raise NotImplementedError("your code here")
    # Convert BGR image to HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Scale the saturation channel within the valid range [0, 255]
    scaled_saturation = np.round(np.clip(hsv_image[:, :, 1] * scale, 0, 255)).astype(np.uint8)

    # Round the scaled values to integers
    hsv_image[:, :, 1] = np.round(scaled_saturation)

    # Convert back to BGR
    result_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    return result_image


def grayscale(image: np.ndarray) -> np.ndarray:
    """Using numpy, reproduce the OpenCV function cv.cvtColor(image, cv.COLOR_BGR2GRAY) to convert
    the given image to grayscale. The returned image should still be in BGR channel format.
    """
    # raise NotImplementedError("your code here")
    # Convert BGR image to grayscale using correct weights
    gray_channel = np.dot(image[..., :3], [0.114, 0.587, 0.299])

    # Round the values and stack the grayscale channel to get BGR format
    gray_image_bgr = np.stack([np.round(gray_channel)] * 3, axis=-1).astype(np.uint8)

    return gray_image_bgr


def tile_bgr(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a 2x2 tiled copy of the image, with the
    original image in the top-left, the blue channel in the top-right, the green channel in the
    bottom-left, and the red channel in the bottom-right. If the original image has shape (H, W, 3),
    the returned image has shape (2 * H, 2 * W, 3).
    """
    # raise NotImplementedError("your code here")
    # Create a copy of the input image to avoid modifying the original
    tiled_image = np.copy(image)

    # Split the image into blue, green, and red channels
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    # Create a black canvas with the specified width and height
    black_canvas = np.zeros_like(blue_channel)

    # Create a 2x2 grid for the tiled image
    top_left = image
    top_right = np.stack([blue_channel, black_canvas, black_canvas], axis=-1)
    bottom_left = np.stack([black_canvas, green_channel, black_canvas], axis=-1)
    bottom_right = np.stack([black_canvas, black_canvas, red_channel], axis=-1)

    # Combine the grid into the final tiled image
    tiled_image = np.vstack([np.hstack([top_left, top_right]),
                             np.hstack([bottom_left, bottom_right])])

    return tiled_image
