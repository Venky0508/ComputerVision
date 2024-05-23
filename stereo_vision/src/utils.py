import numpy as np
import cv2 as cv


def ceil_16(x):
    """Return the smallest multiple of 16 that is greater than or equal to x."""
    return int(np.ceil(x / 16) * 16)


def ceil_odd(x):
    """Return the smallest odd number that is greater than or equal to x."""
    return int(np.ceil(x / 2) * 2 + 1)


def fix16_to_float32(x, fractional=4):
    """Convert 16-bit fixed-point decimal representation with 'fractional' fractional bits to 32-bit
    floating point. Fixed-point is not natively represented in numpy, so it's expected that the
    input should be int16 and interpreted as value*2**fractional."""
    return (x / 2**fractional).astype(np.float32)


def label_image(
    img: np.ndarray,
    label: str,
    color: tuple[int, int, int] = (0, 0, 255),
    location_xy: tuple[int, int] = (10, 30),
) -> np.ndarray:
    return cv.putText(
        img.copy(), label, location_xy, cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, lineType=cv.LINE_AA
    )
