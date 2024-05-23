import cv2 as cv
import numpy as np
from utils import uint8_to_float32, float32_to_uint8


def pyramid_pad(image: np.ndarray, k: int) -> np.ndarray:
    """Pad an image with zeros such that height and width are divisible by 2^(k-1)."""
    divisor = 2 ** (k - 1)
    h, w = image.shape[:2]
    new_h = np.ceil(h / divisor) * divisor
    new_w = np.ceil(w / divisor) * divisor
    pad_h = int(new_h - h)
    pad_w = int(new_w - w)
    return cv.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv.BORDER_CONSTANT, value=0)


def create_gaussian_pyramid(image: np.ndarray, k: int) -> list[np.ndarray]:
    """Create a k-level Gaussian pyramid with the full-resolution image in pyramid[0] and each
    pyramid[i+1] having half the width and height of pyramid[i].
    """
    # raise NotImplementedError("Your code here! (4 lines in the answer key)")
    pyramid = [image]
    for _ in range(1, k):
        image = cv.pyrDown(image)
        pyramid.append(image)
    return pyramid


def create_laplacian_pyramid(image: np.ndarray, k: int) -> list[np.ndarray]:
    """Create a k-level Laplacian pyramid starting with create_gaussian_pyramid and using
    cv.pyrUp() to upsample each level.
    """
    # raise NotImplementedError("Your code here! (4 lines in the answer key)")
    gaussian_pyramid = create_gaussian_pyramid(image, k)
    laplacian_pyramid = []
    for i in range(k - 1):
        upsampled = cv.pyrUp(gaussian_pyramid[i + 1],
                             dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian_pyramid.append(gaussian_pyramid[i] - upsampled)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Add the last level of Gaussian pyramid
    return laplacian_pyramid


def invert_laplacian_pyramid(pyramid: list[np.ndarray]) -> np.ndarray:
    """Given a Laplacian pyramid, invert it to get the original image back."""
    # raise NotImplementedError("Your code here! (4 lines in the answer key)")
    image = pyramid[-1]  # Start with the top level of the Laplacian pyramid
    for level in reversed(pyramid[:-1]):  # Iterate over levels in reverse order
        expanded = cv.pyrUp(image, dstsize=(level.shape[1], level.shape[0]))  # Upsample the image
        image = level + expanded  # Add the upsampled image to the current level
    return image


def blend_images(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray, k: int):
    """Blend two images together using a Laplacian pyramid and a mask. All images and the mask
    must have the same height and width and be in float format (ranging in [0,1]). Where the mask is
    1, image1 is used, and where the mask is 0, image2 is used.

    :param image1: First input image, in float format.
    :param image2: Second input image, in float format.
    :param mask: Binary mask image, in float format.
    :param k: Number of levels in the pyramid.
    :return: The blended image, in float format.
    """
    # raise NotImplementedError("Your code here! (11 lines in the answer key)")
    assert image1.shape[:2] == image2.shape[:2] == mask.shape[:2]

    # Convert images and mask to float format
    image_1 = image1.astype(np.float32)
    image_2 = image2.astype(np.float32)
    mask_updated = mask.astype(np.float32)

    # Pad images and mask to ensure dimensions are divisible by 2^(k-1)
    image_1 = pyramid_pad(image_1, k)
    image_2 = pyramid_pad(image_2, k)
    mask_updated = pyramid_pad(mask_updated, k)

    # Create Laplacian pyramids for both images
    l_pyramid1 = create_laplacian_pyramid(image_1, k)
    l_pyramid2 = create_laplacian_pyramid(image_2, k)

    # Create Gaussian pyramid for the mask
    gaussian_mask = create_gaussian_pyramid(mask_updated, k)

    # Blend the pyramids
    blended_pyramid = []
    if k == 0:
        part_1 = (gaussian_mask[0][:, :, np.newaxis] * l_pyramid1[0])
        part_2 = ((1 - gaussian_mask[0][:, :, np.newaxis]) * l_pyramid2[0])
        blended_level = part_1 + part_2
        blended_pyramid.append(blended_level)
    else:
        for i in range(k):
            part_1 = (gaussian_mask[i][:, :, np.newaxis] * l_pyramid1[i])
            part_2 = ((1 - gaussian_mask[i][:, :, np.newaxis]) * l_pyramid2[i])
            blended_level = part_1 + part_2
            blended_pyramid.append(blended_level)

    # Reconstruct the blended image from the blended Laplacian pyramid
    blended_image = invert_laplacian_pyramid(blended_pyramid)

    # Crop the blended image to the original size
    blended_image = blended_image[:image1.shape[0], :image1.shape[1]]

    # Clip pixel values to be within the valid range
    blended_image = np.clip(blended_image, 0, 255)

    return blended_image


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Pyramid Blending demo.")
    parser.add_argument("--image1", type=Path, help="Path to input image 1.")
    parser.add_argument("--image2", type=Path, help="Path to input image 2.")
    parser.add_argument("--mask", type=Path, help="Path to binary mask image.")
    parser.add_argument(
        "-k",
        "--levels",
        type=int,
        default=None,
        help="Number of levels in pyramid. If unspecified, k=log2(min(h, w)).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Path to save output image."
    )
    args = parser.parse_args()

    if not args.image1.exists():
        raise FileNotFoundError(f"Could not find image {args.image1}")
    if not args.image2.exists():
        raise FileNotFoundError(f"Could not find image {args.image2}")
    if not args.mask.exists():
        raise FileNotFoundError(f"Could not find image {args.mask}")

    image1 = uint8_to_float32(cv.imread(str(args.image1)))
    image2 = uint8_to_float32(cv.imread(str(args.image2)))
    mask = uint8_to_float32(cv.imread(str(args.mask), cv.IMREAD_GRAYSCALE))

    # Check sizes of input images
    h, w = image1.shape[:2]
    if image1.shape != image2.shape:
        raise ValueError("Input images must be the same size and number of channels.")
    if image1.shape[:2] != mask.shape[:2]:
        raise ValueError("Mask must be the same size as input images.")

    if args.levels is None:
        # Max # of levels is the number of times we can divide the smallest dimension by 2
        args.levels = int(np.log2(min(h, w)))
        print(f"Using k={args.levels} levels in the pyramid.")

    # Call the blending routine
    blend = blend_images(image1, image2, mask, args.levels)

    # Display the result
    tiled = np.vstack(
        [np.hstack([image1, image2]), np.hstack([cv.cvtColor(mask, cv.COLOR_GRAY2BGR), blend])]
    )

    if args.output is not None:
        cv.imwrite(str(args.output), float32_to_uint8(blend))
    else:
        cv.imshow("Pyramid Blending", tiled)
        cv.waitKey(0)
        cv.destroyAllWindows()
