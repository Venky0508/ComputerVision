import numpy as np


def uint8_to_float32(image: np.ndarray) -> np.ndarray:
    """Convert an image from uint8 format (ranging in [0,255]) to float format (ranging in [0,1]).
    """
    return image.astype(np.float32) / 255.0


def float32_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image from float format (ranging in [0,1]) to uint8 format (ranging in [0,255]).
    """
    return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)


def non_maximal_suppression(image: np.ndarray, block_size: int) -> np.ndarray:
    """Apply non-maximal suppression to an image.
    """
    pad = block_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="constant")
    stacks = [
        padded[i:image.shape[0] + i, j:image.shape[1] + j]
        for i in range(block_size)
        for j in range(block_size)
    ]
    return np.where(image == np.max(stacks, axis=0), image, 0)
