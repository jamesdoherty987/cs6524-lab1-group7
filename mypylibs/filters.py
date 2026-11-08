import numpy as np
from scipy import ndimage


def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def sobel_kernels():
    kx = np.array([[-1.0, 0.0, 1.0],
                   [-2.0, 0.0, 2.0],
                   [-1.0, 0.0, 1.0]])
    ky = np.array([[-1.0, -2.0, -1.0],
                   [0.0, 0.0, 0.0],
                   [1.0, 2.0, 1.0]])
    return kx, ky


def convolve(image, kernel):
    return ndimage.convolve(image.astype(float), kernel, mode="nearest")


def gaussian_blur(image, size=5, sigma=1.0):
    return convolve(image, gaussian_kernel(size, sigma))


def sobel_magnitude(image):
    kx, ky = sobel_kernels()
    gx = convolve(image, kx)
    gy = convolve(image, ky)
    mag = np.hypot(gx, gy)
    m = mag.max()
    if m > 0:
        mag = mag / m
    return mag


def threshold(image, value):
    return (image >= value).astype(float)


def invert(image):
    return 1.0 - image


def binary_border(mask):
    mask = mask.astype(bool)
    eroded = ndimage.binary_erosion(mask)
    border = mask & ~eroded
    return border.astype(float)
