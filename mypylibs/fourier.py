import numpy as np


def to_complex(x, y):
    return x.astype(float) + 1j * y.astype(float)


def reconstruct(z, keep_frac):
    Z = np.fft.fft(z)
    n = len(Z)
    k = max(1, int(round(keep_frac * n)))
    order = np.argsort(np.abs(Z))[::-1]
    Z_hat = np.zeros_like(Z)
    Z_hat[order[:k]] = Z[order[:k]]
    z_rec = np.fft.ifft(Z_hat)
    return z_rec.real, z_rec.imag, k
