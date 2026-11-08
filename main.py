import yaml
import numpy as np
from PIL import Image
from os import path as opath
import matplotlib.pyplot as plt

from mypylibs import filters, trajectory, fourier

ROOT_IN = "in"
ROOT_OUT = "out"
EXT = "png"


def load_gray(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=float) / 255.0
    return arr


def run_pipeline(gray, params):
    steps = [("grayscale", gray)]

    blurred = filters.gaussian_blur(
        gray,
        size=int(params.get("blur_size", 5)),
        sigma=float(params.get("blur_sigma", 1.5)),
    )
    steps.append(("gaussian blur", blurred))

    work = blurred
    if params.get("invert", False):
        work = filters.invert(work)
        steps.append(("invert", work))

    binary = filters.threshold(work, float(params.get("threshold", 0.5)))
    steps.append(("threshold", binary))

    border = filters.binary_border(binary)
    steps.append(("border", border))
    return steps, border


def save_transforms(case, steps):
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (title, im) in zip(axes, steps):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(opath.join(ROOT_OUT, f"{case:02}_transforms.{EXT}"), dpi=120)
    plt.close(fig)


def save_trajectory(case, x, y):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    n = len(x)
    for i, ax in enumerate(axes):
        end = max(2, int((i + 1) / 10 * n))
        ax.plot(x[:end], y[:end], "b-", lw=1)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"t={(i+1)*10}%", fontsize=8)
        ax.axis("off")
    fig.suptitle("boundary trajectory", fontsize=11)
    fig.tight_layout()
    fig.savefig(opath.join(ROOT_OUT, f"{case:02}_trajectory.{EXT}"), dpi=120)
    plt.close(fig)


def save_final(case, x, y):
    z = fourier.to_complex(x, y)
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        frac = (i + 1) * 0.05
        xr, yr, k = fourier.reconstruct(z, frac)
        ax.plot(xr, yr, "b-", lw=1)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"{int(frac*100)}% (K={k})", fontsize=8)
        ax.axis("off")
    fig.suptitle("fourier reconstruction", fontsize=11)
    fig.tight_layout()
    fig.savefig(opath.join(ROOT_OUT, f"{case:02}_final.{EXT}"), dpi=120)
    plt.close(fig)


def main(case: int):
    img_path = opath.join(ROOT_IN, f"{case:02}.{EXT}")
    cfg_path = opath.join(ROOT_IN, f"{case:02}.yaml")
    gray = load_gray(img_path)
    with open(cfg_path) as f:
        params = yaml.safe_load(f) or {}

    steps, border = run_pipeline(gray, params)
    save_transforms(case, steps)

    x, y = trajectory.calculate(border, num_samples=int(params.get("num_samples", 400)))
    save_trajectory(case, x, y)
    save_final(case, x, y)
    print(f"case {case:02} done, contour samples={len(x)}")


if __name__ == "__main__":
    case = 0
    while True:
        try:
            main(case)
        except FileNotFoundError:
            if case == 0:
                raise
            break
        case += 1
