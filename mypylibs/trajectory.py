import numpy as np


def calculate(img, num_samples=500):
    img = img > 0.5
    edge_y, edge_x = np.where(img)
    if len(edge_x) == 0:
        raise ValueError("no edge pixels found")

    points = np.column_stack((edge_x, edge_y)).astype(float)
    if len(points) > 2500:
        idx = np.linspace(0, len(points) - 1, 2500).astype(int)
        points = points[idx]

    ordered = [points[0]]
    leftover = points[1:].tolist()
    while leftover:
        last = ordered[-1]
        dists = np.sum((np.asarray(leftover) - last) ** 2, axis=1)
        j = int(np.argmin(dists))
        ordered.append(leftover.pop(j))
    ordered = np.asarray(ordered)

    seg = np.sqrt(np.sum(np.diff(ordered, axis=0) ** 2, axis=1))
    cum = np.insert(np.cumsum(seg), 0, 0.0)
    if cum[-1] == 0:
        raise ValueError("degenerate contour")
    targets = np.linspace(0, cum[-1], num_samples, endpoint=False)
    x = np.interp(targets, cum, ordered[:, 0])
    y = np.interp(targets, cum, ordered[:, 1])
    return x, y
