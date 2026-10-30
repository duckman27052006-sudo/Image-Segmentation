"""Cac thuat toan phan vung anh: No vung va Gop vung (Split and Merge)."""

import cv2
import numpy as np


def _gray(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return image.astype(np.float32)


def region_growing(image, seeds, threshold, connectivity=8):
    """No vung (region growing) tu nhieu diem seed.

    Moi diem seed mo rong dan sang cac diem lan can co chenh lech mau
    khong vuot qua `threshold` (0-255). Phat trien dong thoi nhieu seed,
    diem nao duoc chiem truoc thi thuoc vung do.

    Tra ve label map int32, -1 la diem chua duoc gan vao vung nao.
    """
    if image.ndim == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img = np.ascontiguousarray(image).copy()
    h, w = img.shape[:2]
    labels = np.full((h, w), -1, dtype=np.int32)

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask[0, :] = 255
    mask[-1, :] = 255
    mask[:, 0] = 255
    mask[:, -1] = 255

    lo = up = (float(max(threshold, 0)),) * img.shape[2]
    region_id = 0

    for sy, sx in seeds:
        sy = int(np.clip(sy, 0, h - 1))
        sx = int(np.clip(sx, 0, w - 1))
        if labels[sy, sx] != -1:
            continue
        if region_id >= 250:
            break

        region_id += 1
        mask[1:-1, 1:-1][labels != -1] = 255
        flags = (int(connectivity) | cv2.FLOODFILL_MASK_ONLY | (region_id << 8))
        cv2.floodFill(img, mask, (sx, sy), 0, lo, up, flags)
        filled = mask[1:-1, 1:-1] == region_id
        labels[filled] = region_id - 1

    return labels


def fill_unassigned(labels):
    """Gan cac diem chua co nhan (-1) cho vung da co gan nhat."""
    import scipy.ndimage as ndi

    mask = labels < 0
    if not mask.any() or mask.all():
        return labels
    _, (iy, ix) = ndi.distance_transform_edt(mask, return_indices=True)
    return labels[iy, ix]


def split_and_merge(image, split_threshold=22, merge_threshold=12,
                    min_size=150, min_block=4):
    """Phan vung bang tach (quadtree) roi gop cac vung tuong tu.

    - Tach: chia doi moi khoang anh (4 phan) khi do lech max-min > split_threshold.
    - Gop: gop hai vung ke nhau co chenh lech trung binh <= merge_threshold,
      hoac khi mot vung nho hon min_size.
    """
    gray = _gray(image)
    h, w = gray.shape

    leaves = []
    stack = [(0, h, 0, w)]
    while stack:
        y0, y1, x0, x1 = stack.pop()
        block = gray[y0:y1, x0:x1]
        if (y1 - y0) > min_block and (x1 - x0) > min_block:
            if float(block.max() - block.min()) > split_threshold:
                ym = (y0 + y1) // 2
                xm = (x0 + x1) // 2
                stack.append((y0, ym, x0, xm))
                stack.append((y0, ym, xm, x1))
                stack.append((ym, y1, x0, xm))
                stack.append((ym, y1, xm, x1))
                continue
        leaves.append((y0, y1, x0, x1))

    labels = np.empty((h, w), dtype=np.int32)
    for i, (y0, y1, x0, x1) in enumerate(leaves):
        labels[y0:y1, x0:x1] = i
    labels = _relabel(labels)

    for _ in range(60):
        labels, changed = _merge_pass(labels, gray, merge_threshold, min_size)
        if not changed:
            break

    return labels


def _adjacent_pairs(labels, n):
    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    keep = a != b
    a = a[keep].astype(np.int64)
    b = b[keep].astype(np.int64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    enc = np.unique(lo * n + hi)
    return (enc // n).astype(np.int64), (enc % n).astype(np.int64)


def _merge_pass(labels, gray, merge_threshold, min_size):
    n = int(labels.max()) + 1
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=n).astype(np.float64)
    sums = np.bincount(flat, weights=gray.ravel().astype(np.float64), minlength=n)
    means = sums / np.maximum(areas, 1)

    lo, hi = _adjacent_pairs(labels, n)
    if lo.size == 0:
        return labels, False

    diffs = np.abs(means[lo] - means[hi])
    order = np.argsort(diffs, kind="stable")

    parent = np.arange(n)
    root_area = areas.copy()

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    changed = False
    for k in order:
        ra = find(int(lo[k]))
        rb = find(int(hi[k]))
        if ra == rb:
            continue
        should_merge = diffs[k] <= merge_threshold
        if not should_merge and min_size > 0:
            should_merge = min(root_area[ra], root_area[rb]) < min_size
        if not should_merge:
            continue
        if root_area[ra] < root_area[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        root_area[ra] += root_area[rb]
        changed = True

    if not changed:
        return labels, False

    lookup = np.array([find(i) for i in range(n)], dtype=np.int32)
    return _relabel(lookup[labels]), True


def _relabel(labels):
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def _boundary_mask(labels):
    b = np.zeros(labels.shape, dtype=bool)
    diff = labels[:, :-1] != labels[:, 1:]
    b[:, :-1] |= diff
    b[:, 1:] |= diff
    diff = labels[:-1, :] != labels[1:, :]
    b[:-1, :] |= diff
    b[1:, :] |= diff
    return b


def render_result(image, labels, filled=True, show_boundaries=True, seed=12345):
    """To mau cac vung va ve bien de hien thi ket qua (mang RGB uint8)."""
    if image.ndim == 2:
        base = np.stack([image] * 3, axis=-1)
    else:
        base = image.copy()

    if filled:
        n = max(int(labels.max()) + 1, 1)
        rng = np.random.default_rng(seed)
        palette = rng.integers(70, 256, size=(n, 3), dtype=np.uint8)
        out = np.zeros_like(base)
        valid = labels >= 0
        out[valid] = palette[labels[valid]]
    else:
        out = base

    if show_boundaries:
        out[_boundary_mask(labels)] = (0, 0, 0)

    return out
