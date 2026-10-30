"""Cac thuat toan phan vung anh: No vung va Gop vung (Split and Merge)."""

import cv2
import numpy as np


def _gray(image):
    """Chuyen anh ve anh xam float32 de tinh do lech mau don gian.

    Cac thuat toan ben duoi chi can mot kenh do sang; dung float32 de tranh
    tran so khi cong/tru trung binh.
    """
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
    # floodFill cua OpenCV can anh 3 kenh (BGR) va mang lien tuc trong bo nho.
    if image.ndim == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img = np.ascontiguousarray(image).copy()
    h, w = img.shape[:2]

    # -1 = diem chua thuoc vung nao; cac gia tri >= 0 la id vung.
    labels = np.full((h, w), -1, dtype=np.int32)

    # Mask cua floodFill bat buoc co kich thuoc (h+2, w+2). Vien ngoai cung
    # duoc dat 255 de floodFill khong tran ra ngoai bien anh.
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask[0, :] = 255
    mask[-1, :] = 255
    mask[:, 0] = 255
    mask[:, -1] = 255

    # Nguong duoi/tren cho tung kenh mau: chap nhan chenh lech <= threshold.
    lo = up = (float(max(threshold, 0)),) * img.shape[2]
    region_id = 0

    for sy, sx in seeds:
        # Kep toa do seed vao trong anh de tranh loi khi nguoi dung nhap sai.
        sy = int(np.clip(sy, 0, h - 1))
        sx = int(np.clip(sx, 0, w - 1))
        # Seed nam trong vung da duoc chiem thi bo qua (vung nao chiem truoc thang).
        if labels[sy, sx] != -1:
            continue
        # Gia tri mask duoc ma hoa trong 8 bit, nen gioi han so vung < 250.
        if region_id >= 250:
            break

        region_id += 1
        # Danh dau cac diem da co nhan la 255 de floodFill khong lan vao vung cu.
        mask[1:-1, 1:-1][labels != -1] = 255
        # FLOODFILL_MASK_ONLY: chi to mask, khong doi mau anh.
        # region_id << 8: ghi gia tri region_id vao mask de phan biet tung lan loang.
        flags = (int(connectivity) | cv2.FLOODFILL_MASK_ONLY | (region_id << 8))
        cv2.floodFill(img, mask, (sx, sy), 0, lo, up, flags)
        # Cac diem vua duoc loang co dung gia tri region_id trong mask.
        filled = mask[1:-1, 1:-1] == region_id
        labels[filled] = region_id - 1

    return labels


def fill_unassigned(labels):
    """Gan cac diem chua co nhan (-1) cho vung da co gan nhat."""
    import scipy.ndimage as ndi

    mask = labels < 0
    # Khong co diem trong (da phu kin) hoac toan bo trong (khong co vung nao).
    if not mask.any() or mask.all():
        return labels
    # distance_transform_edt tra ve chi so cua diem "khac 0" gan nhat, tuc la
    # diem da co nhan gan nhat. Dung chi so do de "keo" nhan phu vao diem trong.
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

    # Buoc TACH: duyet cay quadtree bang ngan xep (stack) thay vi de quy.
    leaves = []
    stack = [(0, h, 0, w)]
    while stack:
        y0, y1, x0, x1 = stack.pop()
        block = gray[y0:y1, x0:x1]
        # Chi tach khi khoi con du lon va do lech max-min vuot nguong tach.
        if (y1 - y0) > min_block and (x1 - x0) > min_block:
            if float(block.max() - block.min()) > split_threshold:
                ym = (y0 + y1) // 2
                xm = (x0 + x1) // 2
                # Chia khoi thanh 4 khoi con va xu ly tiep.
                stack.append((y0, ym, x0, xm))
                stack.append((y0, ym, xm, x1))
                stack.append((ym, y1, x0, xm))
                stack.append((ym, y1, xm, x1))
                continue
        # Khoi khong tach duoc nua -> tro thanh mot "la" (vung ban dau).
        leaves.append((y0, y1, x0, x1))

    # Gan moi la mot nhan tam thoi, sau do danh so lai lien tuc tu 0.
    labels = np.empty((h, w), dtype=np.int32)
    for i, (y0, y1, x0, x1) in enumerate(leaves):
        labels[y0:y1, x0:x1] = i
    labels = _relabel(labels)

    # Buoc GOP: lap nhieu luot cho den khi khong con vung nao gop duoc.
    # Gioi han 60 luot de tranh vong lap vo han trong truong hop du lieu la.
    for _ in range(60):
        labels, changed = _merge_pass(labels, gray, merge_threshold, min_size)
        if not changed:
            break

    return labels


def _adjacent_pairs(labels, n):
    """Liet ke cac cap nhan (lo, hi) nam canh nhau theo 4-huong.

    Cac cap duoc ma hoa thanh so nguyen ``lo * n + hi`` de loai trung bang
    ``np.unique`` (moi bien gioi chi duoc xet mot lan du xuat hien nhieu diem).
    """
    # Ghep cac cap diem ke nhau theo chieu ngang va chieu doc.
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
    """Thuc hien mot luot gop vung, tra ve (labels_moi, da_thay_doi)."""
    n = int(labels.max()) + 1
    flat = labels.ravel()
    # Dien tich va tong do sang cua tung vung -> tinh trung binh mau.
    areas = np.bincount(flat, minlength=n).astype(np.float64)
    sums = np.bincount(flat, weights=gray.ravel().astype(np.float64), minlength=n)
    means = sums / np.maximum(areas, 1)

    lo, hi = _adjacent_pairs(labels, n)
    if lo.size == 0:
        return labels, False

    # Uu tien gop cac cap co chenh lech mau nho nhat truoc.
    diffs = np.abs(means[lo] - means[hi])
    order = np.argsort(diffs, kind="stable")

    # Union-find: parent[i] la goc cua tap chua vung i.
    parent = np.arange(n)
    root_area = areas.copy()

    def find(a):
        # Nen duong (path compression) de tang toc truy van goc.
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
        # Dieu kien gop: mau trung binh du gan, hoac mot trong hai vung qua nho.
        should_merge = diffs[k] <= merge_threshold
        if not should_merge and min_size > 0:
            should_merge = min(root_area[ra], root_area[rb]) < min_size
        if not should_merge:
            continue
        # Cho vung lon hon lam goc de cay union-find can doi hon.
        if root_area[ra] < root_area[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        root_area[ra] += root_area[rb]
        changed = True

    if not changed:
        return labels, False

    # Anh xa moi nhan cu ve goc dai dien, roi danh so lai tu 0.
    lookup = np.array([find(i) for i in range(n)], dtype=np.int32)
    return _relabel(lookup[labels]), True


def _relabel(labels):
    """Danh so lai cac nhan thanh 0, 1, 2, ... lien tuc."""
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def _boundary_mask(labels):
    """Tra ve mask True tai cac diem nam tren bien giua hai vung.

    Moi cap diem ke nhau khac nhan se danh dau ca hai phia, nho vay bien
    day 2 pixel va de nhin hon khi hien thi.
    """
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
        # Sinh bang mau ngau nhien nhung co dinh (theo `seed`) cho tung vung,
        # nho vay chay lai van cho mau giong nhau.
        n = max(int(labels.max()) + 1, 1)
        rng = np.random.default_rng(seed)
        palette = rng.integers(70, 256, size=(n, 3), dtype=np.uint8)
        out = np.zeros_like(base)
        valid = labels >= 0
        out[valid] = palette[labels[valid]]
    else:
        # Khong to mau: giu nguyen anh goc.
        out = base

    if show_boundaries:
        # Ve bien vung bang mau den.
        out[_boundary_mask(labels)] = (0, 0, 0)

    return out
