"""2D helpers shared by training / real-image inference."""
import numpy as np
from scipy import ndimage as ndi
CANVAS = (352, 320)   # rows, cols ; pixel ~0.65 mm (Lunar)
PIX = 0.65


def to_canvas(a, side, fill=0):
    """mirror left hips so that medial is on the image right; pad/crop to CANVAS (top-left anchored)."""
    if side in ('L', 'left'):
        a = a[:, ::-1]
    out = np.full(CANVAS + a.shape[2:], fill, a.dtype)
    h, w = min(a.shape[0], CANVAS[0]), min(a.shape[1], CANVAS[1])
    out[:h, :w] = a[:h, :w]
    return out, (h, w)


def metrics_from_masks(fem, lt, medial_sign=1):
    """Purely 2D measurement: shaft axis = line through mid-points of the femur silhouette below the LT;
    visible LT = LT pixels medial to the shaft silhouette band. Returns dict (mm) or None."""
    fem = ndi.binary_opening(fem, iterations=1)
    lab, n = ndi.label(fem)
    if n == 0 or lt.sum() < 5:
        return None
    fem = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    lt = lt & ndi.binary_dilation(fem, iterations=3)
    if lt.sum() < 5:
        return None
    rows_lt = np.where(lt.any(1))[0]
    r0 = rows_lt.max() + int(8 / PIX); r1 = min(fem.shape[0] - 3, rows_lt.max() + int(45 / PIX))
    R, M, HW = [], [], []
    for r in range(r0, r1):
        c = np.where(fem[r])[0]
        if len(c) < 10:
            continue
        # take the run containing the femur shaft (longest run)
        runs = np.split(c, np.where(np.diff(c) > 1)[0] + 1)
        run = max(runs, key=len)
        R.append(r); M.append((run[0] + run[-1]) / 2); HW.append((run[-1] - run[0] + 1) / 2)
    if len(R) < 8:
        return None
    k, b = np.polyfit(R, M, 1)
    hw = float(np.median(HW)) * PIX
    rr, cc = np.nonzero(lt)
    # signed medial distance to the axis line (col = k*row + b), perpendicular
    sd = medial_sign * (cc - (k * rr + b)) / np.sqrt(1 + k * k) * PIX
    vis = sd > hw
    return dict(shaft_width_mm=2 * hw, lt_proj_area_mm2=float(lt.sum()) * PIX ** 2,
                lt_visible_area_mm2=float(vis.sum()) * PIX ** 2,
                lt_protrusion_mm=float(sd.max() - hw), lt_protrusion_frac_width=float((sd.max() - hw) / (2 * hw)),
                lt_height_mm=float((rr.max() - rr.min() + 1) * PIX),
                lt_prot_over_height=float((sd.max() - hw) / ((rr.max() - rr.min() + 1) * PIX)),
                axis_k=float(k), axis_b=float(b), axis_rows=(int(R[0]), int(R[-1])))


def shaft_axis(fem, below_row, medial_sign=1):
    """shaft mid-line (col = k*row + b) from femur silhouette rows below `below_row`."""
    R, M, HW = [], [], []
    for r in range(below_row, fem.shape[0] - 3):
        c = np.where(fem[r])[0]
        if len(c) < 10:
            continue
        runs = np.split(c, np.where(np.diff(c) > 1)[0] + 1); run = max(runs, key=len)
        R.append(r); M.append((run[0] + run[-1]) / 2); HW.append((run[-1] - run[0] + 1) / 2)
    if len(R) < 8:
        return None
    k, b = np.polyfit(R, M, 1)
    return k, b, float(np.median(HW)) * PIX, (R[0], R[-1])


def measure_protrusion(fem, prot, medial_sign=1, min_px=3):
    """fem: femur silhouette incl. LT; prot: protruding LT mask. Height = extent along the shaft axis (mm),
    depth = max perpendicular distance of the protrusion beyond the femur contour (mm), area (mm2)."""
    fem = ndi.binary_opening(fem, iterations=1)
    lab, n = ndi.label(fem)
    if n == 0:
        return None
    fem = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    lab, n = ndi.label(prot & ndi.binary_dilation(fem, iterations=4))
    if n:
        sizes = ndi.sum(lab > 0, lab, range(1, n + 1))
        keep = 1 + int(np.argmax(sizes))   # the lesser trochanter = largest protrusion component
        prot = (lab == keep) if sizes.max() >= min_px else np.zeros_like(prot)
    else:
        prot = np.zeros_like(prot)
    ref_row = int(np.where(prot.any(1))[0].max()) + int(8 / PIX) if prot.any() else int(fem.shape[0] * 0.55)
    ax = shaft_axis(fem, ref_row, medial_sign)
    if ax is None:
        ax = shaft_axis(fem, int(fem.shape[0] * 0.45), medial_sign)
    if ax is None:
        return None
    k, b, hw, rows = ax
    d = np.array([1.0, k]); d /= np.linalg.norm(d); nrm = np.array([-d[1], d[0]]) * np.sign(d[0])
    nrm = nrm if nrm[1] * medial_sign > 0 else -nrm
    res = dict(present=bool(prot.sum() >= min_px), area_mm2=float(prot.sum()) * PIX ** 2, height_mm=0.0, depth_mm=0.0,
               shaft_width_mm=2 * hw, axis=(float(k), float(b), int(rows[0]), int(rows[1])))
    if res['present']:
        rr, cc = np.nonzero(prot); along = rr * d[0] + cc * d[1]; perp = rr * nrm[0] + cc * nrm[1]
        res['height_mm'] = float(along.max() - along.min() + 1) * PIX
        fr, fc = np.nonzero(fem & ~prot); fal = fr * d[0] + fc * d[1]; fpe = fr * nrm[0] + fc * nrm[1]
        best = 0.0
        for bnd in np.unique(np.round(along)):
            sel = np.abs(along - bnd) <= 0.5; fs = np.abs(fal - bnd) <= 0.5
            if not sel.any():
                continue
            base = fpe[fs].max() if fs.any() else perp[sel].min() - 1
            best = max(best, perp[sel].max() - base)
        res['depth_mm'] = float(best) * PIX
        res['depth_frac_width'] = res['depth_mm'] / res['shaft_width_mm']
    res['prot_mask'] = prot
    return res
