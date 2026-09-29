"""Selected prototype geometry. Pixels for angles; organizer spacing for lengths."""
import numpy as np
from scipy import ndimage as ndi
from .ct_hip_geometry import shaft_axis, measure_protrusion

SPACING = (1.05, 0.6)


def largest(mask):
    labels, n = ndi.label(mask)
    if not n:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == sizes.argmax()


def aggregate(values):
    """OR with an explicit unknown state; unknown is never silently normal."""
    values = list(values)
    if any(v is True for v in values):
        return True
    return None if not values or any(v is None for v in values) else False


def lt_position(femur, lt):
    # Same dimensionless definition as the selected CT-v4 calibration.
    if lt.sum() < 10:
        return None
    rr = np.flatnonzero(lt.any(1))
    ax = shaft_axis(femur, int(rr.max() + 8 / .65), 1)
    if ax is None:
        rows = np.flatnonzero(femur.any(1))
        ax = shaft_axis(femur, int(rows.min() + .45 * np.ptp(rows)), 1) if len(rows) else None
    if ax is None:
        return None
    k, b, hw, _ = ax
    r, c = np.nonzero(lt)
    return float(np.mean((c - k * r - b) / np.sqrt(1 + k*k)) / (hw / .65))


def physical_protrusion(femur, mask, axis, spacing=SPACING):
    """Measure along/perpendicular to the shaft after transforming coordinates to mm."""
    sy, sx = spacing
    k, b, _, _ = axis
    tangent = np.array([sy, k*sx]); tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])  # medial = right in canonical view
    rr, cc = np.nonzero(mask)
    if not len(rr):
        return {'length_mm': 0., 'depth_mm': 0., 'area_mm2': 0.}
    coords = np.column_stack([rr*sy, cc*sx])
    along, perp = coords @ tangent, coords @ normal
    fr, fc = np.nonzero(femur & ~mask)
    base = np.column_stack([fr*sy, fc*sx])
    ba, bp = base @ tangent, base @ normal
    width = float(abs(tangent[0])*sy + abs(tangent[1])*sx)
    best = 0.; supported = 0
    # Row-sized physical bins preserve local comparison without isotropic assumptions.
    for bin_id in np.unique(np.round(along / width)):
        sel = np.abs(along - bin_id*width) <= width/2
        support = np.abs(ba - bin_id*width) <= width/2
        if sel.any() and support.any():
            supported += int(sel.sum())
            best = max(best, float(perp[sel].max() - bp[support].max()))
    return {'length_mm': float(np.ptp(along) + width),
            'depth_mm': best if supported else None,
            'area_mm2': float(mask.sum()*sx*sy)}


def trochanter_geometry(femur, lt, protrusion, spacing=SPACING):
    center = lt_position(femur | lt, lt)
    old = measure_protrusion(femur | protrusion, protrusion, 1, min_px=1)
    if old is None:
        return {'center_halfwidth': center, 'depth_mm': None, 'length_mm': None,
                'violation': None}, np.zeros_like(protrusion)
    mask = old['prot_mask']
    dims = physical_protrusion(femur | protrusion, mask, old['axis'], spacing)
    depth, length = dims['depth_mm'], dims['length_mm']
    shape_ok = bool(depth is not None and length >= 2 and depth >= 1 and length > depth)
    # Do not silently call missing whole-LT localization a normal result.
    violation = aggregate([None if center is None else bool(center < .70),
                           None if depth is None else bool(depth > 5.25)])
    return dict(center_halfwidth=center, **dims, violation=violation,
                visible_shape=shape_ok, threshold_depth_mm=5.25,
                threshold_center_halfwidth=.70, spacing_yx_mm=list(spacing),
                legacy_depth_mm=old['depth_mm'],
                threshold_status='Selected 5.25 threshold; organizer-spacing integration needs revalidation'), mask


def iliac_geometry(prob, channels, spacing=SPACING):
    h, w = prob.shape[1:]; ix = {n:i for i,n in enumerate(channels)}
    l2 = prob[ix['L2']] > .5
    ref = int(np.flatnonzero(l2.any(1)).max()) if l2.sum() > 300 else int(.45*h)
    out, masks = {}, {}
    for side in ('L', 'R'):
        ids, n = ndi.label(prob[ix['iliac_'+side]] > .5)
        candidates = [ids == j for j in range(1, n+1)
                      if np.nonzero(ids == j)[0].min() >= ref]
        mask = max(candidates, key=np.sum) if candidates else np.zeros((h,w), bool)
        area = float(mask.sum()*spacing[0]*spacing[1]); present = area > 50
        mask = mask if present else np.zeros_like(mask)
        curve = [[int(x), int(np.flatnonzero(mask[:,x])[0])]
                 for x in np.flatnonzero(mask.any(0))]
        out[side] = {'present': present, 'area_mm2': area, 'upper_contour': curve,
                     'ref_row': ref, 'area_threshold_mm2': 50.}
        masks[side] = mask
    return out, masks


def l1_margin(prob, channels, spacing=SPACING):
    ix = {n:i for i,n in enumerate(channels)}
    names = ['T10','T11','T12','T13','L1','L2','L3','L4','L5','L6']
    vp = np.stack([prob[ix[n]] for n in names]); labels = vp.argmax(0)
    mask = largest((labels == names.index('L1')) & (vp.max(0) > .5))
    if mask.sum() < 50:
        return {'violation': None, 'reason': 'L1 не найден'}
    yy, xx = np.nonzero(mask); center = float(np.median(xx))
    mask &= np.abs(np.arange(mask.shape[1]) - center)[None,:] < 25/.6
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return {'violation': None, 'reason': 'Границы L1 не определены'}
    top, bottom = int(yy.min()), int(yy.max())
    # A cut L1 cannot provide its full height; zero margin is nevertheless a violation.
    height = float((bottom-top+1)*spacing[0]); margin = float(top*spacing[0])
    return {'violation': bool(margin < height/2), 'margin_mm': margin,
            'height_mm': height, 'required_margin_mm': height/2,
            'image_top_row': 0, 'top_px': top, 'bottom_px': bottom}


def ct_top_rule(prob, channels, raw, spacing=SPACING):
    """Reviewed 15 mm OR rule, measured from image row zero in native coordinates."""
    sy,sx=spacing
    ix={n:i for i,n in enumerate(channels)}
    names=['T10','T11','T12','T13','L1','L2','L3','L4','L5','L6']
    vp=np.stack([prob[ix[n]] for n in names]);arg=vp.argmax(0);anyv=vp.max(0)>.5
    cx=float(np.median(np.nonzero(anyv)[1])) if anyv.any() else raw.shape[1]/2
    band=np.abs(np.arange(raw.shape[1])-cx)<25/sx
    bodies={}
    for j,n in enumerate(names):
        m=largest(anyv&(arg==j))&band[None,:]
        bodies[n]=m if m.sum()>50 else None
    l1,t12=bodies['L1'],bodies['T12']
    center=lambda m:float(np.nonzero(m)[0].mean())
    lower=[bodies[n] for n in ['L2','L3','L4','L5'] if bodies[n] is not None]
    order_bad=bool((t12 is not None and l1 is not None and center(t12)>center(l1)) or
                   (t12 is not None and any(center(m)<center(t12) for m in lower)) or
                   (l1 is not None and any(center(m)<center(l1) for m in lower)))
    height=float((np.ptp(np.nonzero(t12)[0])+1)*sy) if t12 is not None else None
    margin=float(np.nonzero(l1)[0].min()*sy) if l1 is not None else None
    normal=(height is not None and height>15) or (margin is not None and margin>15)
    violation=False if normal else (True if l1 is not None else None)
    fmt=lambda v:'не определено' if v is None else f'{v:.2f} мм'
    reason=f'Видимая высота Th12: {fmt(height)}; отступ над L1: {fmt(margin)}. Норма: хотя бы одно значение >15 мм'
    if violation is None:reason+='; недостаточно ориентиров для оценки'
    if order_bad:reason+='; требуется проверка нумерации позвонков'
    return dict(violation=violation,t12_height_mm=height,l1_top_margin_mm=margin,
                numbering_inconsistent=order_bad,
                missing_landmarks=[n for n in ('T12','L1') if bodies[n] is None],
                threshold_mm=15.,rule='T12 visible height >15mm OR image-edge-to-L1-top margin >15mm',
                edge='image row 0',spacing_yx_mm=[float(sy),float(sx)],reason=reason)
