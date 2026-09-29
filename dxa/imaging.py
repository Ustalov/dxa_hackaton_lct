from pathlib import Path
import hashlib
import warnings
import cv2
import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter, median_filter
from skimage.feature import hog

REGIONS = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}
TARGETS = ["quality_spine", "quality_hip", "spine_position", "spine_axis", "spine_artifact", "hip_position", "hip_roi"]
VIOLATIONS = ["Некорректная укладка", "Не выровнена ось позвоночника", "Присутствуют посторонние предметы", "Некорректная укладка", "Некорректная область интереса"]

def read_dicom(path, *, infer_region=True):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = pydicom.dcmread(path)
        if not ds.get("StudyInstanceUID") or not ds.get("SOPInstanceUID"):
            raise ValueError("Missing StudyInstanceUID or SOPInstanceUID")
        if int(ds.get("NumberOfFrames", 1)) != 1:
            raise ValueError("Multi-frame DICOM is outside the supported acquisition profile")
        if "prodigy" not in str(ds.get("ManufacturerModelName", "")).lower():
            raise ValueError("Unsupported scanner: model is validated only for the supplied Lunar Prodigy profile")
        if int(ds.get("Rows", 0)) * int(ds.get("Columns", 0)) > 16_000_000:
            raise ValueError("Image exceeds pixel limit")
        pixels = ds.pixel_array
    if pixels.ndim != 2 or min(pixels.shape) < 32:
        raise ValueError("Only single-frame grayscale images >=32 pixels are supported")
    if ds.get("PhotometricInterpretation") not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError("Unsupported photometric interpretation")
    raw_hash = hashlib.sha256(str((pixels.shape, str(pixels.dtype))).encode() + pixels.tobytes()).hexdigest()
    a = pixels.astype(np.float32)
    if not np.isfinite(a).all() or a.max() <= a.min():
        raise ValueError("Nonfinite or constant pixel array")
    # Preserve the 8-bit acquisition intensities; normalize other storage depths.
    if pixels.dtype != np.uint8:
        a = (a - a.min()) * (255.0 / (a.max() - a.min()))
    if ds.PhotometricInterpretation == "MONOCHROME1":
        a = 255 - a
    spacing = ds.get("PixelSpacing")
    if spacing is not None:
        sy, sx = map(float, spacing)
        if not np.isfinite([sy, sx]).all() or min(sy, sx) <= 0:
            raise ValueError("Invalid PixelSpacing")
        spacing_source = "DICOM PixelSpacing"
    elif "prodigy" in str(ds.get("ManufacturerModelName", "")).lower():
        sy, sx = 1.05, 0.6
        spacing_source = "Organizer clarification: Lunar Prodigy, Y=1.05 X=0.6 mm"
    else:
        raise ValueError("No PixelSpacing and scanner is outside the calibrated domain")
    width = a.shape[1]
    # Explicit organizer-approved acquisition profile, not a universal anatomy classifier.
    if not infer_region:
        # The prototype assigns region/side using its image classifier after decoding.
        region, side, confidence = 'unknown', '', 0.0
    elif width == 300:
        region, side, confidence = "spine", "", 1.0
    elif width in (280, 248):
        region = "hip"
        def center(y0, y1):
            strip = a[int(y0*len(a)):int(y1*len(a))]
            weights = np.maximum(strip - 90, 0).sum(0)
            return float(np.dot(weights, np.arange(width)) / max(weights.sum(), 1)) / width
        # Relative proximal/distal displacement is invariant to lateral translation.
        delta = center(.06, .27) - center(.65, .93)
        side = "right" if delta > 0 else "left"
        confidence = min(1., abs(delta) / .1)
    else:
        raise ValueError(f"Unsupported acquisition width: {width}; expected 300/280/248")
    return a.astype(np.uint8), {
        "study_uid": str(ds.StudyInstanceUID), "image_uid": str(ds.SOPInstanceUID),
        "pixel_hash": raw_hash, "region": region, "side": side,
        "side_confidence": confidence, "spacing": [sy, sx], "spacing_source": spacing_source,
        "metadata_warning_count": len(caught), "shape": list(a.shape),
    }

def letterbox(a, size=224, spacing=(1.05, .6)):
    """Whole field, physical aspect ratio, no anatomical crop."""
    h, w = a.shape
    sy, sx = spacing
    scale = size / max(h*sy, w*sx)
    nh, nw = max(1, round(h*sy*scale)), max(1, round(w*sx*scale))
    out = np.zeros((size, size), np.uint8)
    y, x = (size-nh)//2, (size-nw)//2
    out[y:y+nh, x:x+nw] = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    return out

def canonical(a, meta):
    return np.fliplr(a).copy() if meta["side"] == "left" else a

def geometry(a, meta):
    """Exploratory intensity centerline: never labeled Cobb or vertebral localization."""
    h, w = a.shape
    sy, sx = meta["spacing"]
    sm = gaussian_filter(a.astype(float), sigma=(2, 4))
    ys = np.arange(int(.12*h), int(.84*h), max(1, h//60))
    lo, hi = int(.22*w), int(.78*w)
    # Track a bright ridge with a smoothness penalty; no label information is used.
    sub = sm[ys, lo:hi]
    sub = (sub - np.median(sub, axis=1, keepdims=True)) / (np.std(sub, axis=1, keepdims=True)+1)
    grid = np.arange(hi-lo)
    transition = .025 * (grid[:, None] - grid[None, :])**2
    scores = sub[0].copy()
    pointers = []
    for row in sub[1:]:
        costs = scores[:, None] - transition
        pointers.append(costs.argmax(0))
        scores = row + costs.max(0)
    track = [int(scores.argmax())]
    for ptr in reversed(pointers):
        track.append(int(ptr[track[-1]]))
    xs = median_filter(np.array(track[::-1], float) + lo, size=5)
    slope, intercept = np.polyfit(ys*sy, xs*sx, 1)
    angle = float(np.degrees(np.arctan(slope)))
    residual = xs*sx - (slope*ys*sy+intercept)
    rms = float(np.sqrt(np.mean(residual**2)))
    contrast = float(np.mean(sub[np.arange(len(ys)), np.clip(xs.astype(int)-lo, 0, hi-lo-1)]))
    reliable = bool(contrast > .8 and np.mean((xs <= lo+3) | (xs >= hi-4)) < .15)
    result = {"axis_tilt_deg": angle if meta["region"] == "spine" else None,
              "centerline_residual_mm": rms if meta["region"] == "spine" else None,
              "axis_reliable": reliable and meta["region"] == "spine",
              "centerline": [[float(x), int(y)] for x, y in zip(xs, ys)] if meta["region"] == "spine" else [],
              "measurement_kind": "experimental intensity ridge, not Cobb angle",
              "spacing_source": meta["spacing_source"]}
    vals = [h*sy, w*sx, h*sy/(w*sx), a.mean()/255, a.std()/255,
            abs(angle), rms, contrast, float(reliable), float(meta["region"]=="spine")]
    for threshold in [20, 60, 100, 140, 180, 220, 250]:
        mask = a > threshold
        vals.append(mask.mean())
        for piece in [mask[:max(1,h//8)], mask[-max(1,h//8):], mask[:,:max(1,w//8)], mask[:,-max(1,w//8):]]:
            vals.append(piece.mean())
    for block in np.array_split(a, 4, axis=0):
        for tile in np.array_split(block, 4, axis=1):
            vals.extend([tile.mean()/255, tile.std()/255])
    return np.array(vals, np.float32), result

def handcrafted(a, meta):
    a = canonical(a, meta)
    box = letterbox(a, 128, meta["spacing"])
    descriptor = hog(box, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
    geo, detail = geometry(a, {**meta, "side": "right" if meta["region"]=="hip" else ""})
    return descriptor.astype(np.float32), geo, detail
