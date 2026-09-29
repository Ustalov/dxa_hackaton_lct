"""PNG-only adapter for Yi et al. ISBI 2020; upstream architecture is unchanged.

Detections have an ordinal, NOT an anatomical level. Lumbar reference IDs are
known only in the prepared PNG data. Never infer L1 solely from the first peak.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from vendor.vertebra_landmark_detection.models.spinal_net import SpineNet


def make_model(checkpoint=None):
    model = SpineNet(heads={'hm': 1, 'reg': 2, 'wh': 8}, pretrained=False,
                     down_ratio=4, final_kernel=1, head_conv=256)
    if checkpoint is not None:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = ck['state_dict']
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    return model


def order_corners(points):
    """Upstream order: top-left, top-right, bottom-left, bottom-right."""
    p = np.asarray(points, dtype=np.float32).reshape(4, 2)
    left, right = p[np.argsort(p[:, 0])[:2]], p[np.argsort(p[:, 0])[2:]]
    left, right = left[np.argsort(left[:, 1])], right[np.argsort(right[:, 1])]
    return np.stack([left[0], right[0], left[1], right[1]])


def mask_corners(mask, min_iou=.85):
    """Conservative quadrilateral approximation, NOT verified endplate corners."""
    mask = mask.astype('uint8')
    if mask.sum() < 40 or any((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any())):
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    component = np.zeros_like(mask); cv2.drawContours(component, [contour], -1, 1, -1)
    if np.sum((mask > 0) & (component > 0)) / mask.sum() < .98:
        return None
    perimeter = cv2.arcLength(contour, True); candidates = []
    for fraction in (.005, .01, .015, .02, .03, .04, .06, .08):
        polygon = cv2.approxPolyDP(contour, fraction * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            candidates.append(polygon[:, 0])
    # An oriented rectangle is a fallback proposal, accepted by the same IoU rule.
    candidates.append(cv2.boxPoints(cv2.minAreaRect(contour)))
    best = None
    for polygon in candidates:
        q = order_corners(polygon)
        if not ((q[[1, 3], 0] > q[[0, 2], 0]).all() and (q[[2, 3], 1] > q[[0, 1], 1]).all()):
            continue
        h, w = mask.shape
        if (q < 0).any() or (q[:, 0] >= w).any() or (q[:, 1] >= h).any():
            continue
        raster = np.zeros_like(mask)
        cv2.fillConvexPoly(raster, np.rint(q[[0, 1, 3, 2]]).astype('int32'), 1)
        iou = np.sum((raster > 0) & (mask > 0)) / max(1, np.sum((raster > 0) | (mask > 0)))
        if iou >= min_iou and (best is None or iou > best['mask_quad_iou']):
            best = {'points': q.tolist(), 'mask_quad_iou': float(iou),
                    'reference_kind': 'automatic contour quadrilateral; not expert endplate landmarks'}
    return best


def letterbox(image, size=(512, 256)):
    h, w = image.shape; target_h, target_w = size
    scale = min(target_h / h, target_w / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    y, x = (target_h - nh) // 2, (target_w - nw) // 2
    result = np.zeros(size, dtype='uint8')
    result[y:y+nh, x:x+nw] = cv2.resize(image, (nw, nh))
    # Use the actual rounded scale independently on both axes.
    transform = {'scale_xy': [nw / w, nh / h], 'offset_xy': [x, y], 'native_shape': [h, w]}
    return result, transform


def map_points(points, transform, inverse=False):
    p = np.asarray(points, dtype=np.float32)
    s, o = np.array(transform['scale_xy']), np.array(transform['offset_xy'])
    return ((p - o) / s if inverse else p * s + o).astype('float32')


def targets(corners, input_size=(512, 256), max_objects=4, down_ratio=4, corner_visible=None):
    """Variable-count targets; no artificial points for unannotated levels."""
    pts = np.asarray(corners, dtype='float32').reshape(-1, 4, 2) / down_ratio
    visible=np.ones(pts.shape[:2],bool) if corner_visible is None else np.asarray(corner_visible,dtype=bool)
    if visible.shape!=pts.shape[:2]:raise ValueError('Invalid corner visibility shape')
    if len(pts) > max_objects:
        raise ValueError('Too many annotated vertebrae')
    h, w = np.array(input_size) // down_ratio
    hm = np.zeros((1, h, w), 'float32'); valid = np.zeros_like(hm)
    wh = np.zeros((max_objects, 8), 'float32'); reg = np.zeros((max_objects, 2), 'float32')
    ind = np.zeros(max_objects, 'int64'); mask = np.zeros(max_objects, 'float32')
    wh_mask=np.zeros((max_objects,8),'float32')
    yy, xx = np.mgrid[:h, :w]
    if len(pts):
        if ((pts[visible] < 0).any() or (pts[..., 0][visible] >= w).any() or (pts[..., 1][visible] >= h).any()):
            raise ValueError('Corner outside input image; do not clamp cropped vertebrae')
        # Outside the annotated lumbar band other visible vertebrae are unknown.
        low = max(0, int(np.floor(pts[..., 1].min())))
        high = min(h, int(np.ceil(pts[..., 1].max())) + 1)
        valid[:, low:high] = 1
    for k, q in enumerate(pts):
        center = q.mean(0); integer = np.floor(center).astype(int)
        if not (0<=integer[0]<w and 0<=integer[1]<h):continue
        ix = int(integer[1] * w + integer[0])
        if ix in ind[:k][mask[:k]>0]:
            raise ValueError('Two centers collide at model resolution')
        sigma = max(1., min(np.ptp(q[:, 0]), np.ptp(q[:, 1])) / 6)
        gaussian = np.exp(-((xx - integer[0])**2 + (yy - integer[1])**2) / (2 * sigma**2))
        gaussian[gaussian < 1e-4] = 0
        hm[0] = np.maximum(hm[0], gaussian)
        ind[k], reg[k], wh[k], mask[k] = ix, center - integer, (center - q).reshape(8), 1
        wh_mask[k]=np.repeat(visible[k],2)
    return dict(hm=hm, hm_valid=valid, wh=wh, wh_mask=wh_mask, reg=reg, ind=ind, reg_mask=mask)


class LumbarPNG(Dataset):
    def __init__(self, manifest, partition, size=(512, 256)):
        self.rows = [r for r in json.loads(Path(manifest).read_text()) if r['partition'] == partition]
        self.size = size
        for r in self.rows:
            if r.get('dicom_used') or not r['source_path'].startswith('/data/dxa-open-datasets/') or not r['source_path'].lower().endswith('.png'):
                raise ValueError('Only open PNG sources are permitted')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with np.load(r['data_path']) as z:
            image = z['image'].copy()
        image, transform = letterbox(image, self.size)
        corners = map_points(r['corners'], transform)
        result = targets(corners, self.size)
        result['input'] = np.repeat(image[None].astype('float32') / 255 - .5, 3, axis=0)
        return {k: torch.from_numpy(v) for k, v in result.items()}


def landmark_loss(output, target):
    p = output['hm'].float().clamp(1e-5, 1 - 1e-5)
    gt = target['hm']; valid = target['hm_valid']
    pos, neg = (gt == 1).float() * valid, (gt < 1).float() * valid
    focal = -(p.log() * (1-p)**2 * pos + (1-p).log() * p**2 * (1-gt)**4 * neg).sum() / pos.sum().clamp_min(1)
    result = focal
    for name in ('wh', 'reg'):
        field = output[name].float().flatten(2).transpose(1, 2)
        pred = field.gather(1, target['ind'][..., None].expand(-1, -1, field.shape[-1]))
        weight = target['reg_mask'][..., None].expand_as(pred)
        if name=='wh' and 'wh_mask' in target:weight=weight*target['wh_mask']
        result = result + ((pred - target[name]).abs() * weight).sum() / weight.sum().clamp_min(1)
    return result


@torch.no_grad()
def decode(output, transforms, max_detections=12, confidence=.2, down_ratio=4):
    """Return up to K actual candidates, never force 4/17 or attach L1-L5 IDs."""
    heat = output['hm'].float()
    if heat.shape[1] != 1:
        raise ValueError('This adapter expects the single class upstream checkpoint')
    peaks = heat * (heat == F.max_pool2d(heat, 3, stride=1, padding=1))
    h, w = heat.shape[-2:]; results = []
    for b, transform in enumerate(transforms):
        score, indices = peaks[b, 0].flatten().topk(min(max_detections, h*w))
        keep = score >= confidence; score, indices = score[keep], indices[keep]
        xy = torch.stack([indices % w, indices // w], -1).float()
        reg = output['reg'][b].float().flatten(1)[:, indices].T
        offsets = output['wh'][b].float().flatten(1)[:, indices].T.reshape(-1, 4, 2)
        center = (xy + reg) * down_ratio
        corners = center[:, None] - offsets * down_ratio
        native_center = map_points(center.cpu().numpy(), transform, inverse=True)
        native_corners = map_points(corners.cpu().numpy(), transform, inverse=True)
        candidates = []; nh, nw = transform['native_shape']
        for c, q, s in zip(native_center, native_corners, score.cpu().numpy()):
            if not (0 <= c[0] < nw and 0 <= c[1] < nh):
                continue
            candidates.append(dict(center=c.tolist(), corners=q.tolist(), score=float(s),
                                   anatomical_level=None, corners_outside_image=bool((q < 0).any() or (q[:, 0] >= nw).any() or (q[:, 1] >= nh).any())))
        candidates.sort(key=lambda c: c['center'][1])
        for i, candidate in enumerate(candidates):
            candidate['ordinal_top_to_bottom'] = i + 1
        results.append(candidates)
    return results
