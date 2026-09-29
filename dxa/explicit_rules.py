"""Reviewable measurements and rules; no learned quality or foreign-body classifier."""
import copy
import cv2
import numpy as np
from scipy import ndimage as ndi
from .anatomy_measurements import measure, points
from .geometry_targets import feature

RULE_VERSION = 'explicit-v2'

def iliac_candidates(case, bone):
    """Connected lateral bone; measure its actual upper envelope before vertical clipping.

    Lateral search is >=20 mm outside L4 body. Its actual top must be below
    L3/L4. A cropped ROI edge must never manufacture a new horizontal crest.
    """
    sy, sx = case['spacing_yx_mm']; h, w = bone.shape
    p3 = points(case, 'L3_lower', 2); p4 = points(case, 'L4_upper', 2)
    b4 = points(case, 'L4_lower', 2); b5 = points(case, 'L5_lower', 2)
    found = []; masks = {}
    if p3 is None or p4 is None:
        return found, masks
    low = float((p3[:, 1].mean() + p4[:, 1].mean()) / 2)
    ref = np.concatenate([p4, b4]) if b4 is not None else p4
    yy, xx = np.mgrid[:h, :w]
    for side in ['left', 'right']:
        edge = float(ref[:, 0].min() if side == 'left' else ref[:, 0].max())
        lateral = xx <= edge - 20 / sx if side == 'left' else xx >= edge + 20 / sx
        region = ndi.binary_opening((bone > 0) & lateral, structure=np.ones((3, 3)))
        components, count = ndi.label(region)
        for k in range(1, count + 1):
            mask = components == k; cy, cx = np.where(mask)
            if len(cx) < 40:  # inherited anti-speckle floor, not an area acceptance criterion
                continue
            curve = np.array([[x, np.flatnonzero(mask[:, x])[0]] for x in np.unique(cx)], float)
            curve[:, 1] = ndi.median_filter(curve[:, 1], size=3)
            top = curve[curve[:, 1].argmin()]
            if top[1] <= low:
                continue
            delta = np.diff(curve, axis=0) * [sx, sy]
            # A jump between unrelated upper fragments is not contour length.
            breaks = (np.diff(curve[:, 0]) > 1) | (np.abs(np.diff(curve[:, 1])) * sy > 5)
            lengths = np.linalg.norm(delta, axis=1); longest = current = 0.
            for distance, broken in zip(lengths, breaks):
                if broken: current = 0.
                else: current += float(distance)
                longest = max(longest, current)
            level_known = b5 is not None
            level_ok = bool(level_known and p4[:, 1].mean() <= top[1] <= b5[:, 1].mean())
            cid = f'{side}_{k}'
            found.append(dict(id=cid, side=side, area_mm2=float(len(cx)*sy*sx),
                area_pixels=int(len(cx)), upper_contour_length_mm=longest,
                contour_width_mm=float(np.ptp(curve[:, 0])*sx),
                upper_contour=curve.tolist(), top_point=top.tolist(),
                level_L4_L5=level_ok, level_available=level_known,
                bbox_xyxy=[int(cx.min()),int(cy.min()),int(cx.max()+1),int(cy.max()+1)],
                touches_bottom=bool(cy.max()==h-1),
                caudal_support=bool((h-1-cy.max())*sy<=10),
                gap_to_bottom_mm=float((h-1-cy.max())*sy),
                lateral_distance_mm=float((edge-top[0] if side=='left' else top[0]-edge)*sx)))
            masks[cid] = mask
    return found, masks

def iliac_acceptance(candidate, area_threshold_mm2):
    gates = dict(level_L4_L5=bool(candidate['level_L4_L5']),
        upper_contour_20mm=candidate['upper_contour_length_mm'] >= 20 - 1e-6,
        area=candidate['area_mm2'] >= area_threshold_mm2 - 1e-6)
    return any(gates.values()), gates

def apply_iliac_rules(case, bone, area_threshold_mm2, caudal_guard=True):
    case = copy.deepcopy(case)
    candidates, masks = iliac_candidates(case, bone)
    selected = {}; result_mask = np.zeros(bone.shape, 'uint8')
    for c in candidates:
        c['accepted'], c['criteria'] = iliac_acceptance(c, area_threshold_mm2)
        c['passes_or_rule'] = c['accepted']
        c['accepted'] = c['accepted'] and (c['caudal_support'] or not caudal_guard)
    for number, side in enumerate(['left', 'right'], 1):
        eligible = [c for c in candidates if c['side']==side and c['accepted']]
        # Largest component prevents a tiny but high fragment replacing the iliac wing.
        chosen = max(eligible, key=lambda c:c['area_mm2']) if eligible else None
        selected[side] = chosen
        key = 'iliac_'+side
        case['contours'].pop(key, None)
        case['features'][key] = feature('Верхний контур ПК '+side, 1,
            [chosen['top_point']] if chosen else None, source=RULE_VERSION+' connected bone OR rules')
        if chosen:
            case['contours'][key] = chosen['upper_contour']
            result_mask[masks[chosen['id']]] = number
    case['iliac_rules'] = dict(version=RULE_VERSION, area_threshold_mm2=area_threshold_mm2,
        area_threshold_status='provisional automatic measurement pending user review',
        caudal_guard=caudal_guard,
        rule='lateral bone below L3/L4, connected component reaches within 10mm of bottom if guard enabled; then (crest at L4-L5 OR continuous upper contour >=20mm OR area >= threshold)',
        selected=selected, candidates=candidates)
    case['measurements'] = measure(case)
    return case, result_mask

def foreign_density_rule(raw, spacing, anatomical_exclusion=None):
    """Bright high-contrast narrow/compact objects. Intensities are NOT HU.

    Explicit fixed image criteria, exploratory: dense bone or ribs can satisfy them.
    No labels, classifier, heatmap or image identity enter the rule.
    """
    sy,sx=spacing; im=raw.astype('float32')
    background=ndi.median_filter(im,size=(9,15));residual=im-background
    seed=(im>=120)&(residual>=25)
    if anatomical_exclusion is not None:seed &= ~anatomical_exclusion
    seed=ndi.binary_closing(seed,structure=np.ones((2,2)))
    ids,n=ndi.label(seed);mask=np.zeros(raw.shape,'uint8');objects=[]
    for k in range(1,n+1):
        obj=ids==k;ys,xs=np.where(obj)
        if len(xs)<4:continue
        area=len(xs)*sx*sy;coords=np.column_stack([xs*sx,ys*sy])
        ev=np.linalg.eigvalsh(np.cov(coords.T));major=float(4*np.sqrt(max(ev[-1],0)));minor=float(4*np.sqrt(max(ev[0],0)))
        peak=float(im[obj].max());contrast=float(np.median(residual[obj]));ratio=major/max(minor,sx)
        elongated=major>=15 and ratio>=3 and minor<=8 and area>=3
        compact=peak>=250 and contrast>=100 and 2<=area<=50
        accepted=bool(elongated or compact)
        if accepted:mask[obj]=255
        objects.append(dict(area_mm2=float(area),major_mm=major,minor_mm=minor,elongation=ratio,
            peak=peak,median_local_contrast=contrast,elongated=bool(elongated),compact=bool(compact),
            accepted=accepted,bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]))
    return dict(present=bool(mask.any()),objects=objects,
        rule='Outside anatomical exclusion: I>=120 and local excess>=25; connected object: (PCA length>=15mm, elongation>=3, width<=8mm, area>=3mm2) OR (peak>=250, median excess>=100, area 2..50mm2)',
        caveat='Candidate foreign bodies, not validated density or material identification; cannot exclude low-contrast objects',
        learned_classifier_used=False),mask

def foreign_anatomical_exclusion(case):
    """Exclude detected vertebral bodies + 3mm and accepted iliac envelopes.

    This reduces cortical false positives but loses objects superimposed on these
    structures; that limitation is explicitly reported, not treated as absence.
    """
    sy,sx=case['spacing_yx_mm'];h,w=case['shape'];mask=np.zeros((h,w),'uint8')
    for det in case.get('detected_vertebrae',[]):
        q=np.rint(np.asarray(det['corners'])[[0,1,3,2]]).astype('int32');cv2.fillPoly(mask,[q],1)
    mask=cv2.dilate(mask,np.ones((2*round(3/sy)+1,2*round(3/sx)+1),'uint8'))
    for c in case.get('iliac_rules',{}).get('selected',{}).values():
        if c:
            q=np.asarray(c['upper_contour']);q=np.concatenate([q,[[q[-1,0],h-1],[q[0,0],h-1]]])
            cv2.fillPoly(mask,[np.rint(q).astype('int32')],1)
    return mask>0

def evaluate_explicit(case, raw):
    """Four evaluated tasks; lesser-trochanter/hip-rotation task is deferred."""
    m=measure(case)['measurements'];decisions={};details={}
    foreign,foreign_mask=foreign_density_rule(raw,case['spacing_yx_mm'],foreign_anatomical_exclusion(case)) if case['region']=='spine' else ({},np.zeros(raw.shape,'uint8'))
    if case['region']=='spine':
        checks={'L1_top_margin':m['L1_top_margin']['passes']}
        for side in ['left','right']:checks['iliac_'+side]=case['iliac_rules']['selected'][side] is not None
        # Unknown is kept distinct from a measured failure; operational disposition is review/reject.
        decisions['spine_position']=True if any(v is False for v in checks.values()) else None if any(v is None for v in checks.values()) else False
        angle=m.get('axis_L5_topmost',{}).get('value')
        decisions['spine_axis']=None if angle is None else angle>5
        decisions['spine_artifact']=foreign['present']
        details=dict(position_checks=checks,iliac=case['iliac_rules'],axis=m.get('axis_L5_topmost'),foreign=foreign)
    else:
        keys=['greater_trochanter_top','ischium_bottom','femur_lateral_margin']
        checks={k:m[k]['passes'] for k in keys}
        decisions['hip_roi']=True if any(v is False for v in checks.values()) else None if any(v is None for v in checks.values()) else False
        decisions['hip_position']=None
        details=dict(margin_checks=checks,hip_position_status='deferred; lesser trochanter unchanged and not reevaluated')
    known=list(decisions.values());overall=True if any(v is True for v in known) else None if any(v is None for v in known) else False
    return dict(version=RULE_VERSION,task_decisions=decisions,details=details,
        partial_quality_class=overall,full_five_task_class_available=case['region']=='spine' and all(v is not None for v in known),
        review_required=any(v is None for v in known),learned_quality_classifier_used=False,
        operational_reject_or_review=overall is not False),foreign_mask
