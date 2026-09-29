"""All-visible vertebra proposals and explicit user-provided bottom-up numbering.

Published AASCE landmark weights are frozen. Consensus and duplicate suppression
are heuristics, not supervised anatomical identification or a calibrated score.
"""
import copy
import cv2
import numpy as np
import torch
from .isbi2020_adapter import decode,letterbox
from .png_spine_v6 import contrast_views


def number_from_bottom(detections):
    """For user's native DICOM series ONLY: lowest visible vertebra is L5."""
    names = ['L5', 'L4', 'L3', 'L2', 'L1', 'Th12', 'Th11']
    if len(detections)>7:raise ValueError('Native anatomical chain must not extend above Th11')
    rows = sorted(copy.deepcopy(detections), key=lambda r: r['center'][1], reverse=True)
    for i, row in enumerate(rows):
        row['anatomical_level'] = names[i] if i < len(names) else None
        row['numbering_source'] = 'user assumption: lowest visible native DICOM vertebra is L5; consecutive order upward'
        row['numbering_verified'] = False
        row['label_id'] = i + 1 if i >= 5 else 5 - i
    return rows[::-1]


def constrain_native_chain(detections):
    """At most seven bodies, coherent laterally and ordered upwards from lowest L5."""
    rows=sorted(copy.deepcopy(detections),key=lambda d:d['center'][1])
    flags=[]
    if len(rows)<3:return rows,flags
    widths=np.array([np.ptp(np.array(d['corners'])[:,0]) for d in rows]);median_x=np.median([d['center'][0] for d in rows])
    coherent=[d for d in rows if abs(d['center'][0]-median_x)<=.8*np.median(widths)]
    if len(coherent)>=2 and len(coherent)<len(rows):rows=coherent;flags.append('lateral_outlier_candidates_rejected')
    if len(rows)<=7:return rows,flags
    # Score whole chains, rather than labelling every peak as a new vertebra.
    states={}
    for j,d in enumerate(rows):
        states[(j,1)]=(float(d['score'])+.6,[j])
        for k in range(2,8):
            best=None
            for i in range(j):
                if (i,k-1) not in states:continue
                a=rows[i];gap=d['center'][1]-a['center'][1]
                ah=np.ptp(np.array(a['corners'])[:,1]);bh=np.ptp(np.array(d['corners'])[:,1])
                if gap<.65*min(ah,bh):continue
                width=(np.ptp(np.array(a['corners'])[:,0])+np.ptp(np.array(d['corners'])[:,0]))/2
                penalty=.5*abs(np.log(max(gap,1)/max((ah+bh)/2,1)))+.5*abs(d['center'][0]-a['center'][0])/max(width,1)
                score,path=states[(i,k-1)];candidate=(score+float(d['score'])+.6-penalty,path+[j])
                if best is None or candidate[0]>best[0]:best=candidate
            if best:states[(j,k)]=best
    # User specifies that the lowest visible body is L5; preserve that anchor.
    _,chosen=max((v for (j,k),v in states.items() if j==len(rows)-1 and k>=2),key=lambda v:v[0])
    flags.append('candidate_chain_limited_to_L5_Th11')
    return [rows[i] for i in chosen],flags


def consolidate(candidates, shape):
    h, w = shape; filtered = []
    for c in candidates:
        q = np.array(c['corners']); x, y = c['center']
        width, height = np.ptp(q[:, 0]), np.ptp(q[:, 1])
        if not (.18*w < x < .82*w and .08*w < width < .65*w and 4 < height < .28*h):
            continue
        if not ((q[[1, 3], 0] > q[[0, 2], 0]).all() and (q[[2, 3], 1] > q[[0, 1], 1]).all()):
            continue
        filtered.append(c)
    clusters = []
    for c in sorted(filtered, key=lambda r: -r['score']):
        q = np.array(c['corners']); bh = np.ptp(q[:, 1])
        group = None
        for g in clusters:
            anchor = g[0]; height = min(bh, np.ptp(np.array(anchor['corners'])[:, 1]))
            if abs(c['center'][1] - anchor['center'][1]) <= max(5, .42*height) and abs(c['center'][0] - anchor['center'][0]) < .09*w:
                group = g; break
        if group is None: clusters.append([c])
        else: group.append(c)
    proposals = []
    for g in clusters:
        # One vote per scale/view; nearby peaks from one pass cannot inflate support.
        votes = {}
        for c in g:
            key = (c['scale'], c['view'])
            if key not in votes or c['score'] > votes[key]['score']: votes[key] = c
        g = list(votes.values()); scales = sorted({c['scale'] for c in g})
        if len(scales) < 2: continue
        weights = np.array([c['score'] for c in g]); score = float(weights.max())
        center = np.average([c['center'] for c in g], axis=0, weights=weights)
        corners = np.average([c['corners'] for c in g], axis=0, weights=weights)
        # The same floor at both ends: a low-confidence sacral peak must not shift every label.
        if score < .20: continue
        proposals.append(dict(center=center.tolist(), corners=corners.tolist(), score=score,
                              scale_support=scales, vote_count=len(g), weak_candidate=score < .2,
                              corners_outside_image=bool((corners < 0).any() or (corners[:, 0] >= w).any() or (corners[:, 1] >= h).any())))
    # Suppress second detections of the same body; do not request a fixed count.
    selected = []
    for p in sorted(proposals, key=lambda r: -(r['score'] + .025*len(r['scale_support']))):
        ph = np.ptp(np.array(p['corners'])[:, 1])
        if any(abs(p['center'][1]-q['center'][1]) < .82*min(ph, np.ptp(np.array(q['corners'])[:, 1])) for q in selected):
            continue
        selected.append(p)
    selected.sort(key=lambda r: r['center'][1])
    flags = []
    if len(selected) < 6: flags.append('fewer_than_six_candidates_Th12_and_all_lumbar_levels_not_established')
    if len(selected) > 17: flags.append('too_many_candidates_for_T1_to_L5')
    if selected:
        bottom = selected[-1]
        if bottom['weak_candidate']: flags.append('weak_lowest_candidate_check_L5_vs_sacrum')
        if bottom['center'][1] < .72*h: flags.append('lowest_detection_far_from_bottom_possible_missed_L5')
        if any(p['corners_outside_image'] for p in selected): flags.append('partially_visible_candidate')
    if len(selected) > 2:
        gaps = np.diff([p['center'][1] for p in selected])
        if gaps.max() > 1.8*np.median(gaps): flags.append('large_intervertebral_gap_possible_missed_level_numbering_may_shift')
    return selected, flags


@torch.no_grad()
def propose_all(model, image):
    h, w = image.shape; views = contrast_views(image); candidates = []
    # Match training letterboxing. Tall direct stretching caused multiple peaks per body.
    for ih, iw in ((384, 288), (512, 384), (640, 480)):
        for name in ('positive', 'clahe'):
            im,transform = letterbox(views[name],(ih,iw))
            x = torch.from_numpy(np.repeat(im[None].astype('float32') / 255 - .5, 3, 0))[None].to(next(model.parameters()).device)
            output = model(x)
            rows = decode(output, [transform],max_detections=24, confidence=.04)[0]
            for row in rows: row.update(scale=ih, view=name)
            candidates.extend(rows)
    return decode_native_candidates(candidates,image.shape)


def decode_native_candidates(candidates,shape):
    selected, flags = consolidate(candidates, shape)
    selected,chain_flags=constrain_native_chain(selected);flags+=chain_flags
    numbered = number_from_bottom(selected)
    flags.append('bottom_up_numbering_assumes_no_missed_vertebrae_and_lowest_is_L5')
    return numbered, flags, candidates


def instance_masks(detections, bone):
    """Intersect landmark quadrilaterals with existing neural bone foreground.

    These are derived body masks, not a new learned segmentation of each vertebra.
    """
    result = np.zeros(bone.shape, 'uint8')
    for d in sorted(detections, key=lambda r: r['score']):
        if d['anatomical_level'] is None: continue
        q = np.rint(np.array(d['corners'])[[0, 1, 3, 2]]).astype('int32')
        mask = np.zeros_like(result); cv2.fillConvexPoly(mask, q, 1)
        result[(mask > 0) & (bone > 0)] = d['label_id']
    return result
