"""Explicit spine field-coverage criteria, independent of the learned quality head."""
from .anatomy_measurements import measure,points

def spine_position_criteria(case):
    if case['region']!='spine':raise ValueError('Spine case required')
    measurements=measure(case)['measurements']
    margin=measurements['L1_top_margin'];checks={'L1_top_margin':margin.get('passes')}
    missing=[]
    for side in ['left','right']:
        key='iliac_'+side
        observed=points(case,key,1) is not None and measurements[key].get('passes') is True
        checks[key]=bool(observed)
        if not observed:missing.append(side)
    # A normal acquisition must satisfy ALL criteria. A failed detector is not evidence of normality.
    accepted=all(value is True for value in checks.values())
    return dict(position_violation=not accepted,normal_criteria_confirmed=accepted,criteria=checks,
                top_margin_mm=margin.get('value'),required_top_margin_mm=margin.get('required_mm'),
                margin_to_L1_height_ratio=margin.get('ratio'),not_detected_iliac_sides=missing,
                unavailable_L1_measurement=checks['L1_top_margin'] is None,
                rule='L1 top margin >= half L1 height AND left iliac crest detected AND right iliac crest detected',
                interpretation='Rejected if a criterion fails or cannot be confirmed; not-detected bone does not prove anatomical absence',
                learned_quality_classifier_used=False)
