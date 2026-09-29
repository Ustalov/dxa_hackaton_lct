"""Fixed, label-free geometric descriptors for the five-task pipeline."""
import numpy as np

TASKS = ['spine_position', 'spine_axis', 'spine_artifact', 'hip_position', 'hip_roi']
MEASUREMENTS = ['L1_height', 'L1_top_margin', 'axis_L5_topmost', 'axis_L1_L5',
                'iliac_left', 'iliac_right', 'greater_trochanter_top',
                'ischium_bottom', 'femur_lateral_margin', 'endplate_angle']

def geometry_vector(case):
    m = case['measurements']['measurements']; names=[]; values=[]
    def add(name, value):
        names.append(name); values.append(float(value) if value is not None else np.nan)
    for key in MEASUREMENTS:
        item=m.get(key,{})
        # Other trained tasks retain their original physical-angle feature convention.
        value=item.get('physical_value_degrees',item.get('value')) if key=='axis_L5_topmost' else item.get('value')
        add(key,value); add(key+'_missing', value is None)
    for key in ['ratio', 'required_mm']: add('L1_margin_'+key,m.get('L1_top_margin',{}).get(key))
    lt=m.get('lesser_trochanter',{})
    for key in ['length_mm','medial_protrusion_mm','length_greater_than_protrusion','valid']:
        add('lesser_'+key,lt.get(key))
    add('axis_estimated',m.get('axis_L5_topmost',{}).get('estimated'))
    detections=case.get('detected_vertebrae',[])
    add('vertebra_count',len(detections))
    add('mean_corner_score',np.mean([d['score'] for d in detections]) if detections else None)
    h,w=case['shape']; sy,sx=case['spacing_yx_mm']
    add('height_mm',h*sy);add('width_mm',w*sx)
    for key in ['iliac_left','iliac_right','greater_trochanter','ischium','lateral_femur']:
        pts=case.get('features',{}).get(key,{}).get('points',[])
        add(key+'_x_fraction',np.mean(np.asarray(pts)[:,0])/w if pts else None)
        add(key+'_y_fraction',np.mean(np.asarray(pts)[:,1])/h if pts else None)
    # Fraction of predicted Th12 body inside the upper boundary of the acquisition.
    d=next((d for d in detections if d['anatomical_level']=='Th12'),None)
    fraction=None
    if d:
        q=np.asarray(d['corners']); top=q[:2,1].mean();bottom=q[2:,1].mean()
        fraction=float(np.clip((min(h-1,bottom)-max(0,top))/max(bottom-top,1),0,1))
    add('Th12_visible_fraction',fraction)
    return np.asarray(values,dtype=np.float32),names

def task_matrix(features,geometry,mode):
    if mode=='measurements':return geometry
    if mode=='axis':return geometry[:,[4,6]]
    if mode=='geometry': return np.concatenate([features['geo'],geometry],axis=1)
    if mode=='image_geometry':return np.concatenate([features['geo'],geometry,features['deep']],axis=1)
    raise ValueError(mode)

def criterion_flags(case):
    """Transparent physical checks, kept distinct from learned Excel-label predictions."""
    m=case['measurements']['measurements']
    if case['region']=='hip':
        return {key:m.get(key,{}).get('passes') for key in
                ['greater_trochanter_top','ischium_bottom','femur_lateral_margin']}
    axis=m.get('axis_L5_topmost',{}).get('value')
    vector,names=geometry_vector(case);f=dict(zip(names,vector))
    fraction=f['Th12_visible_fraction']
    return dict(axis_within_5_degrees=None if axis is None else bool(axis<=5),
                L1_top_margin=m.get('L1_top_margin',{}).get('passes'),
                Th12_at_least_half_visible=None if not np.isfinite(fraction) else bool(fraction>=.5),
                iliac_left=m.get('iliac_left',{}).get('passes'),
                iliac_right=m.get('iliac_right',{}).get('passes'))
