"""Geometric DXA measurements from explicit, reviewable landmarks.

No landmark detector is implied. Unknown or invisible structures produce no
measurement. Coordinates refer to centers of original image pixels.
"""
import math
import numpy as np

SPACING_YX=(1.05,.6)

def points(case,key,count):
    f=case.get('features',{}).get(key,{})
    if f.get('visibility') not in ('visible','proposed'):return None
    p=np.asarray(f.get('points',[]),dtype=float)
    h,w=case['shape']
    if p.shape!=(count,2) or not np.isfinite(p).all():return None
    if np.any(p[:,0]<0) or np.any(p[:,0]>w-1) or np.any(p[:,1]<0) or np.any(p[:,1]>h-1):return None
    return p

def physical(p,spacing):return np.asarray(p)*np.array([spacing[1],spacing[0]])

def result(value=None,passes=None,unit='mm',reason=None,**extra):
    return {'value':None if value is None else float(value),'passes':None if passes is None else bool(passes),'unit':unit,'reason':reason,**extra}

def lesser_trochanter(base_start,apex,base_end,spacing_yx=SPACING_YX,medial_direction=1):
    """Length along a local shaft baseline; signed prominence towards pelvis."""
    a,p,b=physical(np.array([base_start,apex,base_end],float),spacing_yx)
    vector=b-a;length=float(np.linalg.norm(vector))
    if length<1e-6:return {'valid':False,'reason':'Coincident baseline endpoints'}
    tangent=vector/length;normal=np.array([-tangent[1],tangent[0]])
    if normal[0]*medial_direction<0:normal=-normal
    projection=float(np.dot(p-a,tangent));prominence=float(np.dot(p-a,normal))
    apex_inside=-1e-6<=projection<=length+1e-6
    shape_pass=apex_inside and length>=2-1e-6 and prominence>=1-1e-6 and length>prominence+1e-6
    return {'valid':apex_inside,'length_mm':length,'medial_protrusion_mm':prominence,'length_greater_than_protrusion':length>prominence+1e-6,
            'passes_visibility_shape_rule':bool(shape_pass),'apex_projection_mm':projection,
            'excessive_prominence':None,'excessive_prominence_reason':'Upper limit has not been specified',
            'anatomical_location_must_be_verified':True}

def line_angle(line,spacing):
    p=physical(line,spacing)
    if p[1,0]<p[0,0]:p=p[::-1]
    d=p[1]-p[0]
    if np.linalg.norm(d)<1e-6:return None
    return math.degrees(math.atan2(d[1],d[0]))

def axis_l5_topmost(case, spacing):
    """Use the highest detected body, without silently replacing clipped endpoints."""
    detections=case.get('detected_vertebrae',[])
    if not detections:
        return result(unit='degrees',reason='Visible vertebra chain is unavailable')
    top=min(detections,key=lambda d:d['center'][1])
    name=top.get('anatomical_level')
    a=points(case,str(name)+'_upper',2);b=points(case,'L5_lower',2)
    info=dict(top_level=name,bottom_level='L5',definition='L5 lower midpoint to topmost vertebra upper midpoint')
    method='visible_endplates';estimated=False
    if a is None or b is None:
        if not case.get('axis_allow_estimate'):
            return result(unit='degrees',reason='Topmost upper or L5 lower boundary is missing or clipped',**info)
        bottom=next((d for d in detections if d.get('anatomical_level')=='L5'),None)
        if bottom is None:return result(unit='degrees',reason='L5 detection is missing',**info)
        # Predicted midpoints retain the original definition even for partly clipped plates.
        a=np.asarray(top.get('corners',[]),float);b=np.asarray(bottom.get('corners',[]),float)
        if a.shape==(4,2) and b.shape==(4,2) and np.isfinite(a).all() and np.isfinite(b).all():
            a=a[:2];b=b[2:];method='predicted_endplates_partly_outside_frame'
        else:
            a=np.asarray([top['center']],float);b=np.asarray([bottom['center']],float);method='extreme_vertebra_centers'
        estimated=True
    a=a.mean(0);b=b.mean(0);dx=(a[0]-b[0])*spacing[1];dy=(b[1]-a[1])*spacing[0]
    if dy<=0 and case.get('axis_allow_estimate') and name!='L5':
        bottom=next((d for d in detections if d.get('anatomical_level')=='L5'),None)
        if bottom is not None:
            a=np.asarray(top['center'],float);b=np.asarray(bottom['center'],float)
            dx=(a[0]-b[0])*spacing[1];dy=(b[1]-a[1])*spacing[0];estimated=True;method='extreme_vertebra_centers'
    if name=='L5' or dy<=0:
        return result(unit='degrees',reason='Two distinct ordered vertebrae are required',**info)
    physical_angle=math.degrees(math.atan2(abs(dx),dy))
    pixel_angle=math.degrees(math.atan2(abs(a[0]-b[0]),b[1]-a[1]))
    coordinate_space=case.get('axis_angle_space','physical')
    if coordinate_space not in ('physical','square_pixels'):raise ValueError('Unknown axis angle coordinate space')
    angle=pixel_angle if coordinate_space=='square_pixels' else physical_angle
    return result(angle,passes=angle<=5,unit='degrees',criterion='<= 5°',
                  coordinate_space=coordinate_space,physical_value_degrees=physical_angle,square_pixel_value_degrees=pixel_angle,
                  endpoints_px=[b.tolist(),a.tolist()],
                  direction='right' if dx>0 else 'left' if dx<0 else 'vertical',
                  estimated=estimated,method=method,**info)

def measure(case):
    sy,sx=case.get('spacing_yx_mm',SPACING_YX);spacing=(float(sy),float(sx))
    if not np.isfinite(spacing).all() or min(spacing)<=0:raise ValueError('Invalid pixel spacing')
    h,w=case['shape'];out={'provisional':case.get('review_status')!='approved','spacing_yx_mm':list(spacing),
                          'spacing_source':case.get('spacing_source','user_assumed_dicom_equivalent'),'measurements':{}}
    m=out['measurements']
    if case['region']=='hip':
        specs=[('greater_trochanter','greater_trochanter_top',30),('ischium','ischium_bottom',30),('lateral_femur','femur_lateral_margin',20)]
        for key,name,limit in specs:
            p=points(case,key,1)
            if p is None:m[name]=result(reason='Landmark is missing, invisible or uncertain');continue
            x,y=p[0]
            if key=='greater_trochanter':value=y*sy;target_edge='top';target_point=[float(x),0.]
            elif key=='ischium':value=(h-1-y)*sy;target_edge='bottom';target_point=[float(x),float(h-1)]
            else:
                side=case.get('lateral_image_side')
                if side not in ('left','right'):m[name]=result(reason='Lateral image side unknown');continue
                value=(x if side=='left' else w-1-x)*sx;target_edge=side;target_point=[0. if side=='left' else float(w-1),float(y)]
            m[name]=result(value,value>=limit-1e-6,criterion=f'>= {limit} mm',landmark=key,target_image_edge=target_edge,target_point_px=target_point)
        p=points(case,'lesser_trochanter',3)
        if p is None:m['lesser_trochanter']={'valid':False,'reason':'Mark baseline start, apex and baseline end, or state not visible'}
        else:
            direction=1 if case.get('lateral_image_side')=='left' else -1
            m['lesser_trochanter']=lesser_trochanter(*p,spacing_yx=spacing,medial_direction=direction)
        return out
    upper=points(case,'L1_upper',2);lower=points(case,'L1_lower',2)
    if upper is None or lower is None:
        m['L1_top_margin']=result(reason='Both L1 endplates are required');m['L1_height']=result(reason='Both L1 endplates are required')
    elif lower[:,1].mean()<=upper[:,1].mean():
        m['L1_top_margin']=result(reason='L1 lower endplate must be below upper endplate');m['L1_height']=result(reason='Invalid endplate order')
    else:
        height=float(np.linalg.norm(physical(lower.mean(0)-upper.mean(0),spacing)));margin=float(upper[:,1].min()*sy)
        m['L1_height']=result(height);m['L1_top_margin']=result(margin,margin+1e-6>=height/2,criterion='>= 0.5 × L1 height',required_mm=height/2,ratio=margin/height)
    l5=points(case,'L5_lower',2)
    centers=[];center_levels=[]
    for k in range(1,5):
        a=points(case,f'L{k}_upper',2);b=points(case,f'L{k}_lower',2)
        if a is not None and b is not None and b[:,1].mean()>a[:,1].mean():
            centers.append((a.mean(0)+b.mean(0))/2);center_levels.append(k)
    if len(centers)>=3 and 1 in center_levels and 4 in center_levels and np.all(np.diff(np.array(centers)[:,1])>0):
        c=physical(np.array(centers),spacing)
        if np.ptp(c[:,1])>0:
            slope,intercept=np.polyfit(c[:,1],c[:,0],1)
            m['visible_axis_L1_L4']=result(math.degrees(math.atan(abs(slope))),unit='degrees',
                reason='Descriptive visible-segment tilt only; does not replace L1–L5 criterion',
                fit_x_mm_per_y_mm=float(slope),fit_intercept_mm=float(intercept),
                endpoints_px=[[float((slope*y+intercept)/sx),float(y/sy)] for y in [c[:,1].min(),c[:,1].max()]])
    if upper is None or l5 is None:m['axis_L1_L5']=result(unit='degrees',reason='L1 upper and L5 lower endplates must be visible')
    else:
        a=upper.mean(0);b=l5.mean(0);dx=(a[0]-b[0])*sx;dy=(b[1]-a[1])*sy
        if dy<=0:m['axis_L1_L5']=result(unit='degrees',reason='L5 must be below L1')
        else:
            angle=math.degrees(math.atan2(abs(dx),dy));m['axis_L1_L5']=result(angle,angle<=5+1e-6,unit='degrees',criterion='<= 5°',direction='right' if dx>0 else 'left' if dx<0 else 'vertical')
    if case.get('axis_definition')=='L5_to_topmost':
        m['axis_L5_topmost']=axis_l5_topmost(case,spacing)
    l3bottom=points(case,'L3_lower',2);l4top=points(case,'L4_upper',2);l4bottom=points(case,'L4_lower',2)
    for side in ['left','right']:
        crest=points(case,'iliac_'+side,1);key='iliac_'+side
        if crest is None or l3bottom is None or l4top is None:
            m[key]=result(reason='Crest and L3–L4 reference levels must be identified');continue
        reference=np.concatenate([l4top,l4bottom]) if l4bottom is not None else l4top
        edge=reference[:,0].min() if side=='left' else reference[:,0].max()
        distance=(edge-crest[0,0] if side=='left' else crest[0,0]-edge)*sx
        level=(l3bottom[:,1].mean()+l4top[:,1].mean())/2
        below=bool(crest[0,1]>level);passes=distance>=20-1e-6 and below
        m[key]=result(distance,passes,criterion='>=20 mm lateral and below L3–L4',below_L3_L4=below,crest_visibility_requires_review=True)
    top=case.get('cobb_top','L1');bottom=case.get('cobb_bottom','L4')
    a=points(case,top+'_upper',2);b=points(case,bottom+'_lower',2)
    if a is None or b is None or top not in ['L1','L2','L3','L4','L5'] or bottom not in ['L1','L2','L3','L4','L5'] or int(top[1:])>=int(bottom[1:]):
        m['endplate_angle']=result(unit='degrees',reason='Select visible upper and lower end vertebrae in order')
    else:
        aa=line_angle(a,spacing);bb=line_angle(b,spacing)
        if aa is None or bb is None:m['endplate_angle']=result(unit='degrees',reason='Endplate endpoints coincide')
        else:
            angle=abs(aa-bb)
            complete=bool(case.get('curve_endpoints_confirmed',False))
            m['endplate_angle']=result(angle,unit='degrees',levels=[top,bottom],kind='Cobb angle of selected visible curve' if complete else 'angle between selected endplates; not confirmed Cobb',
                                      curve_endpoints_confirmed=complete,whole_spine_diagnosis=False)
    return out
