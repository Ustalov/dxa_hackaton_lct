"""Local durable landmark review. Independent of the production classifier app."""
import json,threading,hashlib,math,os
from pathlib import Path
from datetime import datetime,timezone
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse
from .anatomy_measurements import measure

ROOT=Path(os.environ.get('DXA_LANDMARK_ROOT','/data/dxa-landmark-review-v1'));WEB=Path(__file__).resolve().parents[1]/'web';lock=threading.Lock()
app=FastAPI(title='DXA landmark review')

def directory(cid):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if cid not in {v['id'] for v in manifest['cases']}:raise HTTPException(404,'Unknown case')
    return ROOT/'cases'/cid

def read(cid):
    p=directory(cid);saved=p/'review.json'
    return json.loads((saved if saved.exists() else p/'proposal.json').read_text())

@app.get('/')
def home():return FileResponse(WEB/'landmark_review.html')

@app.get('/health')
def health():return {'status':'ready','data':str(ROOT),'dicom_used':False}

@app.get('/api/cases')
def cases():
    m=json.loads((ROOT/'manifest.json').read_text())
    for c in m['cases']:
        r=read(c['id']);c.update(review_status=r['review_status'],revision=r['revision'])
    return m

@app.get('/api/cases/{cid}')
def get_case(cid):
    case=read(cid);return {'case':case,'measurements':measure(case)}

@app.get('/images/{cid}/{filename}')
def picture(cid,filename):
    if filename not in ('original.png','mask_overlay.png'):raise HTTPException(404)
    return FileResponse(directory(cid)/filename)

@app.put('/api/cases/{cid}')
async def update(cid,request:Request):
    payload=await request.json()
    if not isinstance(payload,dict):raise HTTPException(422,'Expected JSON object')
    with lock:
        old=read(cid)
        if payload.get('revision')!=old['revision']:raise HTTPException(409,'This case changed in another tab. Reload before saving.')
        features=payload.get('features',{});h,w=old['shape']
        if not isinstance(features,dict):raise HTTPException(422,'Invalid landmark set')
        if set(features)!=set(old['features']):raise HTTPException(422,'Invalid landmark set')
        new=json.loads(json.dumps(old))
        for key,f in features.items():
            if not isinstance(f,dict):raise HTTPException(422,'Invalid landmark')
            if f.get('visibility') not in ('visible','proposed','not_visible','uncertain'):raise HTTPException(422,'Invalid visibility')
            pts=f.get('points',[]);arity=old['features'][key]['arity']
            if not isinstance(pts,list) or len(pts)>arity:raise HTTPException(422,'Invalid point count')
            for p in pts:
                if not isinstance(p,list) or len(p)!=2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in p) or not (0<=p[0]<=w-1 and 0<=p[1]<=h-1):raise HTTPException(422,'Coordinates must be finite and inside the original image')
            new['features'][key].update(points=pts,visibility=f['visibility'])
            if pts!=old['features'][key]['points']:new['features'][key]['provenance']='manually adjusted in expert review'
        review=payload.get('review_status','pending')
        if review not in ('pending','approved','needs_fix'):raise HTTPException(422,'Invalid review status')
        if review=='approved':
            for f in new['features'].values():
                if f['visibility'] in ('proposed','visible'):
                    if len(f['points'])!=f['arity']:raise HTTPException(422,'Complete visible landmarks or mark uncertain / not visible')
                    f['visibility']='visible'
        side=payload.get('lateral_image_side',old.get('lateral_image_side'))
        if side not in ('left','right',None):raise HTTPException(422,'Invalid lateral side')
        new['lateral_image_side']=side
        for key in ['cobb_top','cobb_bottom']:
            v=payload.get(key,old[key])
            if v not in ['L1','L2','L3','L4','L5']:raise HTTPException(422,'Invalid vertebral level')
            new[key]=v
        new['curve_endpoints_confirmed']=payload.get('curve_endpoints_confirmed') is True
        new['notes']=str(payload.get('notes',''))[:4000];new['review_status']=review;new['revision']=old['revision']+1;new['updated_at']=datetime.now(timezone.utc).isoformat()
        new['measurements']=measure(new)
        p=directory(cid);history=p/'revisions';history.mkdir(exist_ok=True)
        text=json.dumps(new,ensure_ascii=False,indent=2)
        (history/f'{new["revision"]:06d}.json').write_text(text)
        tmp=p/'review.tmp';tmp.write_text(text);tmp.replace(p/'review.json')
    return {'case':new,'measurements':new['measurements']}

@app.get('/api/export')
def export():
    with lock:
        manifest=json.loads((ROOT/'manifest.json').read_text());rows=[read(c['id']) for c in manifest['cases']]
        for c in rows:c['measurements']=measure(c)
        return {'pilot':'landmark-review-v1','exported_at':datetime.now(timezone.utc).isoformat(),'cases':rows,
                'dicom_used':False,'spacing_source':'user_assumed_dicom_equivalent'}
