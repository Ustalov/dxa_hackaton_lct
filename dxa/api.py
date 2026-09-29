from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import os,json,uuid,threading,time,shutil
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse
from .service import Analyzer,safe_extract

ROOT=Path(__file__).resolve().parents[1]
JOBS=Path(os.environ.get('DXA_JOBS',str(ROOT/'jobs')))
ART=Path(os.environ.get('DXA_ARTIFACTS',str(ROOT/'artifacts')))
pool=None;lock=threading.Lock();analyzer=None;prototype=None;active_jobs=set()
MAX_JOBS=4
RETENTION_SECONDS=float(os.environ.get('DXA_RETENTION_HOURS','24'))*3600
MAX_JOB_DISK=int(os.environ.get('DXA_JOBS_QUOTA_BYTES',str(8*1024**3)))

def status(path,data):
    temp=path/'status.tmp';temp.write_text(json.dumps(data,ensure_ascii=False));temp.replace(path/'status.json')

@asynccontextmanager
async def lifespan(app):
    global analyzer,pool,prototype
    JOBS.mkdir(parents=True,exist_ok=True)
    active_jobs.clear();pool=ThreadPoolExecutor(max_workers=1)
    for p in JOBS.glob('*/status.json'):
        try:old=json.loads(p.read_text())
        except (ValueError,OSError):
            status(p.parent,{'state':'failed','error':'Invalid saved job status; upload the batch again'});continue
        if old.get('state') in ('uploading','queued','running'):status(p.parent,{'state':'failed','error':'Service restarted; upload the batch again'})
    for p in JOBS.iterdir():
        if p.is_dir() and not (p/'status.json').exists():status(p,{'state':'failed','error':'Interrupted upload; upload the batch again'})
    analyzer=Analyzer(ART)
    from .prototype import PrototypeAnalyzer
    prototype=PrototypeAnalyzer()
    yield
    pool.shutdown(wait=True)

app=FastAPI(title='DXA Quality',version='0.2.0',lifespan=lifespan)
from .prototype_auth import PrototypeAuth
app.add_middleware(PrototypeAuth, config_path=os.environ.get('DXA_AUTH_FILE', str(ROOT/'config'/'prototype-auth.json')))

def job_path(job):
    try:
        if str(uuid.UUID(job))!=job:raise ValueError()
    except ValueError:raise HTTPException(404,'Job not found')
    path=JOBS/job
    if not path.is_dir():raise HTTPException(404,'Job not found')
    return path

def run_job(path,is_zip,variant,spacing_yx=(1.05,.6)):
    try:
        status(path,{'state':'running','phase':'extracting'})
        files=safe_extract(path/'upload',path/'input') if is_zip else list((path/'input').iterdir())
        def progress(done,total,phase):status(path,{'state':'running','phase':phase,'done':done,'total':total})
        result=prototype.analyze(files,path/'output',base=path/'input',progress=progress,spacing_yx=spacing_yx)
        status(path,{'state':'complete',**result['summary']})
    except Exception as e:status(path,{'state':'failed','error':f'{type(e).__name__}: {e}'})
    finally:
        with lock:active_jobs.discard(path.name)

def clean_old_jobs():
    now=time.time()
    for p in JOBS.glob('*/status.json'):
        if p.parent.name in active_jobs:continue
        try:
            saved=json.loads(p.read_text())
            if saved.get('state') in ('complete','failed') and now-p.stat().st_mtime>RETENTION_SECONDS:
                shutil.rmtree(p.parent)
        except (OSError,ValueError):continue

def reserve_job():
    with lock:
        clean_old_jobs()
        if len(active_jobs)>=MAX_JOBS:raise HTTPException(429,'Queue is full, including uploads in progress')
        used=0
        for p in JOBS.rglob('*'):
            try:
                if p.is_file():used+=p.stat().st_size
            except FileNotFoundError:continue  # A worker may finish its temporary pixel cache.
        if used>=MAX_JOB_DISK or shutil.disk_usage(JOBS).free<1024**3:raise HTTPException(507,'Insufficient job storage; remove old results or retry after retention cleanup')
        job=str(uuid.uuid4());path=JOBS/job;path.mkdir();(path/'input').mkdir()
        status(path,{'state':'uploading'});active_jobs.add(job)
        return job,path

@app.get('/health')
def health():return {'status':'ready' if prototype else 'starting','model':prototype.bundle['name'] if prototype else None}

@app.post('/api/jobs',status_code=202)
async def submit(request:Request,kind:str='zip',variant:str='',filename:str='',spacing_x:float=.6,spacing_y:float=1.05):
    import math
    if not all(math.isfinite(v) and .01<=v<=10 for v in (spacing_x,spacing_y)):
        raise HTTPException(400,'Pixel Spacing должен быть от 0,01 до 10 мм/пиксель')
    if kind not in ('zip','dicom','png','jpg'):raise HTTPException(400,'kind must be zip, dicom, png or jpg')
    if variant and variant!=prototype.bundle['name']:raise HTTPException(400,'Prototype uses the selected models')
    if filename and (Path(filename).name!=filename or '\\' in filename or len(filename)>200 or filename in ('.','..') or any(ord(c)<32 for c in filename)):
        raise HTTPException(400,'Invalid filename')
    job,path=reserve_job();queued=False
    target=path/'upload' if kind=='zip' else path/'input'/(filename or ('image.'+('dcm' if kind=='dicom' else kind)))
    size=0;limit=128*1024*1024 if kind=='zip' else 32*1024*1024
    try:
        with target.open('wb') as stream:
            async for chunk in request.stream():
                size+=len(chunk)
                if size>limit:raise HTTPException(413,'Upload exceeds limit')
                stream.write(chunk)
        if not size:raise HTTPException(400,'Empty upload')
        status(path,{'state':'queued'});pool.submit(run_job,path,kind=='zip',variant,(spacing_y,spacing_x));queued=True
    except BaseException:
        status(path,{'state':'failed','error':'Upload incomplete, exceeds limit, or could not be queued'});raise
    finally:
        if not queued:
            with lock:active_jobs.discard(job)
    return {'job_id':job,'status_url':f'/api/jobs/{job}'}

@app.get('/api/models')
def models():
    return {'selected':prototype.bundle['name'],'models':prototype.model_paths}

@app.get('/api/jobs/{job}')
def get_status(job:str):
    p=job_path(job)/'status.json'
    if not p.exists():return {'state':'uploading'}
    return json.loads(p.read_text())

@app.get('/api/jobs/{job}/results')
def get_results(job:str):
    p=job_path(job)/'output'/'results.json'
    if not p.exists():raise HTTPException(409,'Results not ready')
    return json.loads(p.read_text())

@app.get('/api/jobs/{job}/report.xlsx')
def get_report(job:str):
    p=job_path(job)/'output'/'results.xlsx'
    if not p.exists():raise HTTPException(409,'Report not ready')
    return FileResponse(p,filename='dxa-quality.xlsx')

@app.get('/api/jobs/{job}/annotations.zip')
def get_annotations(job:str):
    path=job_path(job)
    state=path/'status.json'
    if not state.is_file() or json.loads(state.read_text()).get('state')!='complete':
        raise HTTPException(409,'Annotations not ready')
    from .prototype_export import annotation_archive
    return FileResponse(annotation_archive(path/'output'),filename='dxa-annotations.zip',media_type='application/zip')

@app.get('/api/jobs/{job}/images/{index}.png')
def get_image(job:str,index:int):
    p=job_path(job)/'output'/'images'/f'{index}.png'
    if not p.exists():raise HTTPException(404,'Image not found')
    return FileResponse(p)

@app.get('/api/jobs/{job}/overlays/{index}.png')
def get_overlay(job:str,index:int):
    p=job_path(job)/'output'/'overlays'/f'{index}.png'
    if not p.is_file():raise HTTPException(404,'Image not found')
    return FileResponse(p)

@app.get('/api/prototype/demo')
def prototype_demo():
    p=ROOT/'reports'/'prototype_demo'/'results.json'
    if not p.is_file():raise HTTPException(404,'Demo not prepared')
    return json.loads(p.read_text())

@app.get('/api/prototype/demo/{kind}/{index}.png')
def prototype_demo_image(kind:str,index:int):
    if kind not in ('images','overlays'):raise HTTPException(404,'Image not found')
    p=ROOT/'reports'/'prototype_demo'/kind/f'{index}.png'
    if not p.is_file():raise HTTPException(404,'Image not found')
    return FileResponse(p)

@app.get('/api/comparison')
def comparison():
    current=ROOT/'reports'/'comparison_v2.json'
    if current.exists():return json.loads(current.read_text())
    p=ROOT/'reports'/'comparison.json'
    if not p.exists():raise HTTPException(404,'Comparison unavailable')
    result=json.loads(p.read_text())
    extra=ROOT/'reports'/'neural_comparison.json'
    if extra.exists():
        extension=json.loads(extra.read_text());result['results'].update(extension['results']);result['extension_note']=extension['phase']
    return result

@app.get('/api/demo')
def demo():
    p=ROOT/'reports'/'demo'/'results.json'
    if not p.exists():raise HTTPException(404,'Demo not prepared')
    return json.loads(p.read_text())

@app.get('/api/demo/images/{index}.png')
def demo_image(index:int):
    p=ROOT/'reports'/'demo'/'images'/f'{index}.png'
    if not p.exists():raise HTTPException(404,'Image not found')
    return FileResponse(p)

@app.get('/api/demo/report.xlsx')
def demo_report():return FileResponse(ROOT/'reports'/'demo'/'results.xlsx',filename='dxa-demo.xlsx')

@app.get('/')
def home():return FileResponse(ROOT/'web'/'prototype.html')

from .annotations import router as annotation_router,create_project

def run_annotation_import(path,kind,name):
    try:
        status(path,{'state':'running','phase':'annotation_import'})
        files=safe_extract(path/'upload',path/'input') if kind=='zip' else list((path/'input').iterdir())
        def progress(done,total,phase):status(path,{'state':'running','phase':phase,'done':done,'total':total})
        result=create_project(files,name,analyzer,base=path/'input',progress=progress)
        status(path,{'state':'complete','project_id':result['id'],'count':result['count'],'errors':result['errors']})
    except Exception as e:status(path,{'state':'failed','error':f'{type(e).__name__}: {e}'})
    finally:
        with lock:active_jobs.discard(path.name)

@app.post('/api/annotations/import',status_code=202)
async def annotation_import(request:Request,kind:str='zip',name:str='Новый датасет'):
    if kind not in ('zip','dicom','png','jpg'):raise HTTPException(400,'kind=zip, dicom, png или jpg')
    job,path=reserve_job();queued=False;target=path/'upload' if kind=='zip' else path/'input'/('image.'+('dcm' if kind=='dicom' else kind))
    try:
        size=0;limit=(128 if kind=='zip' else 32)*1024**2
        with target.open('wb') as stream:
            async for chunk in request.stream():
                size+=len(chunk)
                if size>limit:raise HTTPException(413,'Превышен размер загрузки')
                stream.write(chunk)
        if not size:raise HTTPException(400,'Пустая загрузка')
        status(path,{'state':'queued'});pool.submit(run_annotation_import,path,kind,name);queued=True
    except BaseException:
        status(path,{'state':'failed','error':'Импорт прерван'});raise
    finally:
        if not queued:
            with lock:active_jobs.discard(job)
    return {'job_id':job}

app.include_router(annotation_router)

@app.get('/annotate')
def annotate():return FileResponse(ROOT/'web'/'annotate.html')

@app.get('/annotate.js')
def annotate_js():return FileResponse(ROOT/'web'/'annotate.js',media_type='text/javascript')
