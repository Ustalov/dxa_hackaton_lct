from pathlib import Path
from collections import Counter,OrderedDict
import json,time,tempfile
import numpy as np
from PIL import Image
from .imaging import read_dicom,REGIONS,VIOLATIONS,TARGETS
from .inference import ModelRunner

class Analyzer:
    """Bounded image decoding, isolated failures, and an offline model runner."""
    def __init__(self,artifacts):
        self.artifacts=Path(artifacts);self._runners=OrderedDict()
        self.default_runner=ModelRunner(self.artifacts);self.bundle=self.default_runner.bundle
        self._runners[self.bundle['name']]=self.default_runner

    def runner(self,variant=None):
        name=variant or self.bundle['name']
        if name not in self._runners:
            self._runners[name]=ModelRunner(self.artifacts,name)
            while len(self._runners)>2:
                old=next(k for k in self._runners if k!=self.bundle['name'])
                if old==name:break
                del self._runners[old]
        self._runners.move_to_end(name);return self._runners[name]

    @staticmethod
    def _write_json(path,data):
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(path)

    def analyze(self,files,output,base=None,progress=None,variant=None):
        from .service import COLUMNS,export_xlsx
        start=time.monotonic();runner=self.runner(variant);bundle=runner.bundle
        output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/'images').mkdir(exist_ok=True)
        rows=[];details=[];metas=[];cache={};assignment={};read_times=[];failures={};measurements={};parts=[];successful=[];predictions={}
        files=list(files)
        with tempfile.TemporaryDirectory(prefix='.pixels-',dir=output) as temp:
            scratch=Path(temp)
            for index,file in enumerate(files):
                t=time.monotonic();path=Path(file);display=str(path)
                row=dict.fromkeys(COLUMNS);row.update(path_to_study=display,processing_status='Failure',violation_type='')
                detail={'index':index,'file':display}
                try:
                    display=str(path.relative_to(base)) if base else str(path);row['path_to_study']=display;detail['file']=display
                    if path.stat().st_size>32*1024*1024:raise ValueError('File exceeds 32 MiB limit')
                    a,m=read_dicom(path);detail['metadata']=m
                    row.update(study_uid=m['study_uid'],image_uid=m['image_uid'],anatomical_region=REGIONS[m['region']])
                    # Register for inference only after the required preview is successfully written.
                    Image.fromarray(a).save(output/'images'/f'{index}.png');detail['preview']=f'images/{index}.png'
                    key=(m['study_uid'],m['pixel_hash'],tuple(m['spacing']),m['region'],m['side'])
                    if key not in cache:
                        uid=len(metas);np.save(scratch/f'{uid}.npy',a,allow_pickle=False);metas.append(m);cache[key]=uid
                    assignment[index]=cache[key]
                    del a
                except Exception as e:detail['error']=f'{type(e).__name__}: {e}'
                read_times.append(time.monotonic()-t);rows.append(row);details.append(detail)
                if progress:progress(index+1,len(files),'reading')
            compute_start=time.monotonic()

            def extract(ids):
                try:
                    images=[np.load(scratch/f'{i}.npy',allow_pickle=False) for i in ids]
                    f,d=runner.extract(images,[metas[i] for i in ids])
                    if not all(len(v)==len(ids) and np.isfinite(v).all() for v in f.values()):raise ValueError('Invalid extracted features')
                    parts.append(f);successful.extend(ids);measurements.update(zip(ids,d))
                except Exception as e:
                    if len(ids)>1:
                        mid=len(ids)//2;extract(ids[:mid]);extract(ids[mid:])
                    else:failures[ids[0]]=f'Feature extraction: {type(e).__name__}: {e}'

            for start_at in range(0,len(metas),8):
                extract(list(range(start_at,min(start_at+8,len(metas)))))
                if progress:progress(min(start_at+8,len(metas)),len(metas),'features')
            if successful:
                features={k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
                good_metas=[metas[i] for i in successful]
                context_ready=True
                try:runner.contextualize(features,good_metas)
                except Exception as e:
                    context_ready=False
                    for uid in successful:failures[uid]=f'Context extraction: {type(e).__name__}: {e}'

                def predict(ids):
                    try:
                        f={k:v[ids] for k,v in features.items()};p,b=runner.predict(f)
                        if len(p)!=len(ids):raise ValueError('Model returned incorrect row count')
                        for j,pos in enumerate(ids):predictions[successful[pos]]=(p[j],b[j])
                    except Exception as e:
                        if len(ids)>1:
                            mid=len(ids)//2;predict(ids[:mid]);predict(ids[mid:])
                        else:failures[successful[ids[0]]]=f'Prediction: {type(e).__name__}: {e}'
                if context_ready:
                    for start_at in range(0,len(successful),16):predict(list(range(start_at,min(start_at+16,len(successful)))))
            compute_seconds=time.monotonic()-compute_start
            multiplicity=Counter(assignment.values());failed_groups={metas[i]['study_uid'] for i in failures}
            for index,uid in assignment.items():
                if uid not in predictions:
                    details[index]['error']=failures.get(uid,'No prediction available');continue
                p,b=predictions[uid];m=metas[uid];head=0 if m['region']=='spine' else 1;ids=[2,3,4] if head==0 else [5,6]
                violations=[VIOLATIONS[j-2] for j in ids if b[j]]
                rows[index].update(quality_class=int(b[head]),quality_prob=float(p[head]),violation_type=';'.join(violations),processing_status='Success')
                read_times[index]+=compute_seconds/max(len(metas),1)/multiplicity[uid]
                thresholds=np.asarray(bundle['thresholds']);raw_types=p[ids]>=thresholds[ids]
                details[index].update(geometry=measurements[uid],probabilities={t:float(p[j]) for j,t in enumerate(TARGETS)},
                    thresholds=thresholds.tolist(),quality_type_disagreement=bool(b[head])!=bool(violations),unique_image_index=uid,
                    model=bundle['name'],decision_policy=bundle.get('policy','independent'),
                    type_decision_adjusted=bool(np.any(raw_types!=b[ids])),
                    context_incomplete=m['study_uid'] in failed_groups and bool(runner.keys&{'context','dino_small_context'}))
            for row,seconds in zip(rows,read_times):row['time_of_processing']=round(seconds,6)
            summary={'model':bundle['name'],'files':len(rows),'unique_images':len(metas),'studies':len({r['study_uid'] for r in rows if r['study_uid']}),
                     'success':sum(r['processing_status']=='Success' for r in rows),'failure':sum(r['processing_status']=='Failure' for r in rows),
                     'wall_seconds':time.monotonic()-start,'timing_definition':'Model load, read, inference and XLSX generation; excludes HTTP upload and ZIP extraction. Per-image times allocate batch inference.',
                     'quality_type_policy':bundle.get('policy','independent'),'training_partition':bundle.get('training_partition','see model card')}
            result={'summary':summary,'rows':rows,'details':details}
            # Preserve completed rows even if XLSX export fails (for example, disk full).
            self._write_json(output/'results.json',result)
            try:export_xlsx(rows,output/'results.xlsx')
            except Exception as e:
                summary['export_error']=f'{type(e).__name__}: {e}';self._write_json(output/'results.json',result);raise
            summary['wall_seconds']=time.monotonic()-start;self._write_json(output/'results.json',result)
        if progress:progress(len(files),len(files),'done')
        return result
