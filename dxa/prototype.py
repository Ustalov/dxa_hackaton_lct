"""Selected anatomy weights, image region classifier and native-coordinate task rules."""
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from .imaging import REGIONS, read_dicom
from .prototype_geometry import SPACING, aggregate, largest, trochanter_geometry, iliac_geometry, l1_margin, ct_top_rule
from .region_classifier import WEIGHT as REGION_WEIGHT, INFO as REGION_INFO, get_region_classifier

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'selected-models-20260929-upper15mm-inclusive-margins-export'
WEIGHTS = {
    'region': REGION_WEIGHT,
    'hip_lt': Path('/data/dxa-ct-synth/model/v4/best.pt'),
    'hip_parts': ROOT/'artifacts/png_full_v8/hip/best.pt',
    'spine_ct': Path('/data/dxa-ct-synth/model/spine_v1/best.pt'),
    'spine_corners': Path('/data/dxa-three-stage/stage3/best.pt'),
    'spine_bone': ROOT/'artifacts/png_full_v8/spine/best.pt',
    'foreign': Path('/data/dxa-foreign-model-v3/best.pt'),
}
POLICY = Path('/data/dxa-foreign-threshold-audit-v1/policy_validation_f1.json')
TASK_NAMES = {'spine_position':'Некорректная укладка',
              'spine_axis':'Не выровнена ось позвоночника',
              'spine_artifact':'Присутствуют посторонние предметы',
              'hip_position':'Некорректная укладка',
              'hip_roi':'Некорректная область интереса'}


def read_image(path, region_classifier=None):
    path=Path(path)
    with path.open('rb') as f:
        sig=f.read(8)
    if sig.startswith(b'\x89PNG') or sig.startswith(b'\xff\xd8'):
        with Image.open(path) as image:
            if image.width*image.height>16_000_000:
                raise ValueError('Изображение слишком большое')
            raw=np.array(image.convert('L'))
        if min(raw.shape)<32 or raw.max()==raw.min():
            raise ValueError('Пустое или слишком маленькое изображение')
        digest=hashlib.sha256(raw.tobytes()).hexdigest()
        meta={'shape':list(raw.shape),'pixel_hash':digest,
              'study_uid':None,'image_uid':None,'input_format':'raster'}
    else:
        from pydicom.errors import InvalidDicomError
        try:
            raw,meta=read_dicom(path,infer_region=False)
        except InvalidDicomError as exc:
            raise ValueError('Не удалось прочитать DICOM. Проверьте формат и целостность файла.') from exc
        meta['input_format']='dicom'
    prediction=(region_classifier or get_region_classifier()).predict(raw)
    meta.update(region=prediction['region'],side=prediction['side'],
                side_confidence=prediction['score'],region_classification=prediction)
    meta['input_spacing_yx_mm']=meta.get('spacing')
    meta['spacing']=list(SPACING)
    meta['spacing_source']='Организаторы: X=0,6 мм; Y=1,05 мм'
    meta['angle_space']='square_pixels'
    return raw,meta


class PrototypeAnalyzer:
    def __init__(self):
        self.bundle={'name':VERSION}; self.loaded=False
        self.model_paths={k:str(v) for k,v in WEIGHTS.items()}
        missing=[str(p) for p in [*WEIGHTS.values(),POLICY,REGION_INFO] if not p.is_file()]
        if missing: raise FileNotFoundError('Missing selected weights: '+', '.join(missing))

    def load(self):
        if self.loaded:return
        # CPU service deliberately does not compete with the separate GPU training process.
        torch.set_num_threads(4);cv2.setNumThreads(1)
        from .explicit_pipeline import ExplicitAnalyzer
        from .foreign_specificity import ForeignSpecificity
        import segmentation_models_pytorch as smp
        self.legacy=ExplicitAnalyzer(device='cpu')
        self.ct_hip=smp.Unet('resnet34',encoder_weights=None,in_channels=1,classes=3,
                            aux_params=dict(classes=2,dropout=.2))
        split=json.loads((WEIGHTS['spine_ct'].parent/'split.json').read_text())
        self.channels=split['channels'];self.spine_canvas=tuple(split['canvas'])
        self.ct_spine=smp.Unet('resnet34',encoder_weights=None,in_channels=1,classes=len(self.channels),
                              aux_params=dict(classes=2,dropout=.2))
        for net,key in [(self.ct_hip,'hip_lt'),(self.ct_spine,'spine_ct')]:
            net.load_state_dict(torch.load(WEIGHTS[key],map_location='cpu',weights_only=True));net.eval()
        self.foreign=ForeignSpecificity(policy=POLICY)
        self.loaded=True

    @staticmethod
    @torch.inference_mode()
    def ct_probs(net, raw, canvas, flip=False, return_angle=False):
        h,w=raw.shape
        if h>canvas[0] or w>canvas[1]:
            raise ValueError(f'Размер изображения {w}×{h} превышает поле выбранной модели {canvas[1]}×{canvas[0]}; обрезка автоматически не выполняется')
        a=raw[:,::-1] if flip else raw
        padded=np.zeros(canvas,np.uint8);padded[:h,:w]=a
        x=torch.from_numpy(padded.copy()).float()[None,None]/255
        outputs=[net(x.clamp(1e-6,1)**g) for g in (1.,.8,1.25)]
        probs=torch.stack([o[0].sigmoid() for o in outputs]).mean(0)[0].numpy()
        probs[:,h:]=0;probs[:,:,w:]=0
        if return_angle:
            angle=float(torch.stack([o[1] for o in outputs]).mean(0)[0,0])*40
            return probs[:,:h,:w],angle
        return probs[:,:h,:w]

    def hip(self,raw,meta):
        case,_,_=self.legacy.anatomy(raw,meta)
        flipped=meta['side']=='left'
        try:
            p,angle=self.ct_probs(self.ct_hip,raw,(352,320),flipped,return_angle=True)
            fem=p[0]>.5;lt=largest(p[1]>.5)
            prot=(p[2]>.333)&ndi.binary_dilation(fem,iterations=6)
            lt_details,prot=trochanter_geometry(fem,lt,prot,meta['spacing'])
            lt_details.update(rotation_degrees=angle,rotation_reference_degrees=0.,
                              rotation_is_approximate=True,rotation_does_not_override_criteria=True)
        except ValueError as exc:
            lt=prot=np.zeros_like(raw,dtype=bool)
            lt_details={'violation':None,'center_halfwidth':None,'depth_mm':None,'length_mm':None,
                        'reason':str(exc)}
        if flipped:lt=lt[:,::-1];prot=prot[:,::-1]
        m=case['measurements']['measurements'];lines=[];roi_flags=[]
        for key in ('greater_trochanter_top','ischium_bottom','femur_lateral_margin'):
            v=m.get(key,{})
            roi_flags.append(None if v.get('passes') is None else not v['passes'])
            if v.get('value') is not None:
                lines.append({'start':case['features'][v['landmark']]['points'][0],
                              'end':v['target_point_px'],'mm':v['value'],'key':key})
        decisions={'hip_position':lt_details['violation'],'hip_roi':aggregate(roi_flags)}
        info={'margin_lines':lines,'margins':m,'trochanter':lt_details,
              'lateral_image_side':case.get('lateral_image_side')}
        return decisions,info,{'lt':lt,'protrusion':prot}

    def spine(self,raw,meta):
        p=self.ct_probs(self.ct_spine,raw,self.spine_canvas)
        iliac,_=iliac_geometry(p,self.channels,meta['spacing']);top=ct_top_rule(p,self.channels,raw,meta['spacing'])
        flags=[top['violation'],not iliac['L']['present'],not iliac['R']['present']]
        axis={'value':None};foreign={'present':None};errors=[]
        try:
            case,bone,_=self.legacy.anatomy(raw,meta)
            axis=case['measurements']['measurements'].get('axis_L5_topmost',axis)
        except Exception as e:
            case=None;errors.append('Не удалось определить ось: '+str(e))
        if case is not None:
            try:
                # Preserve the calibrated foreign detector's own density evidence.
                from .explicit_rules import apply_iliac_rules,foreign_density_rule,foreign_anatomical_exclusion
                density_case,_=apply_iliac_rules(case,bone,self.legacy.config['iliac_area_threshold_mm2'])
                density,_=foreign_density_rule(raw,meta['spacing'],foreign_anatomical_exclusion(density_case))
                features=self.foreign.features(raw)
                foreign=self.foreign.decide(features,density['present'],self.foreign.policy)
            except Exception as e:
                errors.append('Не удалось проверить посторонние предметы: '+str(e))
        decisions={'spine_position':aggregate(flags),
                   'spine_axis':None if axis.get('value') is None else bool(axis['value']>5),
                   'spine_artifact':foreign['present']}
        return decisions,{'iliac':iliac,'top_field':top,'direct_l1_margin_diagnostic':l1_margin(p,self.channels,meta['spacing']),
                          'axis':axis,'foreign':foreign,'warnings':errors},{}

    def analyze(self,files,output,base=None,progress=None,variant=None,spacing_yx=SPACING):
        from .prototype_render import render
        from .service import export_xlsx
        spacing_yx=tuple(float(v) for v in spacing_yx)
        if len(spacing_yx)!=2 or not all(np.isfinite(v) and .01<=v<=10 for v in spacing_yx):
            raise ValueError('Pixel Spacing должен быть от 0,01 до 10 мм/пиксель по каждой оси')
        start=time.monotonic();output=Path(output)
        for name in ('images','overlays','masks'):(output/name).mkdir(parents=True,exist_ok=True)
        rows=[];details=[];files=list(files)
        for index,path in enumerate(files):
            path=Path(path);t=time.monotonic()
            name=str(path.relative_to(base)) if base else path.name
            row={'path_to_study':name,'study_uid':None,'image_uid':None,'anatomical_region':None,
                 'quality_class':None,'quality_prob':None,'violation_type':'','processing_status':'Failure'}
            detail={'index':index,'file':name,'has_image':False,'has_overlay':False}
            meta=None;decisions={};geometry={}
            try:
                if path.stat().st_size>32*1024**2:raise ValueError('Файл превышает 32 МиБ')
                raw,meta=read_image(path)
                meta['spacing']=list(spacing_yx)
                meta['spacing_source']='Pixel Spacing из формы загрузки; по умолчанию значения организаторов'
                detail['metadata']=meta
                Image.fromarray(raw).save(output/'images'/f'{index}.png');detail['has_image']=True
                self.load()
                decisions,geometry,masks=(self.hip if meta['region']=='hip' else self.spine)(raw,meta)
                render(raw,meta['region'],geometry,masks,output/'overlays'/f'{index}.png',spacing_yx=spacing_yx)
                detail['has_overlay']=True
                for key,mask in masks.items():Image.fromarray(np.uint8(mask)*255).save(output/'masks'/f'{index}_{key}.png')
            except Exception as e:
                detail['error']=str(e);detail['display_state']='review'
            if meta is not None:
                applicable=('hip_position','hip_roi') if meta['region']=='hip' else ('spine_position','spine_axis','spine_artifact')
                raw_decisions={k:decisions.get(k) for k in applicable}
                unknown_keys=[k for k,v in raw_decisions.items() if v is None]
                unknown=[TASK_NAMES[k] for k in unknown_keys]
                decisions={k:True if v is None else v for k,v in raw_decisions.items()}
                flag=aggregate(decisions.values())
                row.update(study_uid=meta.get('study_uid'),image_uid=meta.get('image_uid'),anatomical_region=REGIONS[meta['region']],
                           quality_class=int(flag),
                           violation_type=';'.join(TASK_NAMES[k] for k,v in decisions.items() if v is True),
                           processing_status='Failure' if unknown or detail.get('error') else 'Success')
                detail.update(task_decisions=decisions,geometry=geometry,unknown_tasks=unknown,
                              raw_task_decisions=raw_decisions,forced_violation_tasks=unknown_keys,
                              unknown_policy='violation',display_state='violation' if flag else 'normal')
                if unknown:
                    note='Нарушение выставлено по правилу: не удалось оценить — '+'; '.join(unknown)
                    detail['error']='. '.join(filter(None,[detail.get('error'),note]))
            row['time_of_processing']=round(time.monotonic()-t,3)
            rows.append(row);details.append(detail)
            if progress:progress(index+1,len(files),'analysis')
        result={'rows':rows,'details':details,'summary':{
            'model':VERSION,'files':len(rows),'success':sum(r['processing_status']=='Success' for r in rows),
            'failure':sum(r['processing_status']=='Failure' for r in rows),'wall_seconds':round(time.monotonic()-start,3),
            'models':self.model_paths,'spacing_yx_mm':list(spacing_yx),'angle_space':'square_pixels'}}
        tmp=output/'results.tmp';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(output/'results.json')
        export_xlsx(rows,output/'results.xlsx')
        return result
