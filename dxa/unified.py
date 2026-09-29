"""Offline native-DICOM inference: separate anatomy models and five task specialists."""
import json,time
from pathlib import Path
import cv2,numpy as np,torch,joblib
from PIL import Image
from .unified_features import TASKS,geometry_vector,task_matrix,criterion_flags
from .foreign_objects import ForeignObjectNet,foreign_prediction
from .imaging import read_dicom,REGIONS,VIOLATIONS
from .features import FeatureExtractor

ROOT=Path(__file__).resolve().parents[1]
TASK_NAMES=dict(zip(TASKS,VIOLATIONS))

class UnifiedAnalyzer:
    def __init__(self,bundle='/data/dxa-unified',device=None,load_anatomy=True):
        torch.set_num_threads(4);cv2.setNumThreads(1)
        self.bundle=Path(bundle);self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.heads=joblib.load(self.bundle/'task_models.joblib')
        self.foreign=ForeignObjectNet()
        cp=torch.load('/data/dxa-foreign-model/best.pt',map_location='cpu',weights_only=True)
        self.foreign.load_state_dict(cp['state_dict']);self.foreign.to(self.device).eval()
        self.foreign_threshold=json.loads(Path('/data/dxa-foreign-model/threshold.json').read_text())['threshold']
        self.extractor=None;self.nets={}
        if load_anatomy:self.load_anatomy()

    def load_anatomy(self):
        from .anatomy_network import AnatomyNet
        from .hip_v6 import HipNet
        from .isbi2020_adapter import make_model
        for name,net in [('spine',AnatomyNet('spine')),('hip',HipNet())]:
            cp=torch.load(ROOT/'artifacts/png_full_v8'/name/'best.pt',map_location='cpu',weights_only=False)
            assert not cp['audit']['dicom_used']
            net.load_state_dict(cp['state_dict']);self.nets[name]=net.to(self.device).eval()
        self.nets['corners']=make_model(Path('/data/dxa-three-stage/stage3/best.pt')).to(self.device).eval()
        self.extractor=FeatureExtractor(ROOT/'artifacts/resnet18-f37072fd.pth')

    def anatomy(self,raw,meta):
        from .png_spine_v6 import contrast_views
        from .spine_v6_inference import global_prob,geometry_from_masks
        from .hip_v6 import clean_parts,neck_from_parts,geometry_from_parts
        from .vertebra_chain import propose_all,instance_masks
        from .geometry_targets import feature
        from .anatomy_measurements import measure
        if not self.nets:self.load_anatomy()
        region=meta['region'];views=contrast_views(raw)
        outputs=[global_prob(self.nets[region],im) for im in views.values()]
        p=np.mean([o[0] for o in outputs],axis=0);bone=np.mean([o[1] for o in outputs],axis=0)>.5
        if region=='hip':
            labels=(p[1:].argmax(0)+1).astype('uint8');labels[~bone]=0;labels=clean_parts(labels)
            case=geometry_from_parts(labels,meta['spacing'],neck_from_parts(labels))
        else:
            detections,flags,_=propose_all(self.nets['corners'],raw)
            if len(detections)<2:raise ValueError('Insufficient vertebral detections for L5-to-topmost axis')
            labels=instance_masks(detections,bone);plates={};names={}
            for d in detections:
                name=d['anatomical_level'];q=np.asarray(d['corners']);names[str(d['label_id'])]=name
                plates[name+'_upper']=q[:2].tolist();plates[name+'_lower']=q[2:].tolist()
            case=geometry_from_masks(labels,bone,meta['spacing'],plates)
            for key,points in plates.items():case['features'][key]=feature(key,2,points,source='stage3 landmark model, bottom-up user numbering rule')
            case.update(vertebra_names=names,detected_vertebrae=detections,chain_quality_flags=flags,
                        numbering_verified=False,axis_definition='L5_to_topmost',axis_allow_estimate=True,axis_angle_space='square_pixels',
                        maximum_cranial_level='Th11',unavailable_tasks=['independent anatomical numbering','validated Cobb angle'])
        case.update(id='image_'+meta['pixel_hash'][:12],pixel_hash=meta['pixel_hash'],spacing_source=meta['spacing_source'],
                    render_kind='Единый алгоритм: сегментация и геометрия',model_version='unified_stage3')
        case['measurements']=measure(case)
        return case,bone,labels

    def classify(self,raw,meta,case,features=None):
        if features is None:
            if self.extractor is None:self.extractor=FeatureExtractor(ROOT/'artifacts/resnet18-f37072fd.pth')
            features,_=self.extractor.extract([raw],[meta])
        vector,names=geometry_vector(case)
        assert names==self.heads['geometry_names']
        scores={};decisions={};thresholds={}
        for task,item in self.heads['models'].items():
            if task=='spine_axis':continue  # User criterion overrides the historical learned head.
            if not task.startswith(meta['region']+'_'):continue
            x=task_matrix(features,vector[None],item['mode'])
            scores[task]=float(item['model'].predict_proba(x)[0,1]);thresholds[task]=item['threshold']
            decisions[task]=scores[task]>=thresholds[task]
        foreign_score,heat=foreign_prediction(self.foreign,raw)
        if meta['region']=='spine':
            angle=case['measurements']['measurements'].get('axis_L5_topmost',{}).get('value')
            if angle is None or not np.isfinite(angle) or angle<0:
                raise ValueError('Spine axis angle unavailable or invalid; cannot classify as normal')
            scores['spine_axis']=float(angle);thresholds['spine_axis']=5.
            decisions['spine_axis']=angle>5.
            scores['spine_artifact']=foreign_score;thresholds['spine_artifact']=self.foreign_threshold
            decisions['spine_artifact']=foreign_score>=self.foreign_threshold
        violations=[TASK_NAMES[t] for t in TASKS if decisions.get(t,False)]
        # An angle is not a probability. Only its binary rule result enters the aggregate.
        aggregate=[float(decisions[t]) if t=='spine_axis' else p for t,p in scores.items()]
        result=dict(quality_class=int(any(decisions.values())),quality_prob=float(1-np.prod([1-p for p in aggregate])),
                    task_scores=scores,task_decisions=decisions,task_thresholds=thresholds,violation_type='; '.join(violations),
                    task_score_units={t:'degrees' if t=='spine_axis' else 'model_score' for t in scores},
                    axis_decision_rule='square-pixel axis_L5_topmost > 5 degrees; equality is normal; learned axis head is not used',
                    physical_criteria=criterion_flags(case),
                    foreign_body=dict(score=foreign_score,present=foreign_score>=self.foreign_threshold,
                                      threshold=self.foreign_threshold,validated_region='spine',
                                      applicable_to_TZ=meta['region']=='spine',
                                      localization='auxiliary heatmap trained on synthetic masks; not validated pixel segmentation'),
                    probabilities_calibrated=False)
        return result,heat

    def analyze(self,path,destination=None):
        start=time.perf_counter();raw,meta=read_dicom(path);case,bone,labels=self.anatomy(raw,meta)
        prediction,heat=self.classify(raw,meta,case)
        result=dict(**prediction,path_to_study=str(Path(path).parent),study_uid=meta['study_uid'],image_uid=meta['image_uid'],
                    anatomical_region=REGIONS[meta['region']],processing_status='success',
                    time_of_processing=round(time.perf_counter()-start,3),source_path=str(path),metadata=meta,
                    measurements=case['measurements'])
        if destination:save_outputs(Path(destination),raw,case,bone,labels,result,heat)
        return result

def save_outputs(dest,raw,case,bone,labels,result,heat,geometry_source=None):
    """Files only, no viewer registration or UI."""
    import shutil
    from .geometry_render import render
    from PIL import ImageDraw,ImageFont
    dest.mkdir(parents=True,exist_ok=True)
    Image.fromarray(raw).save(dest/'original.png');Image.fromarray(np.uint8(bone)*255).save(dest/'bone.png')
    Image.fromarray(labels).save(dest/'labels.png');np.save(dest/'foreign_heatmap.npy',heat.astype('float32'))
    (dest/'geometry.json').write_text(json.dumps(case,ensure_ascii=False,indent=2))
    (dest/'prediction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if geometry_source:shutil.copy2(geometry_source,dest/'geometry.png')
    else:render(case,raw,bone,dest/'geometry.png',parts=labels)
    # One static image contains the complete anatomy figure, classification and localization.
    geometry=Image.open(dest/'geometry.png').convert('RGB');width,height=geometry.size
    canvas=Image.new('RGB',(width+400,height+180),'white');canvas.paste(geometry,(0,180))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);draw=ImageDraw.Draw(canvas)
    draw.text((20,12),'Класс: '+('нарушение' if result['quality_class'] else 'норма'),font=font,fill='black')
    for i,task in enumerate(t for t in TASKS if t in result['task_scores']):
        value=(f"угол={result['task_scores'][task]:.3f}°; нарушение только >5°" if task=='spine_axis'
               else f"score={result['task_scores'][task]:.3f}")
        line=f"{TASK_NAMES[task]}: {'ДА' if result['task_decisions'][task] else 'нет'}; {value}"
        draw.text((20,44+i*28),line,font=font,fill='black')
    rgb=np.repeat(raw[:,:,None],3,2).astype('float32');alpha=np.clip((heat-.3)/.7,0,1)*.65
    overlay=np.uint8(rgb*(1-alpha[:,:,None])+np.array([255,0,100])*alpha[:,:,None])
    im=Image.fromarray(overlay);im.thumbnail((370,height-160));canvas.paste(im,(width+15,260))
    draw.text((width+15,190),'Инородные тела: тепловая карта',font=font,fill='black')
    draw.text((width+15,220),'Локализация предварительная',font=font,fill='black')
    if case['region']=='hip':draw.text((width+15,230+im.height+50),'Для бедра не валидирована',font=font,fill='black')
    canvas.save(dest/'result.png')
