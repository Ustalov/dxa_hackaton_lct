import json,csv,math
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from openpyxl import Workbook
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
results=json.loads(Path(args.results).read_text())
manifest={f"{r['index']:04d}":r for r in json.loads(Path(args.manifest).read_text())}
names={'hip_position':'Укладка бедра','hip_roi':'Область интереса бедра','spine_position':'Укладка позвоночника','spine_axis':'Ось позвоночника','spine_artifact':'Инородные тела','overall':'Любое нарушение'}
records=[];tasks={k:[] for k in names}
def finite(x):return isinstance(x,(float,int)) and math.isfinite(x)
def score(task,g):
 try:
  if task=='hip_position':
   t=g['trochanter'];return max((t['threshold_center_halfwidth']-t['center_halfwidth'])/t['threshold_center_halfwidth'],(t['depth_mm']-t['threshold_depth_mm'])/t['threshold_depth_mm'])
  if task=='hip_roi':
   m=g['margins'];return max((limit-m[key]['value'])/limit for key,limit in [('greater_trochanter_top',30),('ischium_bottom',30),('femur_lateral_margin',20)])
  if task=='spine_axis':return (g['axis']['value']-5)/5
  if task=='spine_artifact':return g['foreign']['score']-g['foreign']['threshold']
  if task=='spine_position':
   top=g['top_field'];margin=max(x for x in [top['t12_height_mm'],top['l1_top_margin_mm']] if finite(x));threshold=top['threshold_mm']
   areas=[g['iliac'][side]['area_mm2'] for side in ['L','R']]
   return max((threshold-margin)/threshold,*[(50-a)/50 for a in areas])
 except (KeyError,ValueError,TypeError):return None
for row,detail in zip(results['rows'],results['details']):
 case=Path(detail['file']).stem;ref=manifest[case];dec=detail['task_decisions'];g=detail['geometry'];group=ref['study_uid']
 base_record={'case':case,'region':ref['region'],'processing_status':row['processing_status'],'forced_tasks':';'.join(detail.get('forced_violation_tasks',[]))}
 for task,pred in dec.items():
  truth=bool(ref[task]);forced=task in detail.get('forced_violation_tasks',[]);s=None if forced else score(task,g)
  reason='Оценка недоступна; нарушение назначено по принятой политике' if forced else ''
  if task=='hip_roi':reason='; '.join(f"{label}: {g.get('margins',{}).get(key,{}).get('value')} мм" for key,label in [('greater_trochanter_top','верх'),('ischium_bottom','низ'),('femur_lateral_margin','латерально')])
  if task=='hip_position' and not forced:reason=f"центр: {g['trochanter'].get('center_halfwidth')}; выступ: {g['trochanter'].get('depth_mm')} мм"
  if task=='spine_axis':reason=f"угол: {g.get('axis',{}).get('value')}°"
  if task=='spine_position':reason='верх: '+('нарушение' if g.get('top_field',{}).get('violation') else 'норма')+'; подвздошные: '+', '.join(f"{side}={g.get('iliac',{}).get(side,{}).get('present')}" for side in ['L','R'])
  if task=='spine_artifact':reason=f"балл детектора: {g.get('foreign',{}).get('score')}; порог: {g.get('foreign',{}).get('threshold')}"
  rec={**base_record,'task':task,'task_name':names[task],'expert':int(truth),'prediction':int(pred),'match':truth==pred,'error_type':('FP' if pred else 'FN') if truth!=pred else '', 'measurement':reason}
  records.append(rec);tasks[task].append((int(truth),int(pred),group,s))
 truth=any(bool(ref[k]) for k in dec);tasks['overall'].append((int(truth),int(row['quality_class']),group,None))
 records.append({**base_record,'task':'overall','task_name':names['overall'],'expert':int(truth),'prediction':row['quality_class'],'match':int(truth)==row['quality_class'],'error_type':('FP' if row['quality_class'] else 'FN') if int(truth)!=row['quality_class'] else '', 'measurement':'ИЛИ применимых критериев'})
def metrics(y,p):
 tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum())
 ratio=lambda a,b:a/b if b else 0.
 sens=ratio(tp,tp+fn);spec=ratio(tn,tn+fp)
 return dict(n=len(y),TP=tp,FP=fp,FN=fn,TN=tn,sensitivity=sens,specificity=spec,precision=ratio(tp,tp+fp),F1=ratio(2*tp,2*tp+fp+fn),accuracy=ratio(tp+tn,len(y)),balanced_accuracy=(sens+spec)/2)
summary=[];rng=np.random.default_rng(29)
for task,vals in tasks.items():
 y=np.array([r[0] for r in vals]);p=np.array([r[1] for r in vals]);groups=np.array([r[2] for r in vals]);ug=np.unique(groups);inds={g:np.flatnonzero(groups==g) for g in ug}
 m=dict(task=task,name=names[task],**metrics(y,p),studies=len(ug));b=[]
 for _ in range(2000):
  ix=np.concatenate([inds[g] for g in rng.choice(ug,len(ug),replace=True)]);b.append(metrics(y[ix],p[ix]))
 for key in ['F1','sensitivity','specificity','accuracy','balanced_accuracy']:
  lo,hi=np.quantile([r[key] for r in b],[.025,.975]);m[key+'_CI95']=f'{lo:.3f}–{hi:.3f}'
 eligible=[r for r in vals if finite(r[3])];m['auc_n']=len(eligible)
 if eligible and len(set(r[0] for r in eligible))==2:
  m['ROC_AUC']=float(roc_auc_score([r[0] for r in eligible],[r[3] for r in eligible]));m['PR_AP']=float(average_precision_score([r[0] for r in eligible],[r[3] for r in eligible]))
 else:m['ROC_AUC']=None;m['PR_AP']=None
 summary.append(m)
macro=float(np.mean([m['F1'] for m in summary if m['task']!='overall']))
(out/'metrics.json').write_text(json.dumps(dict(scope='Retrospective evaluation, not independent test',unique_images=len(results['rows']),unique_studies=len(set(v[2] for v in tasks['overall'])),macro_f1=macro,tasks=summary,summary=results['summary']),ensure_ascii=False,indent=2))
for file,rows in [('cases.csv',records),('errors.csv',[r for r in records if not r['match']]),('metrics.csv',summary)]:
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (out/file).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
wb=Workbook();wb.remove(wb.active)
for name,rows in [('Метрики',summary),('Все решения',records),('Расхождения',[r for r in records if not r['match']]),('Недоступные оценки',[r for r in records if r['forced_tasks']])]:
 ws=wb.create_sheet(name);fields=list(dict.fromkeys(k for r in rows for k in r));ws.append(fields)
 for r in rows:ws.append([r.get(k) for k in fields])
 ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=min(60,max(14,len(str(col[0].value))+3))
wb.save(out/'validation.xlsx')
lines=['# Результаты текущего конвейера','',f"Повторная обработка {len(results['rows'])} уникальных снимков в Docker без сети: 153 бедра и 99 позвоночников. Это ретроспективная оценка знакомых данных, не независимый тест. Часть данных использована при обучении/адаптации компонентов и выборе правил.",'', '| Задача | n | TP | FP | FN | TN | F1 [95% ДИ] | Чувств. | Специф. |','|---|---:|---:|---:|---:|---:|---|---:|---:|']
for m in summary:lines.append(f"| {m['name']} | {m['n']} | {m['TP']} | {m['FP']} | {m['FN']} | {m['TN']} | {m['F1']:.3f} [{m['F1_CI95']}] | {m['sensitivity']:.3f} | {m['specificity']:.3f} |")
lines+=['',f'Macro-F1 по пяти задачам: **{macro:.3f}**.','', '95% интервалы получены 2000 повторениями bootstrap по исследованиям, seed=29. Они отражают вариабельность этой выборки, но не устраняют смещение из-за использования данных при разработке. Независимость пациентов не подтверждена: идентификаторы пациентов в исходных DICOM не позволяют надёжно разделить их.', '', 'ROC AUC и PR AP в таблице рассчитываются только там, где есть конечный непрерывный балл. Это диагностические баллы по измерениям, не вероятности; quality_prob не добавлен. Отсутствующие оценки исключаются только из AUC, их количество видно по auc_n. Из бинарных метрик они не исключены: применяется политика «нет оценки → нарушение». Общий AUC не рассчитывается.', '', f"Время полного прогона: {results['summary']['wall_seconds']:.1f} с, Success: {results['summary']['success']}, Failure: {results['summary']['failure']}. Это проверка в текущей CPU-среде, не обещание скорости на любом сервере.", '', 'Файлы: [validation.xlsx](validation.xlsx), [все решения](cases.csv), [расхождения](errors.csv), [метрики](metrics.csv). Публичные таблицы содержат условные номера кейсов; медицинские изображения, UID и исходные пути не включены.', '', 'Ошибки — несовпадения с имеющейся экспертной таблицей, а не доказательство неверности эксперта или алгоритма. Пороговое решение и геометрическая точность должны проверяться отдельно.']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'tasks':summary,'macro_f1':macro},ensure_ascii=False,indent=2))
