"""Static review images with masks, measurements and reasons on one canvas."""
import json,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

TASK_NAMES={'spine_position':'Охват позвоночника','spine_axis':'Ось >5°','spine_artifact':'Инородные тела',
            'hip_roi':'Отступы бедра','hip_position':'Малый вертел / ротация'}

def state(v):return 'НЕ ОПРЕДЕЛЕНО' if v is None else 'нарушение' if v else 'норма'
def num(v):return 'не определено' if v is None else f'{v:.2f}'

def save_explicit_outputs(dest,raw,case,bone,labels,iliac,foreign,result):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    for name,a in [('original',raw),('bone',np.uint8(bone)*255),('labels',labels),('iliac_mask',iliac),('foreign_mask',foreign)]:
        Image.fromarray(a).save(dest/(name+'.png'))
    (dest/'geometry.json').write_text(json.dumps(case,ensure_ascii=False,indent=2))
    (dest/'prediction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    h,w=raw.shape;scale=2.;iw=round(w*scale);ih=round(h*scale)
    overlay=np.repeat(raw[:,:,None],3,2).astype(float)
    overlay[bone>0]=overlay[bone>0]*.8+np.array([0,220,100])*.2
    palette=[[0,220,240],[255,165,0],[220,70,220],[110,240,70],[255,210,60],[150,120,255],[0,170,255]]
    for k in np.unique(labels):
        if k:overlay[labels==k]=overlay[labels==k]*.65+np.array(palette[(int(k)-1)%len(palette)])*.35
    overlay[iliac>0]=overlay[iliac>0]*.55+np.array([255,145,0])*.45
    overlay[foreign>0]=[255,40,110]
    im=Image.fromarray(np.uint8(np.clip(overlay,0,255))).resize((iw,ih))
    d=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    sy,sx=case['spacing_yx_mm']
    for x in np.arange(0,w,10/sx):d.line([(x*scale,0),(x*scale,ih)],fill=(80,80,80),width=1)
    for y in np.arange(0,h,10/sy):d.line([(0,y*scale),(iw,y*scale)],fill=(80,80,80),width=1)
    def line(p,color,width=2):
        q=np.array(p,float)*scale
        if len(q)>=2:d.line([tuple(v) for v in q],fill=color,width=width)
    for key,c in case.get('contours',{}).items():line(c,'orange',3)
    for det in case.get('detected_vertebrae',[]):
        q=np.asarray(det['corners']);line(q[[0,1,3,2,0]],'white',1)
        center=np.array(det['center'])*scale
        d.text(tuple(center),det['anatomical_level'],font=font,fill='white',stroke_width=2,stroke_fill='black')
    m=case['measurements']['measurements']
    axis=m.get('axis_L5_topmost',{}).get('endpoints_px')
    if axis:
        line(axis,'cyan',3);b=axis[0];line([[b[0],0],b],'yellow',2)
    if case['region']=='spine':
        for key in ['L1_upper','L1_lower']:
            p=case['features'].get(key,{}).get('points',[])
            if len(p)==2:line(p,'magenta',3)
        p=case['features'].get('L1_upper',{}).get('points',[])
        if len(p)==2:
            q=min(p,key=lambda a:a[1]);line([q,[q[0],0]],'magenta',2)
    else:
        for key in ['greater_trochanter_top','ischium_bottom','femur_lateral_margin']:
            v=m[key];a=case['features'].get(v.get('landmark',''),{}).get('points',[])
            if a and v.get('target_point_px'):line([a[0],v['target_point_px']],'cyan',3)
        for key,f in case['features'].items():
            p=f.get('points',[])
            if key=='lesser_trochanter' and len(p)==3:line(p,'yellow',2)
        roi=case.get('diagnostics',{}).get('neck_roi',{})
        if roi.get('cut'):line(roi['cut'],'orange',3)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    lines=[case['id'],f"Масштаб X={sx:g}, Y={sy:g} мм/пкс; сетка 10 мм",'Угол оси: квадратные пиксели.','']
    reference=result.get('reference',{})
    for task,v in result['task_decisions'].items():
        if task=='hip_position':lines.append('Малый вертел: сохранён, не проверяли заново.');continue
        lines.append(TASK_NAMES[task]+': '+state(v))
        if task in reference:lines.append('  Эксперт: '+state(bool(reference[task])))
    if 'expected_violation' in result:lines+=['Ожидаемый охват: '+state(result['expected_violation'])]
    lines+=['']
    if case['region']=='spine':
        v=m['L1_top_margin'];lines += [f"L1 → верх: {num(v.get('value'))} мм",f"Требуется ≥ {num(v.get('required_mm'))} мм (½ L1)",
             f"Ось: {num(m.get('axis_L5_topmost',{}).get('value'))}°; порог >5°"]
        for side,title in [('left','ПК слева на изображении'),('right','ПК справа на изображении')]:
            c=case['iliac_rules']['selected'][side];lines.append(title+(': найдена' if c else ': не найдена'))
            if c:
                lines.append(f"  S={c['area_mm2']:.2f} мм²; контур={c['upper_contour_length_mm']:.2f} мм")
                lines.append('  Прошли: '+', '.join(k for k,v in c['criteria'].items() if v))
        lines+=['ПК: уровень L4–L5 ИЛИ контур ≥20 мм',f"ИЛИ площадь ≥{case['iliac_rules']['area_threshold_mm2']:.2f} мм².",
            'Плюс связность с нижней полосой 10 мм.','Оранжевое — ПК; розовое — кандидаты металла.',
            'Зелёное — костная маска нейросети.', 'Нумерация автоматическая; возможны сдвиги.']
        endplate=m.get('endplate_angle',{})
        lines.append('Угол замыкательных пластинок: '+num(endplate.get('value'))+'°')
        lines.append('Описательная оценка, не подтверждённый Кобб.')
    else:
        for k,title,limit in [('greater_trochanter_top','Большой вертел → верх',30),('ischium_bottom','Седалищная кость → низ',30),('femur_lateral_margin','Бедро → боковой край',20)]:
            v=m[k];lines.append(f"{title}: {num(v.get('value'))} мм; >{limit}")
        lines+=['Голубое — бедро; оранжевое — таз.','Сторона бокового края определяется по тазу.','Результат по бедру неполный: ротация отложена.']
        lesser=m.get('lesser_trochanter',{})
        if lesser.get('valid'):
            lines += ['МВ, сохранённые прежние измерения:',
                f"  Протяжённость {num(lesser.get('length_mm'))} мм",
                f"  Выступ медиально {num(lesser.get('medial_protrusion_mm'))} мм"]
        else:lines.append('МВ: прежний ориентир не определён.')
    warnings=result.get('review_reasons',[])
    if warnings:lines+=['','ПРОВЕРИТЬ:']+warnings
    wrapped=[]
    for s in lines:wrapped+=textwrap.wrap(s,width=65) or ['']
    width=iw*2+790;height=max(ih+130,len(wrapped)*25+95)
    canvas=Image.new('RGB',(width,height),'#f7f9fc');draw=ImageDraw.Draw(canvas)
    orig=Image.fromarray(raw).convert('RGB').resize((iw,ih));canvas.paste(orig,(15,65));canvas.paste(im,(iw+35,65))
    draw.text((15,20),'Исходный снимок',font=font,fill='black');draw.text((iw+35,20),'Сегментации, геометрия и правила',font=font,fill='black')
    x=iw*2+60
    for n,s in enumerate(wrapped):draw.text((x,20+n*25),s,font=small,fill='#112238')
    draw.text((15,ih+80),'Автоматические измерения. Площадь ПК — видимый фрагмент маски. Цветные позвонки: четырёхугольник ∩ кость.',font=small,fill='black')
    canvas.save(dest/'result.png')
