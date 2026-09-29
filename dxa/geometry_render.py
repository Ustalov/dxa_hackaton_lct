"""Static scientific figures, with physical axes and explicit unavailable values."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .anatomy_measurements import measure,points

def render(case,rgb,bone,destination,parts=None):
    sy,sx=case['spacing_yx_mm'];h,w=case['shape'];extent=(-sx/2,(w-.5)*sx,(h-.5)*sy,-sy/2)
    fig,ax=plt.subplots(figsize=(10,8));fig.subplots_adjust(left=.07,right=.55,top=.9,bottom=.09)
    ax.imshow(rgb,extent=extent,cmap='gray')
    rgba=np.zeros((h,w,4));rgba[bone>0]=[0,.9,.5,.16]
    if parts is not None and case['region']=='hip':
        rgba[parts==1]=[0,.9,1,.2];rgba[parts==2]=[1,.6,.1,.2]
    if parts is not None and case['region']=='spine':
        for k,color in enumerate([[0,1,1,.3],[1,1,0,.3],[1,.3,1,.3],[.3,1,.4,.3]],1):
            rgba[parts==k]=color
        if case.get('vertebra_names'):
            palette=plt.get_cmap('tab20')
            for k in np.unique(parts):
                if k>4:
                    color=list(palette(int(k)-1));color[3]=.3;rgba[parts==k]=color
    ax.imshow(rgba,extent=extent)
    if parts is not None:
        for k in np.unique(parts):
            if k==0:continue
            mask=parts==k
            ax.contour(np.arange(w)*sx,np.arange(h)*sy,mask.astype(float),levels=[.5],linewidths=.6,colors=['cyan' if k==1 else 'orange' if case['region']=='hip' else 'white'])
            if case['region']=='spine':
                yy,xx=np.where(mask);name=case.get('vertebra_names',{}).get(str(int(k)),f'L{k}')
                ax.text(np.median(xx)*sx,np.median(yy)*sy,name,color='white',fontsize=12,ha='center',bbox=dict(facecolor='black',alpha=.6,pad=1))
    for contour in case.get('contours',{}).values():
        p=np.asarray(contour)*[sx,sy]
        if len(p):ax.plot(p[:,0],p[:,1],color='orange',lw=2)
    short={'greater_trochanter':'БВ','ischium':'СК','lateral_femur':'ЛК','lesser_trochanter':'МВ','iliac_left':'ПК слева','iliac_right':'ПК справа'}
    colors=['#00ffff','#ffff00','#ff77ff','#55ff88']
    for i,(key,f) in enumerate(case['features'].items()):
        p=points(case,key,f['arity'])
        if p is None:continue
        p=p*[sx,sy];color=colors[i%4]
        if key.startswith('L'):color=colors[(int(key[1])-1)%4]
        ax.plot(p[:,0],p[:,1],'o-',color=color,ms=3,lw=1)
        label=short.get(key,key.replace('_upper',' верх').replace('_lower',' низ'))
        if not case.get('vertebra_names') or key in ('L1_upper','L5_lower','iliac_left','iliac_right'):
            ax.annotate(label,p[0],xytext=(3,-4),textcoords='offset points',color=color,fontsize=7,bbox=dict(facecolor='black',alpha=.45,pad=1))
        if key=='lesser_trochanter':
            a,q,b=p;v=b-a;foot=a+v*np.dot(q-a,v)/max(np.dot(v,v),1e-9)
            ax.plot([a[0],b[0]],[a[1],b[1]],'--',color=color,lw=1.5)
            ax.plot([q[0],foot[0]],[q[1],foot[1]],color='#ff5555',lw=2)
        if case['region']=='hip' and key in ['greater_trochanter','ischium','lateral_femur']:
            x,y=p[0];target=(x,0) if key=='greater_trochanter' else (x,(h-1)*sy)
            if key=='lateral_femur':target=(0 if case['lateral_image_side']=='left' else (w-1)*sx,y)
            ax.annotate('',xy=target,xytext=(x,y),arrowprops=dict(arrowstyle='<->',color=color))
    roi=case.get('diagnostics',{}).get('neck_roi')
    if roi:
        if roi.get('polygon'):
            polygon=np.array(roi['polygon'])*[sx,sy];polygon=np.concatenate([polygon,polygon[:1]])
            ax.plot(polygon[:,0],polygon[:,1],color='orange',lw=1)
        cut=np.array(roi['cut'])*[sx,sy];ax.plot(cut[:,0],cut[:,1],'--',color='orange',lw=2)
    metrics=measure(case)['measurements'];lines=[]
    names={'greater_trochanter_top':'БВ → верх (>30 мм)','ischium_bottom':'СК → низ (>30 мм)',
           'femur_lateral_margin':'ЛК → бок (>20 мм)','L1_height':'Высота L1','L1_top_margin':'L1 → верх (≥½ высоты)',
           'axis_L1_L5':'Ось L1–L5: отклонение','visible_axis_L1_L4':'Наклон видимого L1–L4',
           'iliac_left':'ПК слева: отступ','iliac_right':'ПК справа: отступ','endplate_angle':'Угол линий L1/L4, не Кобб'}
    if case.get('axis_definition')=='L5_to_topmost':
        top=metrics['axis_L5_topmost'].get('top_level') or '?'
        names['axis_L5_topmost']=f'Ось L5–{top}: отклонение'
    for key,value in metrics.items():
        if case.get('axis_definition')=='L5_to_topmost' and key in ('axis_L1_L5','visible_axis_L1_L4'):continue
        if key=='lesser_trochanter':
            if value.get('valid'):
                lines += [f"МВ: протяжённость {value['length_mm']:.1f} мм",f"МВ: выступ медиально {value['medial_protrusion_mm']:.1f} мм",
                          'Форма МВ: '+('L≥2; D≥1; L>D' if value['passes_visibility_shape_rule'] else 'критерий не выполнен')]
            else:lines.append('Малый вертел: не определён')
            continue
        v=value.get('value');unit='°' if value.get('unit')=='degrees' else ' мм'
        lines.append(names[key]+': '+('не определено' if v is None else f'{v:.1f}{unit}'))
        if key=='axis_L5_topmost' and value.get('estimated'):lines.append('Ось: оценка по предсказанным ориентирам.')
    if case['region']=='spine':
        upper=points(case,'L1_upper',2);lower=points(case,'L5_lower',2)
        if case.get('axis_definition')=='L5_to_topmost':
            endpoints=metrics['axis_L5_topmost'].get('endpoints_px')
            if endpoints:
                b,a=np.asarray(endpoints)*[sx,sy]
                ax.plot([a[0],b[0]],[a[1],b[1]],color='cyan',lw=2)
                ax.plot([b[0],b[0]],[0,b[1]],'--',color='yellow',lw=1)
            else:lines+=['Граница крайнего позвонка не видна:', 'ось не подменяется коротким участком.']
        elif upper is not None and lower is not None:
            a=upper.mean(0)*[sx,sy];b=lower.mean(0)*[sx,sy]
            ax.plot([a[0],b[0]],[a[1],b[1]],color='cyan',lw=2)
            ax.plot([b[0],b[0]],[0,b[1]],'--',color='yellow',lw=1)
        elif 'visible_axis_L1_L4' in metrics:
            p=np.array(metrics['visible_axis_L1_L4']['endpoints_px'])*[sx,sy]
            ax.plot(p[:,0],p[:,1],'--',color='cyan',lw=2)
            ax.plot([p[-1,0]]*2,[0,p[-1,1]],':',color='yellow',lw=1.5)
            lines+=['Пунктир: ось видимого участка.', 'Она НЕ заменяет ось L1–L5.']
        if case.get('vertebra_names'):
            lines+=['Нумерация снизу: L5 (правило пользователя).',
                    'Пропуск позвонка сдвигает номера.',
                    'Цветные маски: четырёхугольник ∩ кость.']
            if any(f!='bottom_up_numbering_assumes_no_missed_vertebrae_and_lowest_is_L5' for f in case.get('chain_quality_flags',[])):
                lines+=['Есть флаги проверки цепочки/нижнего уровня.']
    ax.text(1.04,.98,'\n'.join(lines),transform=ax.transAxes,va='top',fontsize=9,linespacing=1.6)
    ax.text(1.04,.13,'Автоматический результат.\nНе экспертный эталон.\nМасштаб Y/X: '+f'{sy:g}/{sx:g} мм/пкс',transform=ax.transAxes,fontsize=8)
    ax.set_xticks(np.arange(0,w*sx,10),minor=True);ax.set_yticks(np.arange(0,h*sy,10),minor=True)
    ax.grid(which='minor',color='white',alpha=.13);ax.set_xlabel('мм; сетка 10 мм');ax.set_ylabel('мм')
    if case.get('axis_angle_space')=='square_pixels':
        ax.set_aspect(sx/sy)
        ax.text(1.04,.045,'Пиксели показаны квадратными.\nУгол оси измерен в координатах пикселей.\nРасстояния — с физическим масштабом.',transform=ax.transAxes,fontsize=8)
    fig.suptitle(case['id']+' — '+('бедро' if case['region']=='hip' else 'позвоночник')+'\n'+case.get('render_kind','Псевдоразметка по PNG'),fontsize=12)
    fig.savefig(destination,dpi=105);plt.close(fig)
