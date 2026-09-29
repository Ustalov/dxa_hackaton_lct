"""Criterion-specific admission for acquisition QC. ROI overlays alone are allowed."""
from .specialists import TASKS

def applicable_tasks(row):
    return TASKS[:3] if row.get('region')=='spine' else TASKS[3:] if row.get('region')=='hip' else []

def task_blocks(row):
    active=applicable_tasks(row)
    blocks={t:'Другая область исследования' for t in TASKS if t not in active}
    for t in active:
        if row.get('projection','unknown')!='ap':
            blocks[t]='Нужна подтверждённая фронтальная проекция';continue
        full=row.get('field_of_view','unknown')=='full'
        if t in ('spine_position','spine_artifact','hip_roi') and not full:
            blocks[t]='Исходное полное поле сканирования не подтверждено';continue
        if t=='hip_position' and not (full or row.get('hip_landmarks_visible') is True):
            blocks[t]='Подтвердите видимость большого/малого вертелов, шейки и седалищной кости';continue
        if t in ('spine_axis','hip_roi') and row.get('spacing_assumed'):
            blocks[t]='Рабочий масштаб датасета не подтверждён физической калибровкой';continue
        if t=='spine_axis' and row.get('spacing') is None and row.get('pixel_aspect_yx') is None:
            blocks[t]='Неизвестно физическое соотношение сторон пикселя';continue
        if t=='hip_roi' and row.get('spacing') is None:
            blocks[t]='Нет калибровки в мм для отступов 30/20 мм';continue
        reason=row.get('task_exclusions',{}).get(t)
        if reason:blocks[t]=reason
    return blocks

def export_admission(row):
    blocks=task_blocks(row)
    usable=row.get('status')=='confirmed' and bool(row.get('group_id')) and row.get('source_training_allowed',True)
    mask={t:bool(usable and t not in blocks and type(row['labels'].get(t)) is int and row['labels'][t] in (0,1)) for t in TASKS}
    ids=applicable_tasks(row)
    quick=row.get('review_mode')=='quick' and row.get('annotation_scope')=='displayed_image'
    complete=row.get('status')=='confirmed' and bool(ids) and all((quick or t not in blocks) and type(row['labels'].get(t)) is int and row['labels'][t] in (0,1) for t in ids)
    return {'task_blocks':blocks,'training_mask':mask,
            'quality_class':int(any(row['labels'][t] for t in ids)) if complete else None}
