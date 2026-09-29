from pathlib import Path, PurePosixPath
import json, time, zipfile, stat
import numpy as np
import joblib
from PIL import Image
from openpyxl import Workbook
from .imaging import read_dicom, REGIONS, VIOLATIONS, TARGETS
from .features import FeatureExtractor, context_features

COLUMNS=['path_to_study','study_uid','image_uid','anatomical_region','quality_class','quality_prob','violation_type','processing_status','time_of_processing']

def safe_extract(source, destination, max_files=2000, max_bytes=512*1024*1024):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    files=[];total=0;seen=set()
    with zipfile.ZipFile(source) as archive:
        members=[i for i in archive.infolist() if not i.is_dir()]
        if not members:raise ValueError('Archive is empty')
        if len(members)>max_files:raise ValueError('Archive contains too many files')
        # Validate everything before writing anything.
        for info in members:
            path=PurePosixPath(info.filename.replace('\\','/'))
            if path.is_absolute() or '..' in path.parts or ':' in str(path) or '\x00' in str(path):
                raise ValueError('Unsafe archive path')
            if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Symlinks are not accepted')
            if str(path) in seen:raise ValueError('Duplicate archive member path')
            seen.add(str(path));total+=info.file_size
            if total>max_bytes or info.file_size>32*1024*1024:raise ValueError('Archive size limit exceeded')
            if info.file_size>max(1024*1024,info.compress_size*300):raise ValueError('Excessive compression ratio')
        for info in members:
            path=PurePosixPath(info.filename.replace('\\','/'));target=destination.joinpath(*path.parts)
            if not target.resolve().is_relative_to(destination.resolve()):raise ValueError('Unsafe archive path')
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(info) as src, target.open('xb') as out:
                count=0
                while chunk:=src.read(1024*1024):
                    count+=len(chunk)
                    if count>info.file_size:raise ValueError('Archive member exceeds declared size')
                    out.write(chunk)
            files.append(target)
    return files

def predict_bundle(bundle,features):
    if 'components' in bundle:return np.mean([predict_bundle(b,features) for b in bundle['components']],axis=0)
    return bundle['model'].predict(features)

def export_xlsx(rows,path):
    wb=Workbook();ws=wb.active;ws.title='Results';ws.append(COLUMNS)
    for row in rows:
        ws.append([row.get(c) for c in COLUMNS])
        for cell in ws[ws.max_row]:
            if isinstance(cell.value,str):cell.data_type='s'
    ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
    for column in ws.columns:ws.column_dimensions[column[0].column_letter].width=min(65,max(20,len(str(column[0].value))+2))
    wb.save(path)

from .batch_analyzer import Analyzer
