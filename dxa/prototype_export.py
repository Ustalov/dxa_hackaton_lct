"""Downloadable, cached annotation bundle for a completed analysis job."""
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from zipfile import ZipFile, ZIP_DEFLATED


_archive_lock = threading.Lock()


def annotation_archive(output: Path) -> Path:
    output = Path(output)
    destination = output / 'annotations.zip'
    with _archive_lock:
        if destination.is_file():
            return destination
        result = json.loads((output / 'results.json').read_text())
        members = [output / 'results.json', output / 'results.xlsx']
        for detail in result['details']:
            index = int(detail['index'])
            if detail.get('has_overlay'):
                members.append(output / 'overlays' / f'{index}.png')
            for mask in sorted((output / 'masks').glob(f'{index}_*.png')):
                if re.fullmatch(r'\d+_[A-Za-z0-9_-]+\.png', mask.name):
                    members.append(mask)
        descriptor, temporary = tempfile.mkstemp(prefix='.annotations-', suffix='.tmp', dir=output)
        os.close(descriptor)
        try:
            with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=1) as archive:
                for path in members:
                    if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
                        raise ValueError('Annotation file is outside the result directory')
                    archive.write(path, path.relative_to(output).as_posix())
                archive.writestr('README.txt',
                    'Разметки DXA\n\n'
                    'overlays/<index>.png — снимки с итоговой визуализацией.\n'
                    'masks/<index>_<structure>.png — доступные бинарные маски (255 = объект).\n'
                    'results.xlsx — таблица решений; results.json — измерения и причины ошибок.\n'
                    'Номер index начинается с 0 и соответствует details[index] и rows[index] '
                    'в results.json. Исходное имя — details[index].file.\n'
                    'Изображения сохранены в исходной пиксельной сетке. '
                    'Для физического соотношения сторон бедра учитывайте metadata.spacing=[Y,X].\n'
                    'При ошибке обработки разметка может отсутствовать; такая строка сохраняется в отчёте.\n'
                    'Исходные DICOM в архив не включены.\n')
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return destination
