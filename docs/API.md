# API и выходные форматы

Все запросы требуют HTTP Basic. Swagger: /docs. POST принимает байты файла, Content-Type: application/octet-stream, не multipart.

| Метод | Путь | Смысл |
|---|---|---|
| POST | /api/jobs?kind=zip&filename=batch.zip&spacing_x=0.6&spacing_y=1.05 | Загрузить архив; kind также dicom/png/jpg |
| GET | /api/jobs/{id} | uploading/queued/running/complete/failed |
| GET | /api/jobs/{id}/results | JSON результатов |
| GET | /api/jobs/{id}/report.xlsx | Таблица |
| GET | /api/jobs/{id}/annotations.zip | Визуализации, доступные маски и отчёты |
| GET | /api/jobs/{id}/images/{index}.png | Исходное изображение |
| GET | /api/jobs/{id}/overlays/{index}.png | Разметка |
| GET | /api/models | Состав моделей |
| GET | /health | Доступность сервиса |

```bash
curl -u imagelab -H 'Content-Type: application/octet-stream' --data-binary @batch.zip 'http://127.0.0.1:8095/api/jobs?kind=zip&filename=batch.zip'
```

curl запросит пароль; замените логин на свой. При завершении задачи скачайте report.xlsx и annotations.zip с тем же способом авторизации.

XLSX: path_to_study, study_uid, image_uid, anatomical_region, quality_class, quality_prob, violation_type, processing_status, time_of_processing. quality_class: 0 — норма, 1 — нарушение; quality_prob оставлено пустым по выбранной политике. Время — секунды. Несколько нарушений разделены `;`. UID отсутствуют у PNG/JPEG и не выдумываются.

Область — «Поясничный отдел позвоночника» или «Проксимальный отдел бедра». Нарушения: «Некорректная укладка», «Не выровнена ось позвоночника», «Присутствуют посторонние предметы», «Некорректная область интереса» — только применимые к области. Общий класс — ИЛИ.

JSON хранит rows, details, summary. В details: task_decisions, raw_task_decisions, forced_violation_tasks, geometry, metadata, error. index начинается с 0 и определяет имя PNG в ZIP. Поле metadata.spacing имеет порядок [Y,X]. PNG сохранены в исходной пиксельной сетке; физические пропорции бедра при просмотре восстанавливаются по масштабу.

Ошибки: 400 — параметры; 401 — авторизация; 409 — ещё нет выгрузки; 413 — размер; 429 — очередь; 507 — место. Ошибка одного файла сохраняется строкой Failure и не отменяет весь пакет. Для файла, который не декодирован, медицинский класс отсутствует.
