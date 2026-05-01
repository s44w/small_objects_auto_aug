# Рабочая папка ВКР

Эта папка подготовлена как рабочий каркас для написания ВКР по текущему проекту
`small_objects_auto_aug`.

Главный свод требований к написанию текста, использованию источников, оформлению рисунков,
таблиц, формул, структуре разделов и сборке `Markdown -> Word` находится в файле
`docs/master-writing-rules.md`. При любом новом этапе работы над дипломом рекомендуется
начинать именно с него, а затем использовать остальные файлы `docs/` как специализированные
дополнения. (источник: diploma/docs/master-writing-rules.md)

## Состав каталога

Внутри уже есть:

- `docs/` - правила оформления, narrative, глоссарий, список источников.
- `text/` - заготовки глав в Markdown.
- `template/blank-template.docx` - шаблон Word для последующей сборки через Pandoc.
- `scripts/` - скрипты сборки и постобработки `.docx`.
- `source-materials/` - места для примеров, PDF с требованиями и черновых материалов.

## Соответствие структуре проекта

Предварительное соответствие структуре проекта:

- `text/04-architecture.md` опирается на `src/pipeline_mvp.py` и модульную структуру `src/`.
- `text/05-implementation.md` опирается на код в `src/data`, `src/analysis`, `src/policy`,
  `src/augmentation`, `src/training`, `src/evaluation`, `src/experiments`, `src/utils`.
- `text/06-experimental-design.md` и `text/07-experiments.md` опираются на `configs/`,
  `tests/`, `docs/` и ноутбуки `notebooks/`.

## Куда складывать материалы

Когда будут готовы примеры `.docx`, их удобно положить в:

- `source-materials/examples/`
- `source-materials/formatting-rules/`

## Сборка документа

Быстрая сборка после наполнения текста:

```bash
bash diploma/scripts/build-docx.sh
```
