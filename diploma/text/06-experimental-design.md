# План экспериментального исследования

В данной главе фиксируется экспериментальная схема, необходимая для проверки эффективности adaptive policy и для корректного сравнения предложенного подхода с базовыми и абляционными вариантами. Структура раздела должна заранее определить набор сценариев, чтобы результаты последующей главы были интерпретируемыми и воспроизводимыми. (источник: diploma/docs/narrative.md; README.md; src/pipeline_mvp.py)

## Датасеты и конфигурации

Базовым датасетом для экспериментальной проверки в текущем проекте является VisDrone-подобный YOLO-набор данных, поскольку именно этот сценарий положен в основу MVP-конвейера и напрямую соответствует задаче детекции малых объектов на аэрофотоснимках и UAV-сценах. Конфигурация по умолчанию в `project_config.yaml` ориентирована именно на этот режим и задает структуру split-ов, перечень классов, а также пороги, используемые в анализе и policy generation. [2, 6, 12, 29] (источник: configs/project_config.yaml; README.md; src/pipeline_mvp.py)

Вторым важным сценарием является COCO-small, то есть подмножество COCO 2017, содержащее только объекты с площадью не более `32^2` пикселей. Этот датасет нужен не как замена VisDrone, а как дополнительный источник для проверки переносимости подхода на общую детекционную постановку, где малые объекты выделяются не предметной областью, а процедурой фильтрации аннотаций. [1, 11, 12] (источник: README.md; src/data/coco_small_manager.py; docs/DATASET_ANALYTICS.md)

В документации проекта дополнительно упоминаются DOTA и xView как потенциальные сценарии дальнейшего применения конвейера. На текущем этапе их целесообразно рассматривать как целевые наборы для расширения исследования, а не как обязательную часть минимальной экспериментальной программы, поскольку основная программная логика уже подготовлена к подобному переносу, но полный набор воспроизводимых запусков для них еще не зафиксирован в основной конфигурации проекта. [18, 19, 30] (источник: README.md; diploma/docs/narrative.md; src/pipeline_mvp.py)

С точки зрения конфигурации экспериментов важную роль играет выбор training profile. В конвейере предусмотрены профили `fast`, `final`, `balanced`, `quality`, `hour` и `max_quality`, которые различаются по числу эпох, размеру входного изображения, размеру батча и доле используемых данных. Для первичной проверки и smoke-тестов целесообразно использовать упрощенные профили, тогда как для итогового сравнения подходов требуется применение сопоставимого и более полного режима обучения. (источник: src/pipeline_mvp.py; configs/project_config.yaml)

## Сравниваемые режимы

Центральным элементом экспериментального плана является сравнение нескольких режимов обучения, запускаемых в рамках единого training suite. В базовой схеме проекта предусмотрены режимы `baseline`, `manual`, `adaptive`, `adaptive_no_mosaic` и `adaptive_no_custom_albu`, что позволяет сопоставлять как полный rule-based подход, так и отдельные его компоненты. (источник: README.md; src/training/train_runner.py; docs/AUGMENTATION_POLICY.md)

Таблица 4 - Сравниваемые режимы обучения в экспериментальном плане. (источник: README.md; src/training/train_runner.py; configs/baseline.yaml; configs/manual.yaml)

| Режим | Назначение | Что показывает |
|---|---|---|
| `baseline` | Базовый режим Ultralytics-подобных аугментаций | Контрольный уровень качества без adaptive policy |
| `manual` | Ручная policy для small-object сценария | Качество экспертно подобранной статической настройки |
| `adaptive` | Rule-generated adaptive policy | Эффект интерпретируемого автоматического подбора |
| `adaptive_no_mosaic` | Adaptive policy без mosaic | Вклад mosaic в итоговый результат |
| `adaptive_no_custom_albu` | Adaptive policy без custom Albumentations | Вклад `BBoxAwareCrop` и `BBoxCopyPaste` |

Режим `baseline` использует консервативную Ultralytics-подобную конфигурацию, тогда как `manual` соответствует вручную подобранной политике для small-object сценария. Наиболее важным для задачи дипломной работы является сопоставление этих двух режимов с `adaptive`, поскольку именно оно показывает, дает ли rule-based выбор аугментаций преимущество по сравнению как с общими дефолтами, так и с экспертной статической настройкой. (источник: configs/baseline.yaml; configs/manual.yaml; src/training/train_runner.py)

Абляционные конфигурации `adaptive_no_mosaic` и `adaptive_no_custom_albu` нужны для декомпозиции итогового эффекта adaptive policy. Первая позволяет понять, какую роль в улучшении играет мозаичная аугментация в плотных сценах, а вторая отделяет вклад встроенных scalar-параметров от вклада пользовательских преобразований, реализованных через Python API Albumentations и object bank. (источник: src/training/train_runner.py; docs/AUGMENTATION_POLICY.md; src/augmentation/albumentations_transforms.py)

Если в исследование включается сопоставление с AutoAug-like подходом, его следует оформлять как отдельный бюджетно-ограниченный сценарий. В текущей реализации `src/experiments/autoaug_search.py` генерирует небольшой набор случайных candidate policies, который может использоваться как low-compute comparator, однако такое сравнение необходимо интерпретировать отдельно от основного baseline/manual/adaptive цикла. [3, 21, 23] (источник: src/experiments/autoaug_search.py; README.md)

## Метрики и критерии интерпретации

Основными метриками экспериментального плана должны выступать показатели, чувствительные к малым объектам, прежде всего `AP_small` и `AR_small`. Именно рост этих значений рассматривается в проекте как главный критерий успеха adaptive policy, поскольку целевая задача связана не с общим улучшением любого детектора, а именно с более устойчивым обнаружением малых экземпляров в сложных сценах. (источник: docs/DATASET_ANALYTICS.md; src/evaluation/coco_eval_runner.py; src/evaluation/metrics_report.py)

Наряду с этим необходимо анализировать и общие COCO-метрики `AP@[0.5:0.95]`, `AP50`, `AP75`, а также `AP_medium` и `AP_large`. Такое требование связано с тем, что локальное повышение качества на малых объектах не должно достигаться за счет существенного ухудшения общей детекционной способности модели на остальных масштабах объектов. (источник: src/evaluation/coco_eval_runner.py; src/evaluation/metrics_report.py; docs/DATASET_ANALYTICS.md)

Внутреннее расширение `AP_tiny` должно использоваться как дополнительная диагностическая метрика для сценариев, где доля очень малых объектов особенно высока. При интерпретации результатов необходимо явно указывать, что эта величина не входит в стандартный набор итоговых метрик COCOeval, а потому должна рассматриваться как проектный индикатор чувствительности к объектам площадью до `16^2` пикселей. (источник: docs/THRESHOLDS.md; src/evaluation/coco_eval_runner.py)

С практической точки зрения критерии интерпретации можно сформулировать следующим образом: adaptive policy считается успешной, если она улучшает `AP_small` относительно `baseline` и `manual`, не вызывает резкого ухудшения общей mAP и при этом сохраняет интерпретируемость через `decision_report.json`. Для ablation-сценариев основным вопросом становится вклад каждого отключаемого компонента в итоговый прирост качества. (источник: docs/AUGMENTATION_POLICY.md; src/policy/rule_engine.py; src/evaluation/metrics_report.py)

## Угрозы валидности

Одной из ключевых угроз валидности является ограниченность вычислительного бюджета. Поскольку training suite включает несколько режимов, а также потенциально допускает multi-seed и AutoAug-like сравнения, слишком короткие профили обучения могут исказить относительное качество режимов, тогда как слишком тяжелые профили затрудняют повторяемость экспериментов и усложняют практическое использование конвейера. (источник: src/pipeline_mvp.py; src/training/train_runner.py; README.md)

Вторая существенная угроза связана с качеством разметки и с различиями между train и val распределениями. Если валидационная выборка заметно отличается по доле малых объектов, плотности сцен или освещенности, то изменение `AP_small` может отражать не только эффект adaptive policy, но и структуру самого датасета. Именно поэтому документация проекта рекомендует анализировать train-val shift до интерпретации итоговых метрик. (источник: docs/DATASET_ANALYTICS.md; src/analysis/dataset_analyzer.py)

Третья угроза относится к стохастичности обучения и особенностям интеграции пользовательских аугментаций. Даже при фиксированном seed часть разброса метрик может определяться внутренними механизмами тренировочного пайплайна, а несовместимость конкретной версии Ultralytics с Python API `augmentations` способна привести к скрытому изменению экспериментального режима. В реализации проекта эта проблема частично смягчается контролем `require_custom_augmentations` и возможностью многоразового запуска по нескольким seeds. (источник: src/training/train_runner.py; configs/project_config.yaml; README.md)

Наконец, отдельной угрозой является переносимость результатов на новые датасеты и предметные области. Даже если adaptive policy показывает преимущество на VisDrone или COCO-small, это еще не гарантирует аналогичный выигрыш на DOTA, xView или иных overhead-сценах без дополнительной калибровки порогов и без повторного анализа статистик конкретной выборки. (источник: docs/THRESHOLDS.md; README.md; src/experiments/autoaug_search.py)

## Источники раздела

- `[1]` COCO: Common Objects in Context. Использован для описания категории small objects и COCO-совместимой оценки. URL: https://arxiv.org/abs/1405.0312
- `[2]` Vision Meets Drones: A Challenge. Использован для описания UAV-сценария и прикладной постановки VisDrone. URL: https://arxiv.org/abs/1804.07437
- `[3]` AutoAugment: Learning Augmentation Policies from Data. Использован для описания search-based подходов к выбору аугментаций. URL: https://arxiv.org/abs/1805.09501
- `[6]` Ultralytics VisDrone Dataset Guide. Использован для описания практического сценария VisDrone. URL: https://docs.ultralytics.com/datasets/detect/visdrone/
- `[11]` COCO - Common Objects in Context. Использован как официальный источник сведений о датасете COCO. URL: https://cocodataset.org/index.htm
- `[12]` You Only Look Once: Unified, Real-Time Object Detection. Использован для описания YOLO-совместимого обучающего контура. URL: https://arxiv.org/abs/1506.02640
- `[18]` DOTA: A Large-Scale Dataset for Object Detection in Aerial Images. Использован для описания возможного сценария переноса на aerial imagery. URL: https://doi.org/10.1109/CVPR.2018.00418
- `[19]` xView: Objects in Context in Overhead Imagery. Использован для описания возможного сценария переноса на overhead imagery. URL: https://arxiv.org/abs/1802.07856
- `[21]` RandAugment: Practical Automated Data Augmentation with a Reduced Search Space. Использован для описания упрощенных search-based подходов. URL: https://arxiv.org/abs/1909.13719
- `[23]` Faster AutoAugment: Learning Augmentation Strategies Using Backpropagation. Использован для описания более эффективных процедур поиска augmentation policy. URL: https://arxiv.org/abs/1911.06987
- `[29]` VISDRONE. Использован как официальный источник сведений о датасете и challenge-сценарии. URL: https://aiskyeye.com/
- `[30]` DOTA. Использован как официальный источник сведений о датасете DOTA. URL: https://captain-whu.github.io/DOTA/
- `diploma/docs/narrative.md`. Использован для согласования структуры экспериментальной главы. (источник: diploma/docs/narrative.md)
- `README.md`. Использован для описания прикладных сценариев и сравниваемых режимов. (источник: README.md)
- `docs/DATASET_ANALYTICS.md`. Использован для описания статистик и метрик, важных для анализа малых объектов. (источник: docs/DATASET_ANALYTICS.md)
- `docs/AUGMENTATION_POLICY.md`. Использован для описания сравниваемых конфигураций policy и абляционных сценариев. (источник: docs/AUGMENTATION_POLICY.md)
- `docs/THRESHOLDS.md`. Использован для согласования порогов и критериев интерпретации. (источник: docs/THRESHOLDS.md)
- `configs/project_config.yaml`. Использован для подтверждения датасетных и тренировочных конфигураций по умолчанию. (источник: configs/project_config.yaml)
- `configs/baseline.yaml`. Использован для описания контрольного baseline-режима. (источник: configs/baseline.yaml)
- `configs/manual.yaml`. Использован для описания вручную заданной static policy. (источник: configs/manual.yaml)
- `src/pipeline_mvp.py`. Использован для подтверждения общей схемы запуска экспериментов. (источник: src/pipeline_mvp.py)
- `src/data/coco_small_manager.py`. Использован для подтверждения сценария COCO-small. (источник: src/data/coco_small_manager.py)
- `src/analysis/dataset_analyzer.py`. Использован для подтверждения значимости анализа train-val структуры и статистик малых объектов. (источник: src/analysis/dataset_analyzer.py)
- `src/policy/rule_engine.py`. Использован для подтверждения роли `decision_report.json` в интерпретации результатов. (источник: src/policy/rule_engine.py)
- `src/augmentation/albumentations_transforms.py`. Использован для подтверждения того, какие custom-компоненты участвуют в ablation-анализе. (источник: src/augmentation/albumentations_transforms.py)
- `src/training/train_runner.py`. Использован для подтверждения состава training suite и multi-seed сценариев. (источник: src/training/train_runner.py)
- `src/evaluation/coco_eval_runner.py`. Использован для подтверждения набора итоговых метрик. (источник: src/evaluation/coco_eval_runner.py)
- `src/evaluation/metrics_report.py`. Использован для подтверждения состава финального markdown-отчета по экспериментам. (источник: src/evaluation/metrics_report.py)
- `src/experiments/autoaug_search.py`. Использован для подтверждения budget-aware AutoAug-like сценария сравнения. (источник: src/experiments/autoaug_search.py)
- `src/experiments/summary.py`. Использован для подтверждения средств агрегации repeated-run метрик. (источник: src/experiments/summary.py)
