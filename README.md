# E-CUP 2026 — Матчинг товаров — команда «uralmash mentality»

**Итог: 6 место. Public LB 0.5414129346430998** (`submission_v46.zip`).
Выбранные в финал сабмиты: **v46** и **v45** (0.5331887134560345).

Метрика — macro PR-AUC по 20 категориям. Контейнер запускается на H100 80GB,
точка входа `python -u run.py`; бюджет **на каждый этап**: 60 с на check
(≤10k пар), 360 с на public (~115k пар), 780 с на private (~275k пар).

---

## Что это за репозиторий

Это **отдельный репозиторий для проверки решения**: только то, из чего собран
финальный сабмит, и то, чем его можно воспроизвести. Рабочий репозиторий
команды содержит ~150 обучающих рецептов и ~410 экспериментов; сюда вынесены
те 4 компонента, которые реально поехали на борд, плюс полный журнал
экспериментов как отдельный файл.

| файл / папка | что это |
|---|---|
| `inference/` | **точный код контейнера**, который дал 0.54141 — побайтово сверен с `submission_v46.zip` |
| `inference/models/gbdt/` | обученный GBDT (v8), 286 МБ — единственные веса, лежащие в git |
| `inference/models/ce-{1,2,3}/` | `config.json` / `tokenizer_config.json` каждого чекпоинта: это решения, а не веса |
| `training/` | рецепты обучения обоих наших кросс-энкодеров и GBDT |
| `evaluation/` | инструменты локальной валидации, в т.ч. `cascade_read.py` |
| `docs/SOLUTION.md` | подробное описание решения и почему оно такое |
| `docs/REPRODUCE.md` | пошаговое воспроизведение |
| `docs/WEIGHTS.md` | манифест весов с sha256, прочитанный из поданного архива (2.96 ГБ, отдельно) |
| `docs/EXPERIMENTS.md` | что проверялось и что не сработало |
| `docs/LEDGER.csv` | полный журнал: 411 экспериментов с заранее зафиксированными критериями приёмки |

Веса кросс-энкодеров (2.96 ГБ, семь больших файлов) в git не лежат — их размеры
и sha256 в `docs/WEIGHTS.md`, прочитанные из самого поданного архива.

**Сборка проверена, а не заявлена.** `inference/build_submission.py` собирает
архив из этого репозитория плюс веса, и с флагом `--verify` сверяет результат с
`submission_v46.zip`: **все 47 записей совпадают по CRC**, размер тот же 2.91 ГБ.
Различаются только timestamp'ы внутри zip.

---

## Решение в одном экране

Каскад из трёх кросс-энкодеров плюс градиентный бустинг, смешиваемые **в
пространстве рангов**:

```
GBDT (v8, LightGBM, ~129 признаков)        — все пары
CE-1  t176full-ep1   mmBERT-base @1024     — все пары
CE-2  t120-pw0.134   bge-reranker-v2-m3 @256 — топ-30% ранжирования CE-1
CE-3  alexbge        bge-reranker-v2-m3 @256 — топ-30% ранжирования CE-1+CE-2

ce  = 0.70·rank01(CE-1) + 0.30·band_rank(CE-2)
ce  = 0.75·ce           + 0.25·band_rank(CE-3)
out = 0.10·rank(GBDT)   + 0.90·rank(ce)
```

Текст для кросс-энкодера одинаков при обучении и инференсе:
`name [SEP] category [SEP] attributes[:2000]` с обеих сторон пары.

**Каскад — не оптимизация скорости, а основной источник качества.** Второй
кросс-энкодер даёт +0.01434 внутри каскада, третий — +0.00822 на борде, причём
третий *слабее* второго поодиночке (0.50727 против 0.51373). Работает
декорреляция, а не сила отдельной модели.

---

## Три вещи, которые стоит знать до чтения кода

1. **CE-1 лежит в архиве по пути `models/ce-e5-base/`.** Это исторический
   артефакт: в v12 там был e5-base, и слот не переименовывали, потому что это
   была бы вторая переменная в сабмите, где менялась ровно одна.

2. **`FORCE_CE=1` (значение по умолчанию) обнуляет `deadline_ts`**, из-за чего
   все внутриконтейнерные проверки «пропустить этап, если мало времени»
   становятся недействующими. Это осознанно: на борде они срабатывали ложно и
   выключали кросс-энкодеры. Подробности и цена ошибки — в `docs/SOLUTION.md`.

3. **Локальная валидация читается только «каскадно».** Оценка чекпоинта в
   одиночку и она же внутри каскада разошлись в 34 раза на решении, которое
   дало главный прирост дня. `evaluation/cascade_read.py` переносит арифметику
   смешивания дословно из `run.py`.

---

# English summary

**6th place. Public LB 0.5414129346430998** (`submission_v46.zip`); the two
selected submissions were v46 and v45 (0.5331887134560345).

This is a **review repository**: the exact inference container that produced the
score, the training recipes for the components inside it, the local-validation
tooling, and the full 411-row experiment ledger. The team's working repo holds
~150 training recipes; only 4 components shipped, and those are what is here.

The solution is a three-stage cross-encoder cascade rank-blended with a
LightGBM stack — see the diagram above and `docs/SOLUTION.md` for the full
description, `docs/REPRODUCE.md` for step-by-step reproduction, and
`docs/EXPERIMENTS.md` for what was tried and rejected.

Documentation under `docs/` is in English, matching the source comments.
