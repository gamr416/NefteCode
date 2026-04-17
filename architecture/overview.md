# Архитектура решения NefteCode 2026

## Поток данных

1. **Вход**: `data/daimler_mixtures_{train,test}.csv`, `data/daimler_component_properties.csv` (путь от корня репозитория; скрипт `release/solution.py` резолвит `REPO_ROOT/data`).

2. **Свойства компонентов**: словарь измеренных пар `(компонент, партия)` и словарь типичных значений по компоненту. Для строки рецептуры: если есть измерения для пары `(comp, batch)`, они **сливаются поверх** типичных значений; при отсутствии пары используются только типичные. Подстановка свойств с **другой партии** того же компонента не выполняется.

3. **Признаки сценария**: глобальные **`GLOBAL_DIM=14`** чисел: условия DOT (4) + **доли массы** (zddp, base_oil, сумма AO, detergent+dispersant) + **химия сценария** (6): прокси **истощения защиты** (TBN у детергентов + P/Zn у ZDDP vs агрессия T×t), **деструкция загустителя** (доля загустителя × T), **сшивание загустителя** (доля × истощение, без таргета окисления), **каталитический след** (масс.-взвешенная сумма показателей Ca, S, B, металл Ca/Mg, P, Zn по подстрокам имён), **азот в дисперсантах** (масс.-взвешенное по строкам dispersant), **dispersant×detergent** (произведение долей). На компонент — **масса, log(масса)**, до `NUM_PROPS` свойств (ранжирование по корреляции на train, `VISC_PROP_RANK_WEIGHT`), **Zn/P для zddp**, **one-hot** (12). Множество компонентов до `MAX_COMPONENTS=50`.

4. **Нормализация**: `StandardScaler` на все **глобальные** признаки; на компонентный тензор — **только** непрерывный блок (`N_CONT_PER_COMP`); **one-hot не масштабируются**. Вектор **синергий 7** — непрерывные `sqrt(frac_i·frac_j)` по парам типов; отдельный **`StandardScaler` на всю матрицу синергий** train→test (не входит в `scaler_c`).

5. **Модели**: `DeepSetsModel` (encoder: phi + **mean и max** по маске) и `SetTransformer` (ISAB-блоки + **PMA** — один learnable seed, `MultiheadAttention` к компонентам с `key_padding_mask`). Вязкость: **`RobustScaler`** + **`SmoothL1Loss`**; окисление: **`StandardScaler`** + MSE.

6. **Обучение**: фаза графиков — **fold0** того же `KFold`, верхняя граница эпох **`EPOCHS_PHASE1_MAX`**, early stop по **`PATIENCE`**. **K-fold**: вязкость — до **`EPOCHS_KFOLD_VISC`** без early stop (чекпоинт по val MAE в z, опционально `CHECKPOINT_MIN_DELTA_VISC`); окисление — до **`EPOCHS_KFOLD_OXID`** с **early stop** `PATIENCE_KFOLD_OXID` и **`CHECKPOINT_MIN_DELTA_OXID`** для чекпоинта. ST по вязкости — пониженный **`LR_ST_VISC`**. LR: warmup + косинус. Предсказания на тесте — усреднение по фолдам; вязкость — OOF-подбор `w_ST`; окисление **clip ≥ 0**.

7. **Выход**: `release/predictions.csv`, `release/learning_curves_v2.png` (сетка 4×2: слева фаза с отложенной выборкой, справа кривые первого фолда K-fold). История лоссов (train/val по эпохам) и краткая сводка — в `release/training_loss_metrics.json` после полного прогона `main()`.

## Воспроизводимость

`set_seed`, детерминированный CUDNN, генератор `DataLoader`; K-fold с фиксированным `random_state`.

## Сабмит

Каталог `release/`: `solution.py`, `inference.ipynb`, `requirements.txt`, `predictions.csv`, `solution.zip`.
