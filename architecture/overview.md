# Архитектура решения NefteCode 2026

## Поток данных

1. **Вход**: `data/daimler_mixtures_{train,test}.csv`, `data/daimler_component_properties.csv` (путь от корня репозитория; скрипт `release/solution.py` резолвит `REPO_ROOT/data`).

2. **Свойства компонентов**: словарь измеренных пар `(компонент, партия)` и словарь типичных значений по компоненту. Для строки рецептуры: если есть измерения для пары `(comp, batch)`, они **сливаются поверх** типичных значений; при отсутствии пары используются только типичные. Подстановка свойств с **другой партии** того же компонента не выполняется.

3. **Признаки сценария**: глобальные условия DOT (4 числа); на компонент — **масса, log(масса)**, до `NUM_PROPS` числовых свойств, **Zn/P только для типа zddp** (иначе 0), **one-hot типа** (12 классов по эвристике имени). **Без** ordinal `LabelEncoder` по имени компонента. Множество компонентов дополняется нулями до `MAX_COMPONENTS=50`.

4. **Нормализация**: `StandardScaler` на глобальные 4 признака; на компонентный тензор — **только первые `N_CONT_PER_COMP` непрерывных** столбцов (масса, log, свойства, p_zn); **one-hot не масштабируются**. Синергийные **7** бинарных флагов на сценарий не входят в `scaler_c`.

5. **Модели**: `DeepSetsModel` (encoder: phi + **mean и max** по маске) и `SetTransformer` (ISAB-блоки + **PMA** — один learnable seed, `MultiheadAttention` к компонентам с `key_padding_mask`). Вязкость: **`RobustScaler`** + **`SmoothL1Loss`**; окисление: **`StandardScaler`** + MSE.

6. **Обучение**: фаза графиков learning curve — **тот же `KFold`, что и финал**: train/val = **fold0** (`fold0_tr` / `fold0_va`); таргет-скейлеры (`RobustScaler` вязкость, `StandardScaler` окисление) для этой фазы **обучаются только на train fold0**. **Early stopping** (`PATIENCE`) только здесь. **Финал (K-fold)**: на каждом фолде таргет-скейлеры **заново fit только на train этого фолда**; обучение **все `EPOCHS` эпох** без early stop на фолде; val-фолд для **лучшего чекпоинта по прокси лидерборда — средний `|pred_z − y_z|` (MAE в z)** на валидации, совпадающий с `scaler_y` фолда. LR: линейный warmup (`WARMUP_FRAC`) + косинус до `eta_min`. `BATCH_SIZE` 32 на GPU, 16 на CPU. Предсказания на тесте **усредняются по фолдам**. Итог: вязкость `0.6 * mean(ST) + 0.4 * mean(DS)`, окисление — среднее ST и DS; окисление **clip ≥ 0**.

7. **Выход**: `release/predictions.csv`, `release/learning_curves_v2.png` (сетка 4×2: слева фаза с отложенной выборкой, справа кривые первого фолда K-fold). История лоссов (train/val по эпохам) и краткая сводка — в `release/training_loss_metrics.json` после полного прогона `main()`.

## Воспроизводимость

`set_seed`, детерминированный CUDNN, генератор `DataLoader`; K-fold с фиксированным `random_state`.

## Сабмит

Каталог `release/`: `solution.py`, `inference.ipynb`, `requirements.txt`, `predictions.csv`, `solution.zip`.
