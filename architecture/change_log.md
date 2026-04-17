# Журнал изменений архитектуры

## 2026-04-16

- **Сборка**: пересобран `release/solution.zip` из актуальных `solution.py`, `inference.ipynb`, `requirements.txt`, `predictions.csv`.
- **Обучение K-fold / phase1**: раздельные бюджеты эпох — `EPOCHS_KFOLD_VISC=5000`, `EPOCHS_KFOLD_OXID=1400`, phase1 cap `EPOCHS_PHASE1_MAX=2500`; на окислении в K-fold **early stop** `PATIENCE_KFOLD_OXID`; чекпоинт с `CHECKPOINT_MIN_DELTA_OXID` для гашения дрожи val; ST по вязкости — `LR_ST_VISC` ниже 5e-4.
- **Химические признаки (орг. рекомендации)**: `build_chem_prop_groups` по подстрокам имён из `daimler_component_properties`; глобальные +6 (истощение AO, загуститель×T, загуститель×истощение, каталитический след из Ca/S/B/металл/P/Zn, N по дисперсантам, disp×det); синергии **7 непрерывных** вместо бинарных + `StandardScaler` на синергии; `GLOBAL_DIM=14`.
- **Обучение (устар.)**: ранее единый `EPOCHS`; см. актуальные `EPOCHS_KFOLD_VISC` / `EPOCHS_KFOLD_OXID` / `EPOCHS_PHASE1_MAX` в коде.
- **Качество (без запуска обучения в этой задаче)**: отбор `NUM_PROPS` свойств по корреляции масс.-взвешенных агрегатов с вязкостью/окислением на train (`VISC_PROP_RANK_WEIGHT`); глобальный вектор расширен до **8** (DOT + доли zddp/base_oil/AO/detergent+dispersant); вес **ST в ансамбле вязкости** подбирается по **OOF** (минимум MAE на train); константы и `training_loss_metrics.config` дополняются `PROPS_ORDER` и `visc_oof_blend_w_st`.
- **Сборка**: пересобран `release/solution.zip` из актуальных `solution.py`, `inference.ipynb`, `requirements.txt`, `predictions.csv` без повторного обучения.
- **Артефакты обучения**: по завершении `release/solution.py` пишется `release/training_loss_metrics.json` — конфиг (`EPOCHS`, `K_FOLD`, `BATCH_SIZE`, device), фаза fold0 с early stop (полные списки train/val loss), K-fold по каждому фолду и модели (те же списки + `min_val`, `argmin_val_epoch`).
- **K-fold без утечки таргета**: после перехода на общий `KFold` исправлено использование скейлеров — в цикле по фолдам `RobustScaler`/`StandardScaler` для вязкости и окисления **обучаются только на `tr_idx` фолда** (`scaler_*_f`); предсказание на тесте инвертируется тем же скейлером, что и обучение модели на этом фолде (ранее оставались ссылки на несуществующие глобальные скейлеры).
- **Чекпоинт и early stop**: по умолчанию `use_leaderboard_checkpoint=True` — лучшая эпоха по **val MAE в z-пространстве**; при заданном `patience` ранняя остановка использует ту же метрику.
- **Фаза графиков**: совпадает с **fold0** того же `KFold`, что и K-fold финал; phase-скейлеры таргетов только на train fold0.
- **Качество / устранение ошибок признаков**: убран ordinal `type_enc`/`LabelEncoder`; тип только one-hot; `StandardScaler` только на непрерывный блок компонента; Zn/P только для ZDDP; синергии — взвешенные пары типов (см. выше).
- **Set Transformer**: вместо mean+max после ISAB — **PMA** (один seed, `nn.MultiheadAttention`).
- **Финальное обучение (K-fold)**: на каждом фолде **полные `EPOCHS`** (сейчас 4000), веса с **минимальным val MAE(z)** за эпоху (без early stopping на фолде — малый val даёт шум и ложные остановки); среднее по фолдам на тесте; окисление — `clip` ≥ 0.
- **Максимизация качества (ранее)**: mean+max в Deep Sets; крупные скрытые слои; `BATCH_SIZE` 32 (GPU) / 16 (CPU); константы `EPOCHS`/`PATIENCE` см. в коде.
- **Learning curves**: после K-fold; справа — первый фолд, полный прогон `EPOCHS` с лучшим чекпоинтом по val.
- **Сборка**: пересобран `release/solution.zip` (только `solution.py`, `inference.ipynb`, `requirements.txt`, `predictions.csv`); в `requirements.txt` добавлен `matplotlib` для обучающего скрипта.
- **Свойства компонентов**: убран fallback на произвольную партию с тем же именем компонента; введено явное слияние measured поверх typical для корректной пары `(компонент, партия)`.
- **Пути**: `release/solution.py` читает данные из `REPO_ROOT/data`, пишет `predictions.csv` и графики в каталог `release/`.
- **Обучение**: см. константы `EPOCHS`, `PATIENCE`, `PATIENCE_FOLD`, `K_FOLD` в `release/solution.py`.
- **Воспроизводимость**: `set_seed`, детерминированный CUDNN, генератор DataLoader.
- **Zn/P**: отношение считается по именам признаков (подстроки «цинк», «фосфор»), без жёстких индексов в отсортированном списке свойств.
- **Препроцессинг**: в `src/prepare_data.py` синхронизирована логика `get_component_properties` и базовый путь к `data/` от корня репозитория.
