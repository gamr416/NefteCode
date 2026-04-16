# Журнал изменений архитектуры

## 2026-04-16

- **Сборка**: пересобран `release/solution.zip` из актуальных `solution.py`, `inference.ipynb`, `requirements.txt`, `predictions.csv` без повторного обучения.
- **Артефакты обучения**: по завершении `release/solution.py` пишется `release/training_loss_metrics.json` — конфиг (`EPOCHS`, `K_FOLD`, `BATCH_SIZE`, device), фаза fold0 с early stop (полные списки train/val loss), K-fold по каждому фолду и модели (те же списки + `min_val`, `argmin_val_epoch`).
- **K-fold без утечки таргета**: после перехода на общий `KFold` исправлено использование скейлеров — в цикле по фолдам `RobustScaler`/`StandardScaler` для вязкости и окисления **обучаются только на `tr_idx` фолда** (`scaler_*_f`); предсказание на тесте инвертируется тем же скейлером, что и обучение модели на этом фолде (ранее оставались ссылки на несуществующие глобальные скейлеры).
- **Чекпоинт и early stop**: по умолчанию `use_leaderboard_checkpoint=True` — лучшая эпоха по **val MAE в z-пространстве**; при заданном `patience` ранняя остановка использует ту же метрику.
- **Фаза графиков**: совпадает с **fold0** того же `KFold`, что и K-fold финал; phase-скейлеры таргетов только на train fold0.
- **Качество / устранение ошибок признаков**: убран ordinal `type_enc`/`LabelEncoder`; тип только one-hot; `StandardScaler` только на непрерывный блок компонента; Zn/P только для ZDDP; 7 синергийных флагов.
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
