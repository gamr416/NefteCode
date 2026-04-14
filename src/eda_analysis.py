import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("=" * 60)
print("1.1 ЗАГРУЗКА И ОСМОТР ДАННЫХ")
print("=" * 60)

train = pd.read_csv('daimler_mixtures_train.csv')
test = pd.read_csv('daimler_mixtures_test.csv')
props = pd.read_csv('daimler_component_properties.csv')

print("\n--- TRAIN ---")
print(f"Shape: {train.shape}")
print(train.info())
print("\n", train.head(10))

print("\n--- TEST ---")
print(f"Shape: {test.shape}")
print(test.info())
print("\n", test.head(10))

print("\n--- PROPERTIES ---")
print(f"Shape: {props.shape}")
print(props.info())
print("\n", props.head(10))

print("\n" + "=" * 60)
print("1.2 ПРОВЕРКА ПРОПУСКОВ")
print("=" * 60)

print("\n--- TRAIN пропуски ---")
print(train.isnull().sum())

print("\n--- TEST пропуски ---")
print(test.isnull().sum())

print("\n--- PROPERTIES пропуски ---")
print(props.isnull().sum())

print("\n" + "=" * 60)
print("1.3 АНАЛИЗ ЦЕЛЕВЫХ ПЕРЕМЕННЫХ")
print("=" * 60)

target1 = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
target2 = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"

print(f"\n--- {target1} ---")
print(train[target1].describe())

print(f"\n--- {target2} ---")
print(train[target2].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train[target1], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title(target1)
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

axes[1].hist(train[target2], bins=50, edgecolor='black', alpha=0.7)
axes[1].set_title(target2)
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('eda_target_distribution.png', dpi=100)
plt.close()

fig, axes = plt.subplots(1, 2)
axes[0].boxplot(train[target1].dropna())
axes[0].set_title(f'{target1}\nBoxplot')
axes[1].boxplot(train[target2].dropna())
axes[1].set_title(f'{target2}\nBoxplot')
plt.tight_layout()
plt.savefig('eda_target_boxplot.png', dpi=100)
plt.close()

print("\n" + "=" * 60)
print("1.4 АНАЛИЗ ЧИСЛА КОМПОНЕНТОВ")
print("=" * 60)

components_per_scenario = train.groupby('scenario_id').size()
print("\n--- Число компонентов на сценарий (TRAIN) ---")
print(components_per_scenario.describe())

components_per_scenario_test = test.groupby('scenario_id').size()
print("\n--- Число компонентов на сценарий (TEST) ---")
print(components_per_scenario_test.describe())

print("\n--- Уникальные значения числа компонентов ---")
print("TRAIN:", sorted(components_per_scenario.unique()))
print("TEST:", sorted(components_per_scenario_test.unique()))

print("\n" + "=" * 60)
print("1.5 ПРОВЕРКА ДУБЛИКАТОВ")
print("=" * 60)

print(f"\n--- TRAIN дубликаты: {train.duplicated().sum()} ---")
print(f"--- TEST дубликаты: {test.duplicated().sum()} ---")
print(f"--- PROPS дубликаты: {props.duplicated().sum()} ---")

print("\n" + "=" * 60)
print("1.6 АНАЛИЗ ПАРАМЕТРОВ ТЕСТА")
print("=" * 60)

temp_col = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
time_col = "Время испытания | - Daimler Oxidation Test (DOT), ч"
bio_col = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
cat_col = "Дозировка катализатора, категория"

print("\n--- Температура ---")
print("TRAIN:", sorted(train[temp_col].unique()))
print("TEST:", sorted(test[temp_col].unique()))

print("\n--- Время ---")
print("TRAIN:", sorted(train[time_col].unique()))
print("TEST:", sorted(test[time_col].unique()))

print("\n--- Биотопливо ---")
print("TRAIN:", sorted(train[bio_col].unique()))
print("TEST:", sorted(test[bio_col].unique()))

print("\n--- Катализатор ---")
print("TRAIN:", sorted(train[cat_col].unique()))
print("TEST:", sorted(test[cat_col].unique()))

print("\n" + "=" * 60)
print("1.7 АНАЛИЗ ТИПОВ КОМПОНЕНТОВ")
print("=" * 60)

print("\n--- Уникальные компоненты (TRAIN) ---")
train_components = train['Компонент'].unique()
print(f"Всего: {len(train_components)}")
for c in sorted(train_components):
    print(f"  {c}")

print("\n--- Уникальные компоненты (TEST) ---")
test_components = test['Компонент'].unique()
print(f"Всего: {len(test_components)}")
for c in sorted(test_components):
    print(f"  {c}")

print("\n--- Новые компоненты в TEST ---")
new_in_test = set(test_components) - set(train_components)
print(f"Новые: {new_in_test}")

print("\n" + "=" * 60)
print("1.8 АНАЛИЗ СВОЙСТВ КОМПОНЕНТОВ")
print("=" * 60)

unique_props = props['Наименование показателя'].dropna().unique()
print(f"\n--- Всего уникальных свойств: {len(unique_props)} ---")
for p in unique_props[:30]:
    print(f"  {p}")
print("  ...")

unique_batches = props['Наименование партии'].dropna().unique()
print(f"\n--- Всего уникальных партий: {len(unique_batches)} ---")

typical_exists = 'typical' in [str(x).lower() for x in unique_batches]
print(f"\n--- Есть 'typical' партия: {typical_exists} ---")

print("\n" + "=" * 60)
print("1.9 СОПОСТАВЛЕНИЕ КОМПОНЕНТОВ И СВОЙСТВ")
print("=" * 60)

train_comp_batch = train[['Компонент', 'Наименование партии']].drop_duplicates()
test_comp_batch = test[['Компонент', 'Наименование партии']].drop_duplicates()

props_comp_batch = props[['Компонент', 'Наименование партии']].drop_duplicates()

train_in_props = train_comp_batch.merge(props_comp_batch, on=['Компонент', 'Наименование партии'], how='left')
test_in_props = test_comp_batch.merge(props_comp_batch, on=['Компонент', 'Наименование партии'], how='left')

print(f"\n--- TRAIN: строк с найденными свойствами ---")
print(f"  Всего: {len(train_comp_batch)}")
print(f"  Найдено: {train_in_props.notna().any(axis=1).sum()}")
print(f"  Не найдено: {train_in_props.isna().any(axis=1).sum()}")

print(f"\n--- TEST: строк с найденными свойствами ---")
print(f"  Всего: {len(test_comp_batch)}")
print(f"  Найдено: {test_in_props.notna().any(axis=1).sum()}")
print(f"  Не найдено: {test_in_props.isna().any(axis=1).sum()}")

print("\n--- PARTS без свойств (TRAIN) ---")
missing_props = train_in_props[train_in_props.isna().any(axis=1)]
print(missing_props.drop_duplicates())

print("\n--- PARTS без свойств (TEST) ---")
missing_props_test = test_in_props[test_in_props.isna().any(axis=1)]
print(missing_props_test.drop_duplicates())

print("\n" + "=" * 60)
print("1.10 АНАЛИЗ 'TYPICAL' ЗНАЧЕНИЙ")
print("=" * 60)

typical_props = props[props['Наименование партии'].astype(str).str.lower() == 'typical']
print(f"\n--- Строк с typical: {len(typical_props)} ---")
if len(typical_props) > 0:
    print(typical_props.head(20))

print("\n" + "=" * 60)
print("1.11 ПРОВЕРКА КОНСИСТЕНТНОСТИ")
print("=" * 60)

train_scenarios = set(train['scenario_id'].unique())
test_scenarios = set(test['scenario_id'].unique())

print(f"\n--- Сценариев TRAIN: {len(train_scenarios)} ---")
print(f"--- Сценариев TEST: {len(test_scenarios)} ---")

test_temp = test.groupby('scenario_id')[temp_col].first()
train_temp = train.groupby('scenario_id')[temp_col].first()
print(f"\n--- TEMP в TEST, отсутствующие в TRAIN ---")
print(test_temp[~test_temp.isin(train_temp.unique())])

print("\n" + "=" * 60)
print("1.12 СОХРАНЕНИЕ ПРОМЕЖУТОЧНЫХ ДАННЫХ")
print("=" * 60)

components_per_scenario.to_csv('components_per_scenario_train.csv')
components_per_scenario_test.to_csv('components_per_scenario_test.csv')

props_pivot = props.pivot_table(
    index=['Компонент', 'Наименование партии'],
    columns='Наименование показателя',
    values='Значение показателя',
    aggfunc='first'
).reset_index()

props_pivot.to_csv('component_properties_wide.csv', index=False)
print("Сохранено: components_per_scenario_train.csv")
print("Сохранено: components_per_scenario_test.csv")
print("Сохранено: component_properties_wide.csv")

print("\n" + "=" * 60)
print("ИТОГИ EDA")
print("=" * 60)

print(f"""
Сводка:
- TRAIN: {train.shape[0]} строк, {len(train_scenarios)} сценариев
- TEST: {test.shape[0]} строк, {len(test_scenarios)} сценариев
- PROPS: {props.shape[0]} строк свойств

Целевые переменные:
- Delta Viscosity: mean={train[target1].mean():.2f}, std={train[target1].std():.2f}
- Oxidation: mean={train[target2].mean():.2f}, std={train[target2].std():.2f}

Число компонентов на сценарий:
- TRAIN: {components_per_scenario.min()}-{components_per_scenario.max()}, median={components_per_scenario.median()}
- TEST: {components_per_scenario_test.min()}-{components_per_scenario_test.max()}, median={components_per_scenario_test.median()}

Новые компоненты в TEST: {new_in_test}
""")

print("\nEDA завершен!")
