import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

DATA_PATH = 'data/'

# Load data
train = pd.read_csv(os.path.join(DATA_PATH, 'daimler_mixtures_train.csv'))
test = pd.read_csv(os.path.join(DATA_PATH, 'daimler_mixtures_test.csv'))
props = pd.read_csv(os.path.join(DATA_PATH, 'daimler_component_properties.csv'))

print(f"Loaded: Train: {train.shape}, Test: {test.shape}, Props: {props.shape}")

# Define columns
TARGET_VISC = 'Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %'
TARGET_OXID = 'Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm'
TEMP_COL = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
TIME_COL = "Время испытания | - Daimler Oxidation Test (DOT), ч"
BIO_COL = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
CAT_COL = "Дозировка катализатора, категория"
COMP_COL = 'Компонент'
BATCH_COL = 'Наименование партии'
MASS_COL = 'Массовая доля, %'
SCENARIO_COL = 'scenario_id'

# Create properties dictionary
def create_properties_dict(props_df):
    props_dict = {}
    typical = props_df[props_df[BATCH_COL].astype(str).str.lower() == 'typical']
    typical_dict = {}
    
    for _, row in typical.iterrows():
        comp = row[COMP_COL]
        prop_name = row['Наименование показателя']
        prop_value = row['Значение показателя']
        if pd.notna(prop_value):
            if comp not in typical_dict:
                typical_dict[comp] = {}
            typical_dict[comp][prop_name] = prop_value
    
    for _, row in props_df.iterrows():
        comp = row[COMP_COL]
        batch = row[BATCH_COL]
        prop_name = row['Наименование показателя']
        prop_value = row['Значение показателя']
        
        if pd.isna(batch) or pd.isna(prop_value):
            continue
            
        key = (comp, batch)
        if key not in props_dict:
            props_dict[key] = {}
        props_dict[key][prop_name] = prop_value
    
    return props_dict, typical_dict

props_dict, typical_dict = create_properties_dict(props)
print(f"Properties: measured={len(props_dict)}, typical={len(typical_dict)}")

# Get component properties
def get_component_properties(comp, batch, props_dict, typical_dict):
    key = (comp, batch)
    if key in props_dict:
        return props_dict[key]
    for k, v in props_dict.items():
        if k[0] == comp:
            return v
    if comp in typical_dict:
        return typical_dict[comp]
    return {}

# Collect numeric properties
all_props = set()
for d in list(props_dict.values()) + list(typical_dict.values()):
    all_props.update(d.keys())

all_props = sorted([p for p in all_props if pd.notna(p)])

numeric_props = []
for p in all_props:
    values = []
    for d in list(props_dict.values()) + list(typical_dict.values()):
        if p in d:
            try:
                val = str(d[p]).replace(',', '.')
                values.append(float(val))
            except Exception:
                pass
    if len(values) > 10:
        numeric_props.append(p)

print(f"Numeric properties: {len(numeric_props)}")

PROPS_ORDER = numeric_props[:20]
print(f"UsingTop20: {PROPS_ORDER}")

# Helper functions
def safe_float(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x).replace(',', '.'))
    except Exception:
        return np.nan

def prepare_scenario_features(scenario_df, scenario_id, props_dict, typical_dict, numeric_props):
    scenario_data = scenario_df[scenario_df[SCENARIO_COL] == scenario_id]
    test_params = scenario_data.iloc[0]
    
    features = {
        'temp': safe_float(test_params[TEMP_COL]),
        'time': safe_float(test_params[TIME_COL]),
        'bio': safe_float(test_params[BIO_COL]),
        'catalyst': safe_float(test_params[CAT_COL]),
    }
    
    components = []
    masses = []
    
    for _, row in scenario_data.iterrows():
        comp = row[COMP_COL]
        batch = row[BATCH_COL] if pd.notna(row[BATCH_COL]) else ''
        mass = safe_float(row[MASS_COL])
        comp_props = get_component_properties(comp, batch, props_dict, typical_dict)
        
        prop_features = {}
        for prop_name in numeric_props:
            if prop_name in comp_props:
                prop_features[prop_name] = safe_float(comp_props[prop_name])
            else:
                prop_features[prop_name] = np.nan
        
        components.append({
            'type': comp,
            'mass': mass,
            'properties': prop_features
        })
        masses.append(mass)
    
    features['components'] = components
    features['masses'] = masses
    features['n_components'] = len(components)
    
    masses_arr = np.array(masses)
    features['mass_mean'] = np.nanmean(masses_arr)
    features['mass_std'] = np.nanstd(masses_arr)
    features['mass_max'] = np.nanmax(masses_arr)
    features['mass_min'] = np.nanmin(masses_arr)
    
    return features

# Encode component types
all_component_types = set(train[COMP_COL].unique()) | set(test[COMP_COL].unique())
le = LabelEncoder()
le.fit(list(all_component_types))
print(f"Component types: {len(le.classes_)}")

def encode_component_type(comp_type, le):
    if comp_type in le.classes_:
        return le.transform([comp_type])[0]
    return -1

# Create scenario vectors
def create_scenario_vector(data_item, le, props_order):
    f = data_item['features']
    
    global_features = [
        f['temp'], f['time'], f['bio'], f['catalyst'],
        f['n_components'],
        f['mass_mean'], f['mass_std'], f['mass_max'], f['mass_min']
    ]
    
    comp_vectors = []
    for comp in f['components']:
        type_enc = encode_component_type(comp['type'], le)
        mass = comp['mass'] if not np.isnan(comp['mass']) else 0
        prop_vals = []
        for p in props_order:
            pv = comp['properties'].get(p, np.nan)
            prop_vals.append(pv if not np.isnan(pv) else 0)
        comp_vectors.append([type_enc, mass] + prop_vals)
    
    comp_arr = np.array(comp_vectors)
    if len(comp_arr) > 0:
        pooled_mean = np.nanmean(comp_arr, axis=0)
        pooled_max = np.nanmax(comp_arr, axis=0)
        pooled_min = np.nanmin(comp_arr, axis=0)
        pooled = np.concatenate([pooled_mean, pooled_max, pooled_min])
    else:
        pooled = np.zeros(3 * (len(props_order) + 2))
    
    return np.concatenate([global_features, pooled])

# Prepare train data
print("Preparing train data...")
train_scenarios = train[SCENARIO_COL].unique()
train_data = []

for sid in train_scenarios:
    features = prepare_scenario_features(train, sid, props_dict, typical_dict, PROPS_ORDER)
    scenario_rows = train[train[SCENARIO_COL] == sid]
    train_data.append({
        'scenario_id': sid,
        'features': features,
        'target_visc': scenario_rows[TARGET_VISC].iloc[0],
        'target_oxid': scenario_rows[TARGET_OXID].iloc[0]
    })

print(f"Train scenarios: {len(train_data)}")

# Prepare test data
print("Preparing test data...")
test_scenarios = test[SCENARIO_COL].unique()
test_data = []

for sid in test_scenarios:
    features = prepare_scenario_features(test, sid, props_dict, typical_dict, PROPS_ORDER)
    test_data.append({
        'scenario_id': sid,
        'features': features
    })

print(f"Test scenarios: {len(test_data)}")

# Create arrays
X_train = []
y_visc = []
y_oxid = []
train_ids_list = []

for item in train_data:
    vec = create_scenario_vector(item, le, PROPS_ORDER)
    X_train.append(vec)
    y_visc.append(item['target_visc'])
    y_oxid.append(item['target_oxid'])
    train_ids_list.append(item['scenario_id'])

X_train = np.array(X_train)
y_visc = np.array(y_visc)
y_oxid = np.array(y_oxid)

X_test = []
test_ids_list = []

for item in test_data:
    vec = create_scenario_vector(item, le, PROPS_ORDER)
    X_test.append(vec)
    test_ids_list.append(item['scenario_id'])

X_test = np.array(X_test)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_visc: {y_visc.shape}, y_oxid: {y_oxid.shape}")

# Normalize
X_train_nan = np.nan_to_num(X_train, nan=0.0)
X_test_nan = np.nan_to_num(X_test, nan=0.0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_nan)
X_test_scaled = scaler.transform(X_test_nan)

print(f"After scaling - X_train: {X_train_scaled.shape}")

# Save
np.save(os.path.join(DATA_PATH, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(DATA_PATH, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(DATA_PATH, 'y_visc.npy'), y_visc)
np.save(os.path.join(DATA_PATH, 'y_oxid.npy'), y_oxid)

pd.DataFrame({'scenario_id': train_ids_list}).to_csv(os.path.join(DATA_PATH, 'train_ids.csv'), index=False)
pd.DataFrame({'scenario_id': test_ids_list}).to_csv(os.path.join(DATA_PATH, 'test_ids.csv'), index=False)

with open(os.path.join(DATA_PATH, 'le.pkl'), 'wb') as f:
    pickle.dump(le, f)
with open(os.path.join(DATA_PATH, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(DATA_PATH, 'props_order.pkl'), 'wb') as f:
    pickle.dump(PROPS_ORDER, f)

print("Data prepared and saved!")