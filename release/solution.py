#!/usr/bin/env python3
"""
NefteCode 2026 - Training with Learning Curves
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DATA_PATH = "data/"

train = pd.read_csv(f"{DATA_PATH}/daimler_mixtures_train.csv")
test = pd.read_csv(f"{DATA_PATH}/daimler_mixtures_test.csv")
props = pd.read_csv(f"{DATA_PATH}/daimler_component_properties.csv")

TARGET_VISC = (
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
)
TARGET_OXID = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
TEMP_COL = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
TIME_COL = "Время испытания | - Daimler Oxidation Test (DOT), ч"
BIO_COL = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
CAT_COL = "Дозировка катализатора, категория"
COMP_COL = "Компонент"
BATCH_COL = "Наименование партии"
MASS_COL = "Массовая доля, %"
SCENARIO_COL = "scenario_id"

MAX_COMPONENTS = 50
COMP_DIM = 23
HIDDEN_DIM = 128
NUM_HEADS = 4
M = 12
DROPOUT = 0.1
GLOBAL_DIM = 4
EPOCHS = 2000


def create_properties_dict(props_df):
    props_dict = {}
    typical = props_df[props_df[BATCH_COL].astype(str).str.lower() == "typical"]
    typical_dict = {}

    for _, row in typical.iterrows():
        comp = row[COMP_COL]
        prop_name = row["Наименование показателя"]
        prop_value = row["Значение показателя"]
        if pd.notna(prop_value):
            if comp not in typical_dict:
                typical_dict[comp] = {}
            typical_dict[comp][prop_name] = prop_value

    for _, row in props_df.iterrows():
        comp = row[COMP_COL]
        batch = row[BATCH_COL]
        prop_name = row["Наименование показателя"]
        prop_value = row["Значение показателя"]

        if pd.isna(batch) or pd.isna(prop_value):
            continue

        key = (comp, batch)
        if key not in props_dict:
            props_dict[key] = {}
        props_dict[key][prop_name] = prop_value

    return props_dict, typical_dict


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


def safe_float(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return np.nan


def prepare_scenario_2d(
    scenario_df, scenario_id, props_dict, typical_dict, numeric_props, le
):
    scenario_data = scenario_df[scenario_df[SCENARIO_COL] == scenario_id]
    test_params = scenario_data.iloc[0]

    global_features = np.array(
        [
            safe_float(test_params[TEMP_COL]),
            safe_float(test_params[TIME_COL]),
            safe_float(test_params[BIO_COL]),
            safe_float(test_params[CAT_COL]),
        ]
    )

    components = []
    for _, row in scenario_data.iterrows():
        comp = row[COMP_COL]
        batch = row[BATCH_COL] if pd.notna(row[BATCH_COL]) else ""
        mass = safe_float(row[MASS_COL])
        log_mass = np.log(mass + 1e-6) if not np.isnan(mass) else 0

        comp_props = get_component_properties(comp, batch, props_dict, typical_dict)

        type_enc = le.transform([comp])[0] if comp in le.classes_ else -1

        prop_vals = []
        for p in numeric_props:
            if p in comp_props:
                pv = safe_float(comp_props[p])
                prop_vals.append(pv if not np.isnan(pv) else 0)
            else:
                prop_vals.append(0)

        comp_vector = np.array([type_enc, mass, log_mass] + prop_vals[:20])
        components.append(comp_vector)

    components = np.array(components)

    if len(components) < MAX_COMPONENTS:
        padding = np.zeros((MAX_COMPONENTS - len(components), 23))
        components = np.vstack([components, padding])
    elif len(components) > MAX_COMPONENTS:
        components = components[:MAX_COMPONENTS]

    mask = np.zeros((MAX_COMPONENTS,))
    mask[: min(len(scenario_data), MAX_COMPONENTS)] = 1

    return global_features, components, mask


class DeepSetsEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, mask):
        phi_out = self.phi(x)
        masked_out = phi_out * mask.unsqueeze(-1)
        rho_out = self.rho(masked_out.mean(dim=1))
        return rho_out


class DeepSetsModel(nn.Module):
    def __init__(self, comp_dim, global_dim, hidden_dim=128, encode_dim=64):
        super().__init__()
        self.encoder = DeepSetsEncoder(comp_dim, hidden_dim, encode_dim)
        self.predictor = nn.Sequential(
            nn.Linear(global_dim + encode_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_x, comp_x, mask):
        encoded = self.encoder(comp_x, mask)
        combined = torch.cat([global_x, encoded], dim=-1)
        return self.predictor(combined).squeeze()


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class ISAB(nn.Module):
    def __init__(self, dim, num_heads=4, m=12):
        super().__init__()
        self.m = m
        self.I = nn.Parameter(torch.randn(1, m, dim))
        self.attn1 = MultiHeadAttention(dim, num_heads)
        self.attn2 = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        I = self.I.repeat(B, 1, 1)
        h = self.norm1(self.attn1(torch.cat([I, x], dim=1))[:, : self.m])
        out = self.norm2(self.attn2(torch.cat([h, x], dim=1))[:, self.m :])
        return out


class SetTransformer(nn.Module):
    def __init__(
        self,
        comp_dim,
        global_dim,
        hidden_dim=128,
        num_heads=4,
        num_isab=2,
        encode_dim=64,
    ):
        super().__init__()
        self.comp_embedding = nn.Linear(comp_dim, hidden_dim)
        self.global_embedding = nn.Linear(global_dim, hidden_dim // 2)
        self.isabs = nn.ModuleList(
            [ISAB(hidden_dim, num_heads, M) for _ in range(num_isab)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_x, comp_x, mask):
        comp_emb = self.comp_embedding(comp_x)
        for isab in self.isabs:
            comp_emb = isab(comp_emb)
        comp_emb = self.norm(comp_emb)
        pooled = (comp_emb * mask.unsqueeze(-1)).mean(dim=1)
        global_emb = self.global_embedding(global_x)
        combined = torch.cat([global_emb, pooled], dim=-1)
        return self.predictor(combined).squeeze()


def train_model_with_curve(
    model,
    X_global_train,
    X_comp_train,
    mask_train,
    y_train,
    X_global_val,
    X_comp_val,
    mask_val,
    y_val,
    epochs=EPOCHS,
    lr=0.001,
    batch_size=32,
    weight_decay=1e-4,
    model_name="model",
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_global_t = torch.tensor(X_global_train, dtype=torch.float32)
    X_comp_t = torch.tensor(X_comp_train, dtype=torch.float32)
    mask_t = torch.tensor(mask_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_global_t, X_comp_t, mask_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    X_global_val_t = torch.tensor(X_global_val, dtype=torch.float32).to(device)
    X_comp_val_t = torch.tensor(X_comp_val, dtype=torch.float32).to(device)
    mask_val_t = torch.tensor(mask_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None

    print(f"Training {model_name} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xg, xc, m, yb in loader:
            xg, xc, m, yb = xg.to(device), xc.to(device), m.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xg, xc, m)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(loader))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_global_val_t, X_comp_val_t, mask_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        if (epoch + 1) % 200 == 0:
            print(
                f"  Epoch {epoch + 1}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}"
            )

    if best_state:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def predict_model(model, X_global, X_comp, mask):
    model.eval()
    X_global_t = torch.tensor(X_global, dtype=torch.float32).to(device)
    X_comp_t = torch.tensor(X_comp, dtype=torch.float32).to(device)
    mask_t = torch.tensor(mask, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(X_global_t, X_comp_t, mask_t).cpu().numpy()
    return pred


def main():
    print("=== Loading data ===")
    props_dict, typical_dict = create_properties_dict(props)

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
                    val = str(d[p]).replace(",", ".")
                    values.append(float(val))
                except Exception:
                    pass
        if len(values) > 10:
            numeric_props.append(p)

    PROPS_ORDER = numeric_props[:20]

    all_component_types = set(train[COMP_COL].unique()) | set(test[COMP_COL].unique())
    le = LabelEncoder()
    le.fit(list(all_component_types))

    print("=== Preparing data ===")
    train_scenarios = train[SCENARIO_COL].unique()
    X_train_global, X_train_components, X_train_mask = [], [], []
    y_visc, y_oxid, train_ids = [], [], []

    for sid in train_scenarios:
        global_f, components, mask = prepare_scenario_2d(
            train, sid, props_dict, typical_dict, PROPS_ORDER, le
        )
        X_train_global.append(global_f)
        X_train_components.append(components)
        X_train_mask.append(mask)

        scenario_rows = train[train[SCENARIO_COL] == sid]
        y_visc.append(scenario_rows[TARGET_VISC].iloc[0])
        y_oxid.append(scenario_rows[TARGET_OXID].iloc[0])
        train_ids.append(sid)

    X_train_global = np.array(X_train_global)
    X_train_components = np.array(X_train_components)
    X_train_mask = np.array(X_train_mask)
    y_visc = np.array(y_visc)
    y_oxid = np.array(y_oxid)

    test_scenarios = test[SCENARIO_COL].unique()
    X_test_global, X_test_components, X_test_mask, test_ids = [], [], [], []

    for sid in test_scenarios:
        global_f, components, mask = prepare_scenario_2d(
            test, sid, props_dict, typical_dict, PROPS_ORDER, le
        )
        X_test_global.append(global_f)
        X_test_components.append(components)
        X_test_mask.append(mask)
        test_ids.append(sid)

    X_test_global = np.array(X_test_global)
    X_test_components = np.array(X_test_components)
    X_test_mask = np.array(X_test_mask)

    scaler_g = StandardScaler()
    X_train_global_scaled = scaler_g.fit_transform(np.nan_to_num(X_train_global, nan=0))
    X_test_global_scaled = scaler_g.transform(np.nan_to_num(X_test_global, nan=0))

    X_train_comp_flat = X_train_components.reshape(X_train_components.shape[0], -1)
    X_test_comp_flat = X_test_components.reshape(X_test_components.shape[0], -1)
    scaler_c = StandardScaler()
    X_train_comp_scaled = scaler_c.fit_transform(
        np.nan_to_num(X_train_comp_flat, nan=0)
    )
    X_test_comp_scaled = scaler_c.transform(np.nan_to_num(X_test_comp_flat, nan=0))

    X_train_components = X_train_comp_scaled.reshape(X_train_components.shape)
    X_test_components = X_test_comp_scaled.reshape(X_test_components.shape)

    idx = np.arange(len(y_visc))
    idx_train, idx_val = train_test_split(idx, test_size=0.2, random_state=42)

    X_global_train = X_train_global_scaled[idx_train]
    X_comp_train = X_train_components[idx_train]
    mask_train = X_train_mask[idx_train]
    y_visc_train = y_visc[idx_train]
    y_oxid_train = y_oxid[idx_train]

    X_global_val = X_train_global_scaled[idx_val]
    X_comp_val = X_train_components[idx_val]
    mask_val = X_train_mask[idx_val]
    y_visc_val = y_visc[idx_val]
    y_oxid_val = y_oxid[idx_val]

    print(f"Train: {len(idx_train)} samples, Val: {len(idx_val)} samples")

    all_train_losses = {"visc": [], "oxid": []}
    all_val_losses = {"visc": [], "oxid": []}

    print("\n=== Training Deep Sets (Viscosity) ===")
    ds_visc = DeepSetsModel(COMP_DIM, GLOBAL_DIM, HIDDEN_DIM, 64)
    ds_visc, train_l, val_l = train_model_with_curve(
        ds_visc,
        X_global_train,
        X_comp_train,
        mask_train,
        y_visc_train,
        X_global_val,
        X_comp_val,
        mask_val,
        y_visc_val,
        epochs=EPOCHS,
        model_name="DS_Viscosity",
    )
    all_train_losses["visc"].append(train_l)
    all_val_losses["visc"].append(val_l)

    print("\n=== Training Deep Sets (Oxidation) ===")
    ds_oxid = DeepSetsModel(COMP_DIM, GLOBAL_DIM, HIDDEN_DIM, 64)
    ds_oxid, train_l, val_l = train_model_with_curve(
        ds_oxid,
        X_global_train,
        X_comp_train,
        mask_train,
        y_oxid_train,
        X_global_val,
        X_comp_val,
        mask_val,
        y_oxid_val,
        epochs=EPOCHS,
        model_name="DS_Oxidation",
    )
    all_train_losses["oxid"].append(train_l)
    all_val_losses["oxid"].append(val_l)

    print("\n=== Training Set Transformer (Viscosity) ===")
    st_visc = SetTransformer(COMP_DIM, GLOBAL_DIM, HIDDEN_DIM, NUM_HEADS, 2, 64)
    st_visc, train_l, val_l = train_model_with_curve(
        st_visc,
        X_global_train,
        X_comp_train,
        mask_train,
        y_visc_train,
        X_global_val,
        X_comp_val,
        mask_val,
        y_visc_val,
        epochs=EPOCHS,
        model_name="ST_Viscosity",
    )
    all_train_losses["visc"].append(train_l)
    all_val_losses["visc"].append(val_l)

    print("\n=== Training Set Transformer (Oxidation) ===")
    st_oxid = SetTransformer(COMP_DIM, GLOBAL_DIM, HIDDEN_DIM, NUM_HEADS, 2, 64)
    st_oxid, train_l, val_l = train_model_with_curve(
        st_oxid,
        X_global_train,
        X_comp_train,
        mask_train,
        y_oxid_train,
        X_global_val,
        X_comp_val,
        mask_val,
        y_oxid_val,
        epochs=EPOCHS,
        model_name="ST_Oxidation",
    )
    all_train_losses["oxid"].append(train_l)
    all_val_losses["oxid"].append(val_l)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(all_train_losses["visc"][0], label="Train Loss (DS)", alpha=0.7)
    axes[0, 0].plot(all_val_losses["visc"][0], label="Val Loss (DS)", alpha=0.7)
    axes[0, 0].set_title("Deep Sets - Viscosity")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(all_train_losses["oxid"][0], label="Train Loss (DS)", alpha=0.7)
    axes[0, 1].plot(all_val_losses["oxid"][0], label="Val Loss (DS)", alpha=0.7)
    axes[0, 1].set_title("Deep Sets - Oxidation")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(all_train_losses["visc"][1], label="Train Loss (ST)", alpha=0.7)
    axes[1, 0].plot(all_val_losses["visc"][1], label="Val Loss (ST)", alpha=0.7)
    axes[1, 0].set_title("Set Transformer - Viscosity")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(all_train_losses["oxid"][1], label="Train Loss (ST)", alpha=0.7)
    axes[1, 1].plot(all_val_losses["oxid"][1], label="Val Loss (ST)", alpha=0.7)
    axes[1, 1].set_title("Set Transformer - Oxidation")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    print("\nSaved: learning_curves.png")

    print("\n=== Final Predictions (full training) ===")

    ds_visc_full = DeepSetsModel(COMP_DIM, GLOBAL_DIM, HIDDEN_DIM, 64)
    ds_visc_full, _, _ = train_model_with_curve(
        ds_visc_full,
        X_train_global_scaled,
        X_train_components,
        X_train_mask,
        y_visc,
        X_train_global_scaled,
        X_train_components,
        X_train_mask,
        y_visc,
        epochs=EPOCHS,
        model_name="DS_Viscosity_Full",
    )

    ds_oxid_full = DeepSetsModel(COMP_DIM, GLOBAL_DIM, HIDDEN_DIM, 64)
    ds_oxid_full, _, _ = train_model_with_curve(
        ds_oxid_full,
        X_train_global_scaled,
        X_train_components,
        X_train_mask,
        y_oxid,
        X_train_global_scaled,
        X_train_components,
        X_train_mask,
        y_oxid,
        epochs=EPOCHS,
        model_name="DS_Oxidation_Full",
    )

    st_visc_full = SetTransformer(COMP_DIM, GLOBAL_DIM, HIDDEN_DIM, NUM_HEADS, 2, 64)
    st_visc_full, _, _ = train_model_with_curve(
        st_visc_full,
        X_train_global_scaled,
        X_train_components,
        X_train_mask,
        y_visc,
        X_train_global_scaled,
        X_train_components,
        X_train_mask,
        y_visc,
        epochs=EPOCHS,
        model_name="ST_Viscosity_Full",
    )

    st_oxid_full = SetTransformer(COMP_DIM, GLOBAL_DIM, HIDDEN_DIM, NUM_HEADS, 2, 64)
    st_oxid_full, _, _ = train_model_with_curve(
        st_oxid_full,
        X_train_global_scaled,
        X_train_components,
        X_train_mask,
        y_oxid,
        X_train_global_scaled,
        X_train_components,
        X_train_mask,
        y_oxid,
        epochs=EPOCHS,
        model_name="ST_Oxidation_Full",
    )

    pred_ds_visc = predict_model(
        ds_visc_full, X_test_global_scaled, X_test_components, X_test_mask
    )
    pred_st_visc = predict_model(
        st_visc_full, X_test_global_scaled, X_test_components, X_test_mask
    )
    pred_ds_oxid = predict_model(
        ds_oxid_full, X_test_global_scaled, X_test_components, X_test_mask
    )
    pred_st_oxid = predict_model(
        st_oxid_full, X_test_global_scaled, X_test_components, X_test_mask
    )

    pred_visc = 0.6 * pred_st_visc + 0.4 * pred_ds_visc
    pred_oxid = (pred_ds_oxid + pred_st_oxid) / 2

    print(f"\nViscosity: mean={pred_visc.mean():.2f}, std={pred_visc.std():.2f}")
    print(f"Oxidation: mean={pred_oxid.mean():.2f}, std={pred_oxid.std():.2f}")

    submission = pd.DataFrame(
        {"scenario_id": test_ids, TARGET_VISC: pred_visc, TARGET_OXID: pred_oxid}
    )

    submission.to_csv("predictions.csv", index=False, encoding="utf-8")
    print("\nSaved: predictions.csv")
    print(submission.head())


if __name__ == "__main__":
    main()
