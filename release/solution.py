#!/usr/bin/env python3
"""
NefteCode 2026 — решение с упором на качество (вязкость + окисление).
- Тип компонента: только one-hot (без ordinal LabelEncoder).
- StandardScaler только на непрерывные признаки компонента; one-hot не масштабируются.
- Deep Sets (mean+max) + Set Transformer (ISAB + PMA pool); синергии 7 бинарных флагов.
- Вязкость: RobustScaler + SmoothL1; окисление: StandardScaler + MSE; Zn/P только для ZDDP.
- K-fold: скейлеры таргетов только на train фолда; лучший чекпоинт по прокси метрики (MAE в z, как лидерборд); warmup+cosine LR; фаза графиков = fold0 того же KFold.
"""

from pathlib import Path
import json
import math

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_PATH = REPO_ROOT / "data"

RNG_SEED = 42


def set_seed(seed: int = RNG_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RNG_SEED)


def _loss_curves_payload(train_losses, val_losses):
    tr = [float(x) for x in train_losses]
    va = [float(x) for x in val_losses]
    return {
        "train": tr,
        "val": va,
        "n_epochs": len(tr),
        "final_train": tr[-1] if tr else None,
        "final_val": va[-1] if va else None,
        "min_val": float(min(va)) if va else None,
        "argmin_val_epoch": int(np.argmin(va)) + 1 if va else None,
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train = pd.read_csv(DATA_PATH / "daimler_mixtures_train.csv")
test = pd.read_csv(DATA_PATH / "daimler_mixtures_test.csv")
props = pd.read_csv(DATA_PATH / "daimler_component_properties.csv")

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
HIDDEN_DIM = 128
ENCODE_DIM = 96
NUM_HEADS = 4
NUM_ISAB = 3
M = 12
DROPOUT = 0.25
GLOBAL_DIM = 4
EPOCHS = 4000
PATIENCE = 180
NUM_PROPS = 24
BATCH_SIZE = 32 if torch.cuda.is_available() else 16
WARMUP_FRAC = 0.05
# K-fold: val только для выбора лучшего чекпоинта (без early stop — мало точек, лосс шумный)
K_FOLD = 5
K_FOLD_SEED = 42

COMPONENT_TYPES = [
    "base_oil",
    "ao_phenol",
    "ao_amine",
    "ao_other",
    "detergent",
    "dispersant",
    "zddp",
    "molybdenum",
    "thickener",
    "depressor",
    "antifoam",
    "other",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(COMPONENT_TYPES)}
SYNERGY_DIM = 7
N_CAT = len(COMPONENT_TYPES)
# [mass, log_mass] + NUM_PROPS + [p_zn_ratio] — только это под StandardScaler
N_CONT_PER_COMP = 2 + NUM_PROPS + 1


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
    """
    Measured properties for (comp, batch) override typical values per task rules.
    Never substitute properties from another batch of the same component.
    """
    typical_vals = dict(typical_dict.get(comp, {}))
    if pd.isna(batch) or batch == "":
        return typical_vals
    key = (comp, batch)
    if key not in props_dict:
        return typical_vals
    measured = props_dict[key]
    merged = dict(typical_vals)
    merged.update(measured)
    return merged


def safe_float(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return np.nan


def compute_zn_p_ratio(prop_vals, prop_names):
    """Zn/P style ratio from named properties; avoids magic indices into sorted props."""
    zn_idx = phos_idx = None
    for i, name in enumerate(prop_names):
        nl = str(name).lower()
        if zn_idx is None and "цинк" in nl:
            zn_idx = i
        if phos_idx is None and "фосфор" in nl:
            phos_idx = i
    if (
        zn_idx is None
        or phos_idx is None
        or zn_idx >= len(prop_vals)
        or phos_idx >= len(prop_vals)
    ):
        return 0.0
    p = float(prop_vals[phos_idx])
    zn = float(prop_vals[zn_idx])
    if p <= 0:
        return 0.0
    return min(zn / (p + 1e-6), 10.0)


def get_component_type(comp_name):
    name_lower = comp_name.lower() if isinstance(comp_name, str) else ""
    if "базовое" in name_lower or "масло" in name_lower:
        return "base_oil"
    elif "антиоксидант" in name_lower:
        if "фенольн" in name_lower:
            return "ao_phenol"
        elif "аминн" in name_lower:
            return "ao_amine"
        return "ao_other"
    elif "детергент" in name_lower or "моющее" in name_lower:
        return "detergent"
    elif "дисперсант" in name_lower:
        return "dispersant"
    elif "противоизнос" in name_lower or "zddp" in name_lower:
        return "zddp"
    elif "молибден" in name_lower or "modtc" in name_lower or "modtp" in name_lower:
        return "molybdenum"
    elif "загуститель" in name_lower or "полимер" in name_lower:
        return "thickener"
    elif "депрессор" in name_lower:
        return "depressor"
    elif "антипен" in name_lower:
        return "antifoam"
    return "other"


def prepare_scenario_v2(scenario_df, scenario_id, props_dict, typical_dict, numeric_props):
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
    comp_types = []

    for _, row in scenario_data.iterrows():
        comp = row[COMP_COL]
        batch = row[BATCH_COL] if pd.notna(row[BATCH_COL]) else ""
        mass = safe_float(row[MASS_COL])
        log_mass = np.log(mass + 1e-6) if not np.isnan(mass) else 0

        comp_props = get_component_properties(comp, batch, props_dict, typical_dict)
        comp_type = get_component_type(comp)
        comp_types.append(comp_type)

        prop_vals = []
        for p in numeric_props:
            if p in comp_props:
                pv = safe_float(comp_props[p])
                prop_vals.append(pv if not np.isnan(pv) else 0)
            else:
                prop_vals.append(0)

        if comp_type == "zddp":
            p_zn_ratio = compute_zn_p_ratio(prop_vals, numeric_props[:NUM_PROPS])
        else:
            p_zn_ratio = 0.0

        one_hot = np.zeros(N_CAT)
        if comp_type in TYPE_TO_IDX:
            one_hot[TYPE_TO_IDX[comp_type]] = 1

        comp_vector = np.concatenate(
            [
                [mass, log_mass],
                np.array(prop_vals[:NUM_PROPS], dtype=float),
                [p_zn_ratio],
                one_hot,
            ]
        )

        components.append(comp_vector)

    components = np.array(components)
    if len(components) < MAX_COMPONENTS:
        padding = np.zeros((MAX_COMPONENTS - len(components), components.shape[1]))
        components = np.vstack([components, padding])
    elif len(components) > MAX_COMPONENTS:
        components = components[:MAX_COMPONENTS]

    mask = np.zeros((MAX_COMPONENTS,))
    mask[: min(len(scenario_data), MAX_COMPONENTS)] = 1

    comp_types_set = set(comp_types)
    has_phenol_amine = int(
        "ao_phenol" in comp_types_set and "ao_amine" in comp_types_set
    )
    has_amine_molybdenum = int(
        "ao_amine" in comp_types_set and "molybdenum" in comp_types_set
    )
    has_zddp_phenol = int("zddp" in comp_types_set and "ao_phenol" in comp_types_set)
    has_detergent_zddp = int("detergent" in comp_types_set and "zddp" in comp_types_set)
    has_dispersant_zddp = int("dispersant" in comp_types_set and "zddp" in comp_types_set)
    has_thickener_zddp = int("thickener" in comp_types_set and "zddp" in comp_types_set)
    has_phenol_molybdenum = int(
        "ao_phenol" in comp_types_set and "molybdenum" in comp_types_set
    )

    synergy_flags = np.array(
        [
            has_phenol_amine,
            has_amine_molybdenum,
            has_zddp_phenol,
            has_detergent_zddp,
            has_dispersant_zddp,
            has_thickener_zddp,
            has_phenol_molybdenum,
        ],
        dtype=float,
    )

    return global_features, components, mask, synergy_flags


class DeepSetsEncoder(nn.Module):
    """Deep Sets с агрегированием mean + max по компонентам (маска учитывается)."""

    def __init__(self, input_dim, hidden_dim=128, output_dim=96):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, mask):
        phi_out = self.phi(x)
        m = mask.unsqueeze(-1)
        masked = phi_out * m
        denom = m.sum(dim=1).clamp(min=1e-6)
        pooled_mean = masked.sum(dim=1) / denom
        neg_mask = torch.full_like(phi_out, -1e4)
        safe = torch.where(m > 0, phi_out, neg_mask)
        pooled_max, _ = safe.max(dim=1)
        rho_in = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.rho(rho_in)


class DeepSetsModel(nn.Module):
    def __init__(self, comp_dim, global_dim, hidden_dim=128, encode_dim=96):
        super().__init__()
        self.encoder = DeepSetsEncoder(comp_dim, hidden_dim, encode_dim)
        self.predictor = nn.Sequential(
            nn.Linear(global_dim + encode_dim + SYNERGY_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, global_x, comp_x, mask, synergy_flags):
        encoded = self.encoder(comp_x, mask)
        combined = torch.cat([global_x, encoded, synergy_flags], dim=-1)
        return self.predictor(combined).squeeze(-1)


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
    def __init__(self, dim, num_heads=4, m=8):
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


class PMAPool(nn.Module):
    """PMA: один обучаемый seed-вектор, cross-attention к множеству компонентов."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=min(0.1, DROPOUT)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, H = x.shape
        s = self.S.expand(B, -1, -1)
        key_padding_mask = mask < 0.5
        out, _ = self.mha(s, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))


class SetTransformer(nn.Module):
    def __init__(
        self,
        comp_dim,
        global_dim,
        hidden_dim=128,
        num_heads=4,
        num_isab=3,
        encode_dim=96,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.comp_embedding = nn.Linear(comp_dim, hidden_dim)
        self.global_embedding = nn.Linear(global_dim, hidden_dim // 2)
        self.isabs = nn.ModuleList(
            [ISAB(hidden_dim, num_heads, M) for _ in range(num_isab)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.pma = PMAPool(hidden_dim, num_heads)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + SYNERGY_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, global_x, comp_x, mask, synergy_flags):
        comp_emb = self.comp_embedding(comp_x)
        for isab in self.isabs:
            comp_emb = isab(comp_emb)
        comp_emb = self.norm(comp_emb)
        pooled = self.pma(comp_emb, mask)
        global_emb = self.global_embedding(global_x)
        combined = torch.cat([global_emb, pooled, synergy_flags], dim=-1)
        return self.predictor(combined).squeeze(-1)


def _set_lr_cosine_warmup(optimizer, epoch, epochs, base_lr, warmup_epochs, eta_min):
    if epoch < warmup_epochs:
        alpha = (epoch + 1) / max(warmup_epochs, 1)
        lr_cur = base_lr * (0.1 + 0.9 * alpha)
    else:
        t = epoch - warmup_epochs
        T = max(epochs - warmup_epochs, 1)
        if T <= 1:
            lr_cur = eta_min
        else:
            lr_cur = eta_min + (base_lr - eta_min) * 0.5 * (
                1.0 + math.cos(math.pi * t / max(T - 1, 1))
            )
    for pg in optimizer.param_groups:
        pg["lr"] = lr_cur


def train_model_early_stop(
    model,
    X_global_train,
    X_comp_train,
    mask_train,
    y_train,
    X_global_val,
    X_comp_val,
    mask_val,
    y_val,
    synergy_train,
    synergy_val,
    scaler_y,
    epochs=EPOCHS,
    lr=0.001,
    batch_size=BATCH_SIZE,
    weight_decay=5e-5,
    model_name="model",
    patience=None,
    use_smooth_l1=False,
    generator_seed=None,
    use_leaderboard_checkpoint=True,
):
    """Чекпоинт по прокси лидерборда: MAE в z-пространстве (тот же scaler_y, fit на train фолда)."""
    if generator_seed is None:
        generator_seed = RNG_SEED
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    eta_min = lr * 1e-5
    warmup_epochs = max(1, min(int(epochs * WARMUP_FRAC), epochs - 1))

    criterion = nn.SmoothL1Loss(beta=1.0) if use_smooth_l1 else nn.MSELoss()

    y_train_norm = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_val_norm = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    X_global_t = torch.tensor(X_global_train, dtype=torch.float32)
    X_comp_t = torch.tensor(X_comp_train, dtype=torch.float32)
    mask_t = torch.tensor(mask_train, dtype=torch.float32)
    synergy_t = torch.tensor(synergy_train, dtype=torch.float32)
    y_t = torch.tensor(y_train_norm, dtype=torch.float32)
    dataset = TensorDataset(X_global_t, X_comp_t, mask_t, synergy_t, y_t)
    gen = torch.Generator().manual_seed(int(generator_seed) % (2**32))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=gen
    )

    X_global_val_t = torch.tensor(X_global_val, dtype=torch.float32).to(device)
    X_comp_val_t = torch.tensor(X_comp_val, dtype=torch.float32).to(device)
    mask_val_t = torch.tensor(mask_val, dtype=torch.float32).to(device)
    synergy_val_t = torch.tensor(synergy_val, dtype=torch.float32).to(device)
    y_val_norm_t = torch.tensor(y_val_norm, dtype=torch.float32).to(device)

    train_losses, val_losses = [], []
    best_score = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(
        f"Training {model_name} for up to {epochs} epochs "
        f"(warmup={warmup_epochs}, LB_checkpoint={use_leaderboard_checkpoint})..."
    )

    for epoch in range(epochs):
        _set_lr_cosine_warmup(
            optimizer, epoch, epochs, lr, warmup_epochs, eta_min
        )

        model.train()
        epoch_loss = 0
        for xg, xc, m, sf, yb in loader:
            xg, xc, m, sf, yb = (
                xg.to(device),
                xc.to(device),
                m.to(device),
                sf.to(device),
                yb.to(device),
            )
            optimizer.zero_grad()
            pred = model(xg, xc, m, sf)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(loader))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_global_val_t, X_comp_val_t, mask_val_t, synergy_val_t)
            val_loss = criterion(val_pred, y_val_norm_t).item()
            val_mae_z = (val_pred - y_val_norm_t).abs().mean().item()
        val_losses.append(val_loss)

        if use_leaderboard_checkpoint:
            score = val_mae_z
        else:
            score = val_loss

        if score < best_score - 1e-9:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience is not None and epochs_no_improve >= patience:
            tag = "val MAE(z) proxy" if use_leaderboard_checkpoint else "val loss"
            print(
                f"  Early stop at epoch {epoch + 1} (no {tag} improvement for {patience} epochs)"
            )
            break

        if (epoch + 1) % 500 == 0:
            print(
                f"  Epoch {epoch + 1}: train={train_losses[-1]:.4f}, "
                f"val_loss={val_loss:.4f}, val_MAE_z={val_mae_z:.4f}"
            )

    if best_state:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def predict_model(model, X_global, X_comp, mask, synergy_flags, scaler_y):
    model.eval()
    X_global_t = torch.tensor(X_global, dtype=torch.float32).to(device)
    X_comp_t = torch.tensor(X_comp, dtype=torch.float32).to(device)
    mask_t = torch.tensor(mask, dtype=torch.float32).to(device)
    synergy_t = torch.tensor(synergy_flags, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_norm = model(X_global_t, X_comp_t, mask_t, synergy_t).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
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

    PROPS_ORDER = numeric_props[:NUM_PROPS]

    print("=== Preparing data ===")
    train_scenarios = train[SCENARIO_COL].unique()
    X_train_global, X_train_components, X_train_mask, X_train_synergy = [], [], [], []
    y_visc, y_oxid, train_ids = [], [], []

    for sid in train_scenarios:
        global_f, components, mask, synergy = prepare_scenario_v2(
            train, sid, props_dict, typical_dict, PROPS_ORDER
        )
        X_train_global.append(global_f)
        X_train_components.append(components)
        X_train_mask.append(mask)
        X_train_synergy.append(synergy)

        scenario_rows = train[train[SCENARIO_COL] == sid]
        y_visc.append(scenario_rows[TARGET_VISC].iloc[0])
        y_oxid.append(scenario_rows[TARGET_OXID].iloc[0])
        train_ids.append(sid)

    X_train_global = np.array(X_train_global)
    X_train_components = np.array(X_train_components)
    X_train_mask = np.array(X_train_mask)
    X_train_synergy = np.array(X_train_synergy)
    y_visc = np.array(y_visc)
    y_oxid = np.array(y_oxid)

    test_scenarios = test[SCENARIO_COL].unique()
    X_test_global, X_test_components, X_test_mask, X_test_synergy, test_ids = (
        [],
        [],
        [],
        [],
        [],
    )

    for sid in test_scenarios:
        global_f, components, mask, synergy = prepare_scenario_v2(
            test, sid, props_dict, typical_dict, PROPS_ORDER
        )
        X_test_global.append(global_f)
        X_test_components.append(components)
        X_test_mask.append(mask)
        X_test_synergy.append(synergy)
        test_ids.append(sid)

    X_test_global = np.array(X_test_global)
    X_test_components = np.array(X_test_components)
    X_test_mask = np.array(X_test_mask)
    X_test_synergy = np.array(X_test_synergy)

    scaler_g = StandardScaler()
    X_train_global_scaled = scaler_g.fit_transform(np.nan_to_num(X_train_global, nan=0))
    X_test_global_scaled = scaler_g.transform(np.nan_to_num(X_test_global, nan=0))

    n_s, n_pad, d_comp = X_train_components.shape
    assert d_comp == N_CONT_PER_COMP + N_CAT
    cont_train = X_train_components[:, :, :N_CONT_PER_COMP].reshape(-1, N_CONT_PER_COMP)
    cat_train = X_train_components[:, :, N_CONT_PER_COMP:]
    cont_test = X_test_components[:, :, :N_CONT_PER_COMP].reshape(-1, N_CONT_PER_COMP)
    cat_test = X_test_components[:, :, N_CONT_PER_COMP:]

    scaler_c = StandardScaler()
    cont_train_s = scaler_c.fit_transform(np.nan_to_num(cont_train, nan=0))
    cont_test_s = scaler_c.transform(np.nan_to_num(cont_test, nan=0))

    X_train_components = np.concatenate(
        [
            cont_train_s.reshape(n_s, n_pad, N_CONT_PER_COMP),
            cat_train,
        ],
        axis=2,
    )
    X_test_components = np.concatenate(
        [
            cont_test_s.reshape(X_test_components.shape[0], n_pad, N_CONT_PER_COMP),
            cat_test,
        ],
        axis=2,
    )

    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=K_FOLD_SEED)
    all_idx = np.arange(len(y_visc))
    kf_splits = list(kf.split(all_idx))
    fold0_tr, fold0_va = kf_splits[0]

    scaler_visc_phase = RobustScaler()
    scaler_visc_phase.fit(y_visc[fold0_tr].reshape(-1, 1))
    scaler_oxid_phase = StandardScaler()
    scaler_oxid_phase.fit(y_oxid[fold0_tr].reshape(-1, 1))

    comp_dim = X_train_components.shape[2]
    print(
        f"Comp dim: {comp_dim} (cont={N_CONT_PER_COMP}, cat={N_CAT}), synergy={X_train_synergy.shape[1]}"
    )

    idx_train, idx_val = fold0_tr, fold0_va

    X_global_train = X_train_global_scaled[idx_train]
    X_comp_train = X_train_components[idx_train]
    mask_train = X_train_mask[idx_train]
    synergy_train = X_train_synergy[idx_train]
    y_visc_train = y_visc[idx_train]
    y_oxid_train = y_oxid[idx_train]

    X_global_val = X_train_global_scaled[idx_val]
    X_comp_val = X_train_components[idx_val]
    mask_val = X_train_mask[idx_val]
    synergy_val = X_train_synergy[idx_val]
    y_visc_val = y_visc[idx_val]
    y_oxid_val = y_oxid[idx_val]

    print(
        f"Phase1 curves = K-fold fold0: train={len(idx_train)}, val={len(idx_val)} "
        "(target scalers fit on fold0 train only)"
    )

    all_curves = {}

    print("\n=== Training Deep Sets (Viscosity) ===")
    ds_visc, tr_l, val_l = train_model_early_stop(
        DeepSetsModel(comp_dim, GLOBAL_DIM, HIDDEN_DIM, ENCODE_DIM),
        X_global_train,
        X_comp_train,
        mask_train,
        y_visc_train,
        X_global_val,
        X_comp_val,
        mask_val,
        y_visc_val,
        synergy_train,
        synergy_val,
        scaler_visc_phase,
        model_name="DS_Viscosity",
        patience=PATIENCE,
        use_smooth_l1=True,
        use_leaderboard_checkpoint=True,
    )
    all_curves["ds_visc"] = (tr_l, val_l)

    print("\n=== Training Deep Sets (Oxidation) ===")
    ds_oxid, tr_l, val_l = train_model_early_stop(
        DeepSetsModel(comp_dim, GLOBAL_DIM, HIDDEN_DIM, ENCODE_DIM),
        X_global_train,
        X_comp_train,
        mask_train,
        y_oxid_train,
        X_global_val,
        X_comp_val,
        mask_val,
        y_oxid_val,
        synergy_train,
        synergy_val,
        scaler_oxid_phase,
        model_name="DS_Oxidation",
        patience=PATIENCE,
        use_leaderboard_checkpoint=True,
    )
    all_curves["ds_oxid"] = (tr_l, val_l)

    print("\n=== Training Set Transformer (Viscosity) ===")
    st_visc, tr_l, val_l = train_model_early_stop(
        SetTransformer(
            comp_dim, GLOBAL_DIM, HIDDEN_DIM, NUM_HEADS, NUM_ISAB, ENCODE_DIM
        ),
        X_global_train,
        X_comp_train,
        mask_train,
        y_visc_train,
        X_global_val,
        X_comp_val,
        mask_val,
        y_visc_val,
        synergy_train,
        synergy_val,
        scaler_visc_phase,
        lr=0.0005,
        model_name="ST_Viscosity",
        patience=PATIENCE,
        use_smooth_l1=True,
        use_leaderboard_checkpoint=True,
    )
    all_curves["st_visc"] = (tr_l, val_l)

    print("\n=== Training Set Transformer (Oxidation) ===")
    st_oxid, tr_l, val_l = train_model_early_stop(
        SetTransformer(
            comp_dim, GLOBAL_DIM, HIDDEN_DIM, NUM_HEADS, NUM_ISAB, ENCODE_DIM
        ),
        X_global_train,
        X_comp_train,
        mask_train,
        y_oxid_train,
        X_global_val,
        X_comp_val,
        mask_val,
        y_oxid_val,
        synergy_train,
        synergy_val,
        scaler_oxid_phase,
        lr=0.0005,
        model_name="ST_Oxidation",
        patience=PATIENCE,
        use_leaderboard_checkpoint=True,
    )
    all_curves["st_oxid"] = (tr_l, val_l)

    pred_ds_visc_list = []
    pred_st_visc_list = []
    pred_ds_oxid_list = []
    pred_st_oxid_list = []
    full_curves = None
    kfold_loss_log = []

    for fold, (tr_idx, va_idx) in enumerate(kf_splits):
        set_seed(RNG_SEED + fold)
        print(f"\n=== K-fold final {fold + 1}/{K_FOLD} (train={len(tr_idx)}, val={len(va_idx)}) ===")

        X_tr_g = X_train_global_scaled[tr_idx]
        X_tr_c = X_train_components[tr_idx]
        m_tr = X_train_mask[tr_idx]
        syn_tr = X_train_synergy[tr_idx]
        yv_tr = y_visc[tr_idx]
        yo_tr = y_oxid[tr_idx]

        X_va_g = X_train_global_scaled[va_idx]
        X_va_c = X_train_components[va_idx]
        m_va = X_train_mask[va_idx]
        syn_va = X_train_synergy[va_idx]
        yv_va = y_visc[va_idx]
        yo_va = y_oxid[va_idx]

        scaler_visc_f = RobustScaler()
        scaler_visc_f.fit(y_visc[tr_idx].reshape(-1, 1))
        scaler_oxid_f = StandardScaler()
        scaler_oxid_f.fit(y_oxid[tr_idx].reshape(-1, 1))

        curves_fold = {}

        ds_visc = DeepSetsModel(comp_dim, GLOBAL_DIM, HIDDEN_DIM, ENCODE_DIM)
        ds_visc, tr_f, val_f = train_model_early_stop(
            ds_visc,
            X_tr_g,
            X_tr_c,
            m_tr,
            yv_tr,
            X_va_g,
            X_va_c,
            m_va,
            yv_va,
            syn_tr,
            syn_va,
            scaler_visc_f,
            epochs=EPOCHS,
            model_name=f"DS_Viscosity_fold{fold}",
            patience=None,
            use_smooth_l1=True,
            generator_seed=RNG_SEED + fold,
            use_leaderboard_checkpoint=True,
        )
        curves_fold["ds_visc"] = _loss_curves_payload(tr_f, val_f)
        if fold == 0:
            full_curves = {"ds_visc": (tr_f, val_f)}

        ds_oxid = DeepSetsModel(comp_dim, GLOBAL_DIM, HIDDEN_DIM, ENCODE_DIM)
        ds_oxid, tr_f, val_f = train_model_early_stop(
            ds_oxid,
            X_tr_g,
            X_tr_c,
            m_tr,
            yo_tr,
            X_va_g,
            X_va_c,
            m_va,
            yo_va,
            syn_tr,
            syn_va,
            scaler_oxid_f,
            epochs=EPOCHS,
            model_name=f"DS_Oxidation_fold{fold}",
            patience=None,
            generator_seed=RNG_SEED + fold + 7,
            use_leaderboard_checkpoint=True,
        )
        curves_fold["ds_oxid"] = _loss_curves_payload(tr_f, val_f)
        if fold == 0:
            full_curves["ds_oxid"] = (tr_f, val_f)

        st_visc = SetTransformer(
            comp_dim, GLOBAL_DIM, HIDDEN_DIM, NUM_HEADS, NUM_ISAB, ENCODE_DIM
        )
        st_visc, tr_f, val_f = train_model_early_stop(
            st_visc,
            X_tr_g,
            X_tr_c,
            m_tr,
            yv_tr,
            X_va_g,
            X_va_c,
            m_va,
            yv_va,
            syn_tr,
            syn_va,
            scaler_visc_f,
            lr=0.0005,
            epochs=EPOCHS,
            model_name=f"ST_Viscosity_fold{fold}",
            patience=None,
            use_smooth_l1=True,
            generator_seed=RNG_SEED + fold + 13,
            use_leaderboard_checkpoint=True,
        )
        curves_fold["st_visc"] = _loss_curves_payload(tr_f, val_f)
        if fold == 0:
            full_curves["st_visc"] = (tr_f, val_f)

        st_oxid = SetTransformer(
            comp_dim, GLOBAL_DIM, HIDDEN_DIM, NUM_HEADS, NUM_ISAB, ENCODE_DIM
        )
        st_oxid, tr_f, val_f = train_model_early_stop(
            st_oxid,
            X_tr_g,
            X_tr_c,
            m_tr,
            yo_tr,
            X_va_g,
            X_va_c,
            m_va,
            yo_va,
            syn_tr,
            syn_va,
            scaler_oxid_f,
            lr=0.0005,
            epochs=EPOCHS,
            model_name=f"ST_Oxidation_fold{fold}",
            patience=None,
            generator_seed=RNG_SEED + fold + 19,
            use_leaderboard_checkpoint=True,
        )
        curves_fold["st_oxid"] = _loss_curves_payload(tr_f, val_f)
        if fold == 0:
            full_curves["st_oxid"] = (tr_f, val_f)

        kfold_loss_log.append(
            {
                "fold": fold,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "models": curves_fold,
            }
        )

        pred_ds_visc_list.append(
            predict_model(
                ds_visc,
                X_test_global_scaled,
                X_test_components,
                X_test_mask,
                X_test_synergy,
                scaler_visc_f,
            )
        )
        pred_st_visc_list.append(
            predict_model(
                st_visc,
                X_test_global_scaled,
                X_test_components,
                X_test_mask,
                X_test_synergy,
                scaler_visc_f,
            )
        )
        pred_ds_oxid_list.append(
            predict_model(
                ds_oxid,
                X_test_global_scaled,
                X_test_components,
                X_test_mask,
                X_test_synergy,
                scaler_oxid_f,
            )
        )
        pred_st_oxid_list.append(
            predict_model(
                st_oxid,
                X_test_global_scaled,
                X_test_components,
                X_test_mask,
                X_test_synergy,
                scaler_oxid_f,
            )
        )

    pred_ds_visc = np.mean(np.stack(pred_ds_visc_list, axis=0), axis=0)
    pred_st_visc = np.mean(np.stack(pred_st_visc_list, axis=0), axis=0)
    pred_ds_oxid = np.mean(np.stack(pred_ds_oxid_list, axis=0), axis=0)
    pred_st_oxid = np.mean(np.stack(pred_st_oxid_list, axis=0), axis=0)

    metrics_payload = {
        "config": {
            "EPOCHS": EPOCHS,
            "PATIENCE_phase1": PATIENCE,
            "K_FOLD": K_FOLD,
            "BATCH_SIZE": BATCH_SIZE,
            "device": str(device),
            "RNG_SEED": RNG_SEED,
        },
        "phase1_fold0_early_stop": {
            k: _loss_curves_payload(tr, va) for k, (tr, va) in all_curves.items()
        },
        "kfold_full_epochs_per_fold": kfold_loss_log,
    }
    metrics_path = SCRIPT_DIR / "training_loss_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved loss metrics: {metrics_path}")

    curve_keys = ["ds_visc", "ds_oxid", "st_visc", "st_oxid"]
    titles_split = [
        "DS — Viscosity (K-fold fold0, same as final split)",
        "DS — Oxidation (K-fold fold0)",
        "ST — Viscosity (K-fold fold0)",
        "ST — Oxidation (K-fold fold0)",
    ]
    titles_full = [
        f"DS — Viscosity (fold0 final, {EPOCHS} ep, best MAE_z)",
        f"DS — Oxidation (fold0 final, {EPOCHS} ep, best MAE_z)",
        f"ST — Viscosity (fold0 final, {EPOCHS} ep, best MAE_z)",
        f"ST — Oxidation (fold0 final, {EPOCHS} ep, best MAE_z)",
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle(
        f"Learning curves | {K_FOLD}-fold mean on test, fold0 curves",
        fontsize=12,
        y=1.002,
    )
    for i, key in enumerate(curve_keys):
        tr_s, va_s = all_curves[key]
        ep_s = np.arange(1, len(tr_s) + 1)
        axes[i, 0].plot(ep_s, tr_s, label="Train", alpha=0.75)
        axes[i, 0].plot(ep_s, va_s, label="Val", alpha=0.75)
        axes[i, 0].set_title(titles_split[i])
        axes[i, 0].set_xlabel("Epoch")
        axes[i, 0].set_ylabel("Loss")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        tr_f, va_f = full_curves[key]
        ep_f = np.arange(1, len(tr_f) + 1)
        axes[i, 1].plot(ep_f, tr_f, label="Train", alpha=0.75)
        axes[i, 1].plot(ep_f, va_f, label="Val", alpha=0.75)
        axes[i, 1].set_title(titles_full[i])
        axes[i, 1].set_xlabel("Epoch")
        axes[i, 1].set_ylabel("Loss")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = SCRIPT_DIR / "learning_curves_v2.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"\nSaved: {curves_path}")

    pred_visc = 0.6 * pred_st_visc + 0.4 * pred_ds_visc
    pred_oxid = (pred_ds_oxid + pred_st_oxid) / 2
    pred_oxid = np.maximum(pred_oxid, 0.0)

    print(f"Viscosity: mean={pred_visc.mean():.2f}, std={pred_visc.std():.2f}")
    print(f"Oxidation: mean={pred_oxid.mean():.2f}, std={pred_oxid.std():.2f}")

    submission = pd.DataFrame(
        {"scenario_id": test_ids, TARGET_VISC: pred_visc, TARGET_OXID: pred_oxid}
    )
    pred_path = SCRIPT_DIR / "predictions.csv"
    submission.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"\nSaved: {pred_path}")


if __name__ == "__main__":
    main()
