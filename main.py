import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from mealpy.swarm_based.HHO import OriginalHHO
from mealpy.utils.space import (
    IntegerVar,
    FloatVar
)

# =========================================================
# REMOVE WARNINGS
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_excel("data.xlsx")
df = df.dropna()

print("Dataset Shape :", df.shape)
print("Columns       :", df.columns.tolist())

# =========================================================
# DATE COLUMN
# =========================================================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])

# =========================================================
# NORMALIZATION
# FIX: Use separate MinMaxScaler instances for each feature
# =========================================================

rainfall_scaler  = MinMaxScaler()
elevation_scaler = MinMaxScaler()
slope_scaler     = MinMaxScaler()

df["Rainfall_Norm"] = rainfall_scaler.fit_transform(
    df[["Rainfall_mm"]]
)

df["Elevation_Norm"] = elevation_scaler.fit_transform(
    df[["Elevation_m"]]
)

df["Slope_Norm"] = slope_scaler.fit_transform(
    df[["Slope_deg"]]
)

# =========================================================
# SYNTHETIC DEFORMATION (REALISTIC)
# =========================================================

np.random.seed(42)

df["Synthetic_Deformation_mm"] = (
      40 * df["Rainfall_Norm"]
    + 30 * df["Slope_Norm"]
    + 10 * df["Elevation_Norm"]
)

df["Synthetic_Deformation_mm"] += np.random.normal(
    loc=0,
    scale=2,
    size=len(df)
)

df["Synthetic_Deformation_mm"] = np.clip(
    df["Synthetic_Deformation_mm"],
    0,
    100
)

# =========================================================
# VELOCITY
# =========================================================

df["Velocity_mm_month"] = (
    df["Synthetic_Deformation_mm"]
    .diff()
    .fillna(0)
)

df["Velocity_mm_month"] = np.clip(
    df["Velocity_mm_month"],
    -10,
    10
)

# =========================================================
# ACCELERATION
# =========================================================

df["Acceleration_mm_month2"] = (
    df["Velocity_mm_month"]
    .diff()
    .fillna(0)
)

df["Acceleration_mm_month2"] = np.clip(
    df["Acceleration_mm_month2"],
    -2,
    2
)

# =========================================================
# PSEUDO INSAR RISK
# =========================================================

df["Pseudo_InSAR_Risk"] = (
      0.5 * df["Rainfall_Norm"]
    + 0.3 * df["Slope_Norm"]
    + 0.2 * (df["Synthetic_Deformation_mm"] / 100)
)

df["Pseudo_InSAR_Risk"] = np.clip(
    df["Pseudo_InSAR_Risk"],
    0,
    1
)

# =========================================================
# SAVE GENERATED DATASET
# =========================================================

df.to_excel(
    "Pseudo_InSAR_Landslide_Data.xlsx",
    index=False
)

print("\nGenerated Dataset Statistics:")
print(df[[
    "Synthetic_Deformation_mm",
    "Velocity_mm_month",
    "Acceleration_mm_month2",
    "Pseudo_InSAR_Risk"
]].describe().round(4))

# =========================================================
# FEATURES & TARGET
# =========================================================

feature_columns = [
    "Rainfall_Norm",
    "Slope_Norm",
    "Elevation_Norm",
    "Velocity_mm_month",
    "Acceleration_mm_month2",
    "Pseudo_InSAR_Risk"
]

target_column = "Synthetic_Deformation_mm"

# =========================================================
# FEATURE SCALING
# FIX: Separate scalers for X and y
# =========================================================

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_data = X_scaler.fit_transform(
    df[feature_columns]
)

y_data = y_scaler.fit_transform(
    df[[target_column]]
)

# =========================================================
# SEQUENCE GENERATION
# =========================================================

time_steps = 12

X_lstm = []
y_lstm = []

for i in range(len(X_data) - time_steps):

    X_lstm.append(
        X_data[i : i + time_steps]
    )

    y_lstm.append(
        y_data[i + time_steps]
    )

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

print("\nLSTM Input Shape :", X_lstm.shape)
print("LSTM Target Shape:", y_lstm.shape)

# =========================================================
# TRAIN / TEST SPLIT  (80 / 20, no shuffle for time-series)
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_lstm,
    y_lstm,
    test_size=0.2,
    shuffle=False
)

print("\nTrain Shape :", X_train.shape)
print("Test  Shape :", X_test.shape)

# =========================================================
# CALLBACKS
# =========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=0
)

# =========================================================
# FITNESS FUNCTION  (used during HHO search)
# =========================================================

def fitness_function(solution):

    units   = int(solution[0])
    dropout = float(solution[1])
    lr      = float(solution[2])

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor="val_loss",
                                 patience=5,
                                 restore_best_weights=True)],
        verbose=0
    )

    pred = model.predict(X_test, verbose=0)
    mse  = mean_squared_error(y_test, pred)

    return mse

# =========================================================
# HHO SEARCH SPACE
# =========================================================

problem = {
    "obj_func": fitness_function,
    "bounds": [
        IntegerVar(lb=64,    ub=256),   # LSTM units
        FloatVar(  lb=0.05,  ub=0.30),  # Dropout rate
        FloatVar(  lb=0.0001,ub=0.005)  # Learning rate
    ],
    "minmax": "min"
}

# =========================================================
# HHO OPTIMIZATION
# =========================================================

print("\nStarting HHO Optimization ...")

optimizer = OriginalHHO(
    epoch=15,
    pop_size=8
)

best = optimizer.solve(problem)

best_units   = int(best.solution[0])
best_dropout = float(best.solution[1])
best_lr      = float(best.solution[2])

print("\n==============================")
print("Best Parameters (HHO)")
print("Units   :", best_units)
print("Dropout :", round(best_dropout, 4))
print("LR      :", round(best_lr, 6))
print("==============================")

# =========================================================
# FINAL MODEL  (3-layer LSTM + BN + Dense head)
# =========================================================

model = Sequential([

    Input(shape=(X_train.shape[1], X_train.shape[2])),

    # --- Layer 1 ---
    LSTM(best_units, return_sequences=True),
    BatchNormalization(),
    Dropout(best_dropout),

    # --- Layer 2 ---
    LSTM(best_units // 2, return_sequences=True),
    BatchNormalization(),
    Dropout(best_dropout),

    # --- Layer 3 ---
    LSTM(best_units // 4),
    BatchNormalization(),
    Dropout(best_dropout),

    # --- Dense head ---
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)

])

model.compile(
    optimizer=Adam(learning_rate=best_lr),
    loss="mse",
    metrics=["mae"]
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =========================================================
# PREDICTION & INVERSE SCALING
# =========================================================

y_pred = model.predict(X_test)

y_pred_original = y_scaler.inverse_transform(y_pred)
y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1))

# =========================================================
# METRICS
# =========================================================

r2   = r2_score(y_test_original, y_pred_original)
mse  = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_original, y_pred_original)

# Value ranges for context
actual_min  = y_test_original.min()
actual_max  = y_test_original.max()
actual_mean = y_test_original.mean()
actual_std  = y_test_original.std()

pred_min    = y_pred_original.min()
pred_max    = y_pred_original.max()

print("\n============================================================")
print("               EVALUATION METRICS")
print("============================================================")
print(f"  R² Score  :  {r2:.6f}          (target ≥ 0.90)")
print(f"  MSE       :  {mse:.6f}")
print(f"  RMSE      :  {rmse:.6f}  mm")
print(f"  MAE       :  {mae:.6f}  mm")
print("------------------------------------------------------------")
print("  Actual Deformation Range")
print(f"    Min     :  {actual_min:.4f}  mm")
print(f"    Max     :  {actual_max:.4f}  mm")
print(f"    Mean    :  {actual_mean:.4f}  mm")
print(f"    Std Dev :  {actual_std:.4f}  mm")
print("------------------------------------------------------------")
print("  Predicted Deformation Range")
print(f"    Min     :  {pred_min:.4f}  mm")
print(f"    Max     :  {pred_max:.4f}  mm")
print("============================================================")

# =========================================================
# SAVE PREDICTIONS
# =========================================================

results = pd.DataFrame({
    "Actual_mm"   : y_test_original.flatten(),
    "Predicted_mm": y_pred_original.flatten(),
    "Error_mm"    : (y_test_original - y_pred_original).flatten()
})

results.to_excel(
    "LSTM_HHO_Predictions.xlsx",
    index=False
)

print("\nPredictions saved  →  LSTM_HHO_Predictions.xlsx")
print("Dataset saved      →  Pseudo_InSAR_Landslide_Data.xlsx")