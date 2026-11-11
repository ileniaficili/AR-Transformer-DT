"""
AR-Transformer-DT: joint estimation of daily canopy transpiration (ET) and
stomatal conductance (g_s) from intra-day meteorological sequences and
lag-1 plant physiology.

Pipeline
  1. Load the plant-level daily records and the 15-min meteorological records.
  2. Chronological split: first 80% of recording days -> training, last 20% -> test.
  3. Min-Max normalisation of meteorological channels, static input and targets.
     --scaler-fit train (default): scalers fitted on the training partition only.
     --scaler-fit full: scalers fitted on the full dataset before the split
     (setting used for the results reported in the paper).
  4. Autoregressive features: ET(t-1) and g_s(t-1) of the same plant.
  5. Model: multi-head self-attention encoder (2 heads, key_dim = 32) +
     layer normalisation + global average pooling + dropout, concatenated with
     the static vector [deficit level, ET(t-1), g_s(t-1)] and followed by a
     shared multi-task head (32 -> 16 -> 2). Total trainable parameters: 2,919.
  6. Training: Adam (lr = 1e-3), batch 32, MSE loss, max 50 epochs,
     early stopping on a 10% validation split (patience 10, best weights restored).
  7. Evaluation on the test set (metrics computed on normalised values).

Usage
  python ar_transformer_dt.py --data-dir <folder with the three CSV files> [--scaler-fit train|full] [--seed 7] [--out results]
"""
import argparse, os, random, json

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, Concatenate, MultiHeadAttention,
                                     LayerNormalization, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Input file names and column names (see README.md for the expected format)
FILE_PLANTS = 'piante_pulite.csv'
FILE_METEO_EXT = 'meteo_esterno_15m.csv'
FILE_METEO_INT = 'meteo_interno_15m.csv'
METEO_COLUMNS = ['Temp_Ext_2m', 'RH_Ext_2m', 'Rad_Solare_Ext',
                 'Temp (°C)', 'RH (%)', 'PARLight (µmol/m<sup>2</sup>/s)', 'VPD (kPa)']
COL_DATE, COL_TREATMENT = 'Data', 'Trt'
COL_DEFICIT = 'Dehydratation rate (%)'
COL_ET, COL_GS = 'Traspirazione', 'gs_canopy'
STEPS_PER_DAY = 96          # 24 h at 15-min resolution
TEST_FRACTION = 0.20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_and_prepare(data_dir, scaler_fit='train'):
    """Load the CSV files and build the model tensors (steps 1-4)."""
    meteo = pd.merge(pd.read_csv(os.path.join(data_dir, FILE_METEO_EXT), parse_dates=['Datetime']),
                     pd.read_csv(os.path.join(data_dir, FILE_METEO_INT), parse_dates=['Datetime']),
                     on='Datetime', how='inner')
    plants = pd.read_csv(os.path.join(data_dir, FILE_PLANTS))
    plants[COL_DATE] = pd.to_datetime(plants[COL_DATE])
    plants = plants.dropna(subset=[COL_ET, COL_GS]).reset_index(drop=True)

    # Chronological split by recording day
    days = sorted(plants[COL_DATE].unique())
    n_test = max(1, int(len(days) * TEST_FRACTION))
    train_days, test_days = set(days[:-n_test]), set(days[-n_test:])

    # Min-Max scalers: fitted on the training partition only ('train')
    # or on the full dataset before the split ('full', as in the paper)
    if scaler_fit == 'full':
        plant_train = np.ones(len(plants), dtype=bool)
        meteo_train = np.ones(len(meteo), dtype=bool)
    else:
        plant_train = plants[COL_DATE].isin(train_days).values
        meteo_train = meteo['Datetime'].dt.normalize().isin(train_days).values
    sc_meteo, sc_static, sc_target = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    sc_meteo.fit(meteo.loc[meteo_train, METEO_COLUMNS])
    sc_static.fit(plants.loc[plant_train, [COL_DEFICIT]])
    sc_target.fit(plants.loc[plant_train, [COL_ET, COL_GS]])
    meteo[METEO_COLUMNS] = sc_meteo.transform(meteo[METEO_COLUMNS])
    plants[COL_DEFICIT] = sc_static.transform(plants[[COL_DEFICIT]])
    plants[[COL_ET, COL_GS]] = sc_target.transform(plants[[COL_ET, COL_GS]])

    # Plant identifier: position of the plant within each day after sorting by
    # treatment (the dataset contains the same plants, in the same order, every day).
    plants = plants.sort_values([COL_DATE, COL_TREATMENT]).reset_index(drop=True)
    plants['plant_id'] = plants.groupby(COL_DATE).cumcount()
    plants = plants.sort_values(['plant_id', COL_DATE]).reset_index(drop=True)

    # Lag-1 physiological features of the same plant (0 for its first day)
    plants['prev_ET'] = plants.groupby('plant_id')[COL_ET].shift(1).fillna(0)
    plants['prev_gs'] = plants.groupby('plant_id')[COL_GS].shift(1).fillna(0)

    # Build tensors: one sample per plant-day with a complete 96-step day
    meteo = meteo.set_index('Datetime')
    T = {k: [] for k in ('m_tr', 's_tr', 'y_tr', 'm_te', 's_te', 'y_te')}
    for _, row in plants.iterrows():
        day = row[COL_DATE]
        try:
            m_day = meteo.loc[day.strftime('%Y-%m-%d')]
        except KeyError:
            continue
        if len(m_day) != STEPS_PER_DAY:
            continue
        part = 'tr' if day in train_days else 'te'
        T['m_' + part].append(m_day[METEO_COLUMNS].values)
        T['s_' + part].append([row[COL_DEFICIT], row['prev_ET'], row['prev_gs']])
        T['y_' + part].append([row[COL_ET], row[COL_GS]])
    return {k: np.asarray(v, dtype='float32') for k, v in T.items()}


def build_model():
    """AR-Transformer-DT architecture (step 5)."""
    meteo_in = Input(shape=(STEPS_PER_DAY, len(METEO_COLUMNS)), name='meteo_sequence')
    x = MultiHeadAttention(num_heads=2, key_dim=32, name='mha')(meteo_in, meteo_in)
    x = LayerNormalization(name='layer_norm')(x)
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)
    x = Dropout(0.2, name='dropout')(x)
    static_in = Input(shape=(3,), name='static_ar')       # [deficit, ET(t-1), g_s(t-1)]
    z = Concatenate(name='fusion')([x, static_in])
    z = Dense(32, activation='relu', name='dense_32')(z)
    z = Dense(16, activation='relu', name='dense_16')(z)
    out = Dense(2, activation='linear', name='output_et_gs')(z)
    model = Model([meteo_in, static_in], out, name='AR_Transformer_DT')
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default='.', help='folder containing the three input CSV files')
    ap.add_argument('--scaler-fit', choices=['train', 'full'], default='train',
                    help="'train': scalers fitted on the training partition only (default); "
                         "'full': scalers fitted on the full dataset (setting used in the paper)")
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='results', help='output folder')
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    T = load_and_prepare(args.data_dir, args.scaler_fit)
    print(f'Training samples: {len(T["y_tr"])} | test samples: {len(T["y_te"])}')

    model = build_model()
    print(f'Trainable parameters: {model.count_params():,}')
    model.fit([T['m_tr'], T['s_tr']], T['y_tr'], validation_split=0.1, epochs=50, batch_size=32,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
              verbose=2)

    y_pred = model.predict([T['m_te'], T['s_te']], verbose=0)
    y_te = T['y_te']
    metrics = {
        'seed': args.seed,
        'scaler_fit': args.scaler_fit,
        'R2_ET': float(r2_score(y_te[:, 0], y_pred[:, 0])),
        'R2_gs': float(r2_score(y_te[:, 1], y_pred[:, 1])),
        'MSE': float(mean_squared_error(y_te, y_pred)),
        'MAE_ET': float(mean_absolute_error(y_te[:, 0], y_pred[:, 0])),
        'MAE_gs': float(mean_absolute_error(y_te[:, 1], y_pred[:, 1])),
        'params': int(model.count_params()),
    }
    print(json.dumps(metrics, indent=2))
    with open(os.path.join(args.out, f'metrics_{args.scaler_fit}_seed{args.seed}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    model.save(os.path.join(args.out, f'ar_transformer_dt_{args.scaler_fit}_seed{args.seed}.keras'))


if __name__ == '__main__':
    main()
