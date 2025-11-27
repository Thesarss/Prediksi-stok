import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import timedelta
import os
import glob
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

# ===============================================
# CONFIG
# ===============================================
st.set_page_config(
    page_title="Prediksi Stok & Restock Kedai Kopi Ole",
    layout="wide"
)

_BASE_DIR = os.path.dirname(__file__)
EXCEL_CANDIDATES = [
    os.path.join(_BASE_DIR, "KopiOle_Forecast_Master_6M_Pipeline_UPDATED.xlsx"),
    os.path.join(_BASE_DIR, "KopiOle_Forecast_Master_6M_Pipeline.xlsx"),
    os.path.join(_BASE_DIR, "KopiOle_Forecast_Master.xlsx"),
]

SHEET_MODELREADY = "07b_ModelReady_MA7Lag"
SHEET_AGG = "05_Agregasi_Harian"
SHEET_STOK = "00_Stok_Info"
SHEET_CONV = "21_Conversion_Factors"
SHEET_REAL = "22_Cons_Real_Harian"

TARGETS = ["Kopi", "Susu", "Teh", "Milo", "Sirup", "Indomie", "Jeruk", "Lainnya"]


# ===============================================
# METRIC HELPERS
# ===============================================
def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0


# ===============================================
# LOAD DATA
# ===============================================
@st.cache_data
def load_data():
    last_err = None
    candidates = []
    for p in EXCEL_CANDIDATES:
        if p not in candidates:
            candidates.append(p)
    for p in sorted(glob.glob(os.path.join(_BASE_DIR, "KopiOle_Forecast_Master*.xlsx"))):
        if p not in candidates:
            candidates.append(p)

    for path in candidates:
        try:
            df_model = pd.read_excel(path, sheet_name=SHEET_MODELREADY)
            df_model["Tanggal"] = pd.to_datetime(df_model["Tanggal"])

            df_agg = pd.read_excel(path, sheet_name=SHEET_AGG)
            df_agg["Tanggal"] = pd.to_datetime(df_agg["Tanggal"])

            stok_info = pd.read_excel(path, sheet_name=SHEET_STOK)

            conv = pd.read_excel(path, sheet_name=SHEET_CONV)
            conv = conv.set_index("Bahan")

            cons_real = pd.read_excel(path, sheet_name=SHEET_REAL)
            cons_real["Tanggal"] = pd.to_datetime(cons_real["Tanggal"])

            return df_model, df_agg, stok_info, conv, cons_real, path
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError("File Excel master tidak ditemukan/terbaca: " + str(last_err))


# ===============================================
# TRAIN MODELS (Linear, Ridge, Lasso + pilih terbaik)
# ===============================================
@st.cache_resource
def train_models(df_model: pd.DataFrame, _version: int = 0):
    """
    - Split time series: 70% train, 20% val, 10% test (chronological)
    - Kandidat model: LinearRegression, Ridge(alpha grid), Lasso(alpha grid)
    - Pilih model dengan SMAPE terendah pada validation
    - Retrain model terbaik di (train + val)
    - Hitung metrik di test set
    """

    TARGETS = ["Kopi", "Susu", "Teh", "Milo", "Sirup", "Indomie", "Jeruk", "Lainnya"]
    feature_cols = [c for c in df_model.columns if c not in ["Tanggal"] + TARGETS]

    X = df_model[feature_cols].values
    y_all = df_model[TARGETS]

    n = len(df_model)
    if n < 20:
        raise RuntimeError("Data terlalu sedikit untuk split train/val/test")

    idx_train_end = int(0.7 * n)
    idx_val_end = int(0.9 * n)

    # ---------- imputasi & scaling di TRAIN saja ----------
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X[:idx_train_end])
    X_val = imputer.transform(X[idx_train_end:idx_val_end])
    X_test = imputer.transform(X[idx_val_end:])

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # Untuk in-sample prediction seluruh periode
    X_all_sc = scaler.transform(imputer.transform(X))

    def smape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        mask = denom != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0

    models = {}
    metrics = {}
    models_by_algo = {}
    metrics_all = {}

    # Grid hyperparameter
    ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    lasso_alphas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    for bahan in TARGETS:
        y = y_all[bahan].values.astype(float)

        y_train = y[:idx_train_end]
        y_val = y[idx_train_end:idx_val_end]
        y_test = y[idx_val_end:]

        mask_train = ~np.isnan(y_train)
        mask_val = ~np.isnan(y_val)
        mask_test = ~np.isnan(y_test)

        if mask_train.sum() == 0 or mask_val.sum() == 0:
            # tidak cukup data untuk bahan ini
            continue

        best_algo = None
        best_model_val = None
        best_smape_val = np.inf
        lin_model_val = None
        ridge_model_val = None
        lasso_model_val = None
        ridge_best_alpha = None
        lasso_best_alpha = None
        lin_smape = np.inf
        ridge_smape = np.inf
        lasso_smape = np.inf

        # ---------- 1) Linear Regression ----------
        lin = LinearRegression()
        lin.fit(X_train_sc[mask_train], y_train[mask_train])
        y_val_pred = np.maximum(lin.predict(X_val_sc[mask_val]), 0)
        smape_val = smape(y_val[mask_val], y_val_pred)
        lin_model_val = lin
        lin_smape = smape_val
        best_algo = "LinearRegression"
        best_model_val = lin
        best_smape_val = smape_val

        # ---------- 2) Ridge (tuning alpha) ----------
        for alpha in ridge_alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train_sc[mask_train], y_train[mask_train])
            y_val_pred = np.maximum(model.predict(X_val_sc[mask_val]), 0)
            smape_val = smape(y_val[mask_val], y_val_pred)
            if smape_val < ridge_smape:
                ridge_smape = smape_val
                ridge_model_val = model
                ridge_best_alpha = alpha
            if smape_val < best_smape_val:
                best_smape_val = smape_val
                best_algo = f"Ridge(alpha={alpha})"
                best_model_val = model

        # ---------- 3) Lasso (tuning alpha) ----------
        for alpha in lasso_alphas:
            model = Lasso(alpha=alpha, max_iter=20000)
            model.fit(X_train_sc[mask_train], y_train[mask_train])
            y_val_pred = np.maximum(model.predict(X_val_sc[mask_val]), 0)
            smape_val = smape(y_val[mask_val], y_val_pred)
            if smape_val < lasso_smape:
                lasso_smape = smape_val
                lasso_model_val = model
                lasso_best_alpha = alpha
            if smape_val < best_smape_val:
                best_smape_val = smape_val
                best_algo = f"Lasso(alpha={alpha})"
                best_model_val = model

        # ---------- 4) Retrain best model di TRAIN+VAL ----------
        X_trainval_sc = np.vstack([X_train_sc, X_val_sc])
        y_trainval = np.concatenate([y_train, y_val])
        mask_trainval = ~np.isnan(y_trainval)

        final_model = type(best_model_val)(**getattr(best_model_val, "get_params")())
        final_model.fit(X_trainval_sc[mask_trainval], y_trainval[mask_trainval])
        final_lin = None
        final_ridge = None
        final_lasso = None
        if lin_model_val is not None:
            final_lin = type(lin_model_val)(**lin_model_val.get_params())
            final_lin.fit(X_trainval_sc[mask_trainval], y_trainval[mask_trainval])
        if ridge_model_val is not None:
            final_ridge = type(ridge_model_val)(**ridge_model_val.get_params())
            final_ridge.fit(X_trainval_sc[mask_trainval], y_trainval[mask_trainval])
        if lasso_model_val is not None:
            final_lasso = type(lasso_model_val)(**lasso_model_val.get_params())
            final_lasso.fit(X_trainval_sc[mask_trainval], y_trainval[mask_trainval])

        # ---------- 5) Hitung metrik di TEST ----------
        algo_metrics = {}
        if mask_test.sum() > 0:
            y_true_t = y_test[mask_test]
            def calc_metrics(pred):
                pred = np.maximum(pred, 0)
                ae = np.abs(y_true_t - pred)
                se = (y_true_t - pred) ** 2
                mae = float(ae.mean())
                rmse = float(np.sqrt(se.mean()))
                mape_v = float(
                    np.mean(np.abs((y_true_t - pred) / np.where(y_true_t == 0, 1, y_true_t))) * 100.0
                )
                smape_v = float(smape(y_true_t, pred))
                ss_res = float(np.sum((y_true_t - pred) ** 2))
                ss_tot = float(np.sum((y_true_t - np.mean(y_true_t)) ** 2))
                r2_v = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else np.nan
                return mae, rmse, mape_v, smape_v, r2_v
            mae, rmse, mape_val, smape_test, r2 = calc_metrics(final_model.predict(X_test_sc[mask_test]))
            if final_lin is not None:
                m_lin = calc_metrics(final_lin.predict(X_test_sc[mask_test]))
                algo_metrics["LinearRegression"] = {
                    "MAE_test": m_lin[0], "RMSE_test": m_lin[1], "MAPE_test": m_lin[2], "SMAPE_test": m_lin[3], "R2_test": m_lin[4]
                }
            if final_ridge is not None:
                m_ridge = calc_metrics(final_ridge.predict(X_test_sc[mask_test]))
                algo_metrics["Ridge"] = {
                    "MAE_test": m_ridge[0], "RMSE_test": m_ridge[1], "MAPE_test": m_ridge[2], "SMAPE_test": m_ridge[3], "R2_test": m_ridge[4]
                }
            if final_lasso is not None:
                m_lasso = calc_metrics(final_lasso.predict(X_test_sc[mask_test]))
                algo_metrics["Lasso"] = {
                    "MAE_test": m_lasso[0], "RMSE_test": m_lasso[1], "MAPE_test": m_lasso[2], "SMAPE_test": m_lasso[3], "R2_test": m_lasso[4]
                }
        else:
            mae = rmse = mape_val = smape_test = r2 = np.nan

        models[bahan] = final_model
        models_by_algo[bahan] = {
            "LinearRegression": final_lin,
            "Ridge": final_ridge,
            "Lasso": final_lasso,
        }
        metrics_all[bahan] = algo_metrics
        metrics[bahan] = {
            "algo": best_algo,
            "MAE_test": mae,
            "RMSE_test": rmse,
            "MAPE_test": mape_val,
            "SMAPE_test": smape_test,
            "R2_test": r2,
            "SMAPE_val": float(best_smape_val),
        }

    splits = {
        "idx_train_end": idx_train_end,
        "idx_val_end": idx_val_end,
        "n": n,
    }

    return models, scaler, feature_cols, imputer, metrics, splits, X_all_sc, models_by_algo, metrics_all
# ===============================================
# IN-SAMPLE PREDICTION (pakai model terbaik per bahan)
# ===============================================
def predict_in_sample(df_model: pd.DataFrame, _models, _X_all_scaled):
    pred_df = pd.DataFrame({"Tanggal": df_model["Tanggal"]})
    pred_df.set_index("Tanggal", inplace=True)

    for bahan in TARGETS:
        if bahan in _models:
            y_hat = _models[bahan].predict(_X_all_scaled)
            y_hat = np.maximum(y_hat, 0)
            pred_df[bahan] = y_hat
        else:
            pred_df[bahan] = np.nan
    return pred_df


# ===============================================
# FORECAST UNIT (PORSI) N HARI KE DEPAN
# ===============================================
def forecast_next_days_units(df_model, models, scaler, feature_cols, imputer, n_days=14):
    last_date = df_model["Tanggal"].iloc[-1]
    last_known = df_model.tail(1)[feature_cols].iloc[0]

    last_di = df_model["Day_Index"].iloc[-1] if "Day_Index" in df_model.columns else None
    future_rows = []
    dates = []

    for step in range(1, int(n_days) + 1):
        tgl = last_date + timedelta(days=step)
        row = last_known.copy()

        if "Day_Index" in feature_cols:
            base_di = int(last_di) if last_di is not None else int(step)
            row["Day_Index"] = base_di + step
        if "DayOfWeek" in feature_cols:
            row["DayOfWeek"] = int(tgl.weekday())
        if "Month" in feature_cols:
            row["Month"] = int(tgl.month)
        if "IsWeekend" in feature_cols:
            row["IsWeekend"] = int(1 if tgl.weekday() in [5, 6] else 0)
        if "IsMonthStart" in feature_cols:
            row["IsMonthStart"] = int(1 if tgl.day == 1 else 0)
        if "IsMonthEnd" in feature_cols:
            row["IsMonthEnd"] = int(1 if (tgl + timedelta(days=1)).month != tgl.month else 0)
        if "WeekOfYear" in feature_cols:
            row["WeekOfYear"] = int(tgl.isocalendar().week)
        if "Quarter" in feature_cols:
            row["Quarter"] = int(((tgl.month - 1) // 3) + 1)

        future_rows.append(row.to_frame().T)
        dates.append(tgl)

    df_future = pd.concat(future_rows, ignore_index=True)
    df_future = df_future[feature_cols]

    X_future_imp = imputer.transform(df_future)
    X_future_scaled = scaler.transform(X_future_imp)

    forecast_dict = {}
    for bahan in TARGETS:
        if bahan not in models:
            forecast_dict[bahan] = [0.0] * len(dates)
            continue
        y_pred = models[bahan].predict(X_future_scaled)
        y_pred = np.maximum(y_pred, 0)
        forecast_dict[bahan] = y_pred.tolist()

    forecast_df = pd.DataFrame(forecast_dict, index=dates)
    forecast_df.index.name = "Tanggal"
    return forecast_df


# ===============================================
# KONVERSI UNIT ➜ STOK REAL (kg/kaleng/bungkus)
# ===============================================
def convert_to_real(forecast_units, conv):
    factors = conv["Factor_Real_per_Unit"]
    real_df = forecast_units.copy()
    for b in TARGETS:
        factor = factors.get(b, np.nan)
        if np.isnan(factor):
            real_df[b] = np.nan
        else:
            real_df[b] = forecast_units[b] * factor
    return real_df


# ===============================================
# SIMULASI STOK REAL & TANGGAL RESTOCK
# ===============================================
def simulate_stock_real(forecast_real, stok_awal, safety_stock=None):
    hasil = []

    for bahan in TARGETS:
        total_pred = forecast_real[bahan].sum(skipna=True)
        if np.isnan(total_pred):
            hasil.append({
                "Bahan": bahan,
                "Satuan": "-",
                "Stok Awal (Real)": stok_awal.get(bahan, 0.0),
                "Total Prediksi Pemakaian (Real)": np.nan,
                "Sisa di Akhir Horizon (Real)": np.nan,
                "Tanggal Melewati Safety Stock": "-",
                "Tanggal Habis (<=0)": "-",
            })
            continue

        stok = stok_awal.get(bahan, 0.0)
        ss = safety_stock.get(bahan, 0.0) if safety_stock else 0.0

        tanggal_ss = None
        tanggal_habis = None

        for tanggal, row in forecast_real.iterrows():
            pemakaian = row[bahan]
            if np.isnan(pemakaian):
                continue
            stok -= pemakaian

            if tanggal_ss is None and stok <= ss:
                tanggal_ss = tanggal
            if stok <= 0 and tanggal_habis is None:
                tanggal_habis = tanggal
                break

        sisa = stok
        hasil.append({
            "Bahan": bahan,
            "Satuan": "-",
            "Stok Awal (Real)": stok_awal.get(bahan, 0.0),
            "Total Prediksi Pemakaian (Real)": round(total_pred, 3),
            "Sisa di Akhir Horizon (Real)": round(sisa, 3),
            "Tanggal Melewati Safety Stock": tanggal_ss.date() if tanggal_ss is not None else "-",
            "Tanggal Habis (<=0)": tanggal_habis.date() if tanggal_habis is not None else "-",
        })

    return pd.DataFrame(hasil)


# ===============================================
# MA7 dari df_agg (untuk visualisasi Data Understanding)
# ===============================================
def compute_ma7_from_df_agg(df_agg: pd.DataFrame):
    df_ma = df_agg.copy()
    for b in TARGETS:
        if b in df_ma.columns:
            df_ma[f"{b}_MA7"] = df_ma[b].rolling(window=7, min_periods=1).mean()
    return df_ma

def get_cons_real_wide(cons_real: pd.DataFrame):
    df = cons_real.copy()
    df["Tanggal"] = pd.to_datetime(df["Tanggal"]) if "Tanggal" in df.columns else df.index
    if "Bahan" in df.columns:
        value_cols = [c for c in df.columns if c not in ["Tanggal", "Bahan"] and is_numeric_dtype(df[c])]
        col = value_cols[0] if value_cols else None
        if col is not None:
            pv = df.pivot_table(index="Tanggal", columns="Bahan", values=col, aggfunc="sum")
            return pv.sort_index()
    cols = [c for c in df.columns if c in TARGETS and is_numeric_dtype(df[c])]
    if cols:
        return df.set_index("Tanggal")[cols].sort_index()
    return pd.DataFrame()


# ===============================================
# MAIN UI
# ===============================================
def main():
    st.title("📦 Dashboard Prediksi Stok & Restock – Kedai Kopi Ole")

    # Load
    try:
        df_model, df_agg, stok_info, conv, cons_real, active_path = load_data()
    except Exception as e:
        st.error(f"Gagal membaca file Excel: {e}")
        st.stop()

    if "train_version" not in st.session_state:
        st.session_state["train_version"] = 0
    if st.sidebar.button("Train ulang"):
        st.session_state["train_version"] += 1
        st.experimental_rerun()

    models, scaler, feature_cols, imputer, metrics, splits, X_all_scaled, models_by_algo, metrics_all = train_models(df_model, _version=st.session_state["train_version"])
    in_sample_pred = predict_in_sample(df_model, models, X_all_scaled)

    # Sidebar konfigurasi
    st.sidebar.header("⚙️ Pengaturan Umum")
    st.sidebar.write(f"File aktif: `{os.path.basename(active_path)}`")
    st.sidebar.write(f"Periode data: **{df_agg['Tanggal'].min().date()}** s.d. **{df_agg['Tanggal'].max().date()}**")
    horizon = st.sidebar.slider("Horizon prediksi (hari ke depan)", 7, 30, 14, 1)

    # Stok awal & safety stock
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Stok Awal (Real)")
    default_stok = {
        "Kopi": float(stok_info.loc[stok_info["Nama Barang"] == "Kopi", "Restock Size"].sum() or 15),
        "Susu": float(stok_info.loc[stok_info["Nama Barang"] == "Susu", "Restock Size"].sum() or 72),
        "Teh": float(stok_info.loc[stok_info["Nama Barang"].str.contains("Teh", case=False), "Restock Size"].sum() or 5),
        "Milo": float(stok_info.loc[stok_info["Nama Barang"] == "Milo", "Restock Size"].sum() or 1),
        "Sirup": float(stok_info.loc[stok_info["Nama Barang"].str.contains("Sirup", case=False), "Restock Size"].sum() or 1),
        "Indomie": float(stok_info.loc[stok_info["Nama Barang"].str.contains("Indomie", case=False), "Restock Size"].sum() or 24),
        "Jeruk": float(stok_info.loc[stok_info["Nama Barang"] == "Jeruk", "Restock Size"].sum() or 0.5),
        "Lainnya": 0.0,
    }

    stok_awal = {}
    safety_stock = {}
    for b in TARGETS:
        stok_awal[b] = st.sidebar.number_input(
            f"Stok awal {b}",
            min_value=0.0,
            value=float(default_stok.get(b, 0.0)),
            step=1.0
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Safety Stock (Real)")
    for b in TARGETS:
        safety_stock[b] = st.sidebar.number_input(
            f"Safety stock {b}",
            min_value=0.0,
            value=0.0,
            step=0.5
        )

    # Kalibrasi magnitude prediksi
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Kalibrasi Magnitude Prediksi (Unit)")
    use_cal = st.sidebar.checkbox("Aktifkan kalibrasi otomatis", value=True)
    win_cal = st.sidebar.slider("Window baseline (hari)", 3, 30, 7, 1)
    max_scale = st.sidebar.slider("Batas skala x", 1, 50, 10, 1)

    # Forecast
    algo_choice = st.sidebar.radio("Algoritma Forecast", ["Auto (Terbaik)", "LinearRegression", "Ridge", "Lasso"], index=0)
    use_models = models
    if algo_choice != "Auto (Terbaik)":
        use_models = {}
        for b in TARGETS:
            m = models_by_algo.get(b, {}).get(algo_choice)
            if m is not None:
                use_models[b] = m
    forecast_units = forecast_next_days_units(df_model, use_models, scaler, feature_cols, imputer, n_days=horizon)

    # Kalibrasi ke konsumsi real akhir
    if use_cal:
        last_row = df_model.tail(1)[feature_cols]
        X_last_imp = imputer.transform(last_row)
        X_last_s = scaler.transform(X_last_imp)

        baseline_map = {}
        for b in TARGETS:
            if b in df_agg.columns:
                baseline_map[b] = float(df_agg[b].tail(win_cal).mean())
            else:
                baseline_map[b] = None

        for b in TARGETS:
            if b in models and baseline_map[b] is not None:
                y_hat_last = float(np.maximum(models[b].predict(X_last_s)[0], 1e-6))
                scale = float(baseline_map[b] / y_hat_last) if y_hat_last != 0 else 1.0
                scale = float(np.clip(scale, 1.0 / max_scale, max_scale))
                forecast_units[b] = forecast_units[b] * scale

    forecast_real = convert_to_real(forecast_units, conv)
    stok_sim = simulate_stock_real(forecast_real, stok_awal, safety_stock)
    df_ma = compute_ma7_from_df_agg(df_agg)

    # ===============================================
    # TAB LAYOUT
    # ===============================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1️⃣ Overview",
        "2️⃣ Pola & MA7",
        "3️⃣ Evaluasi Model",
        "4️⃣ Prediksi & Stok Real",
        "5️⃣ Simulasi Stok & Restock",
        "6️⃣ Faktor & Data Teknis",
        "7️⃣ Data Mentah"
    ])

    # ---------- TAB 1: OVERVIEW ----------
    with tab1:
        st.header("📊 Overview Konsumsi Bahan Baku")

        bahan_list = st.multiselect(
            "Pilih bahan yang ditampilkan:",
            options=TARGETS,
            default=["Kopi", "Susu", "Milo"]
        )

        if bahan_list:
            st.subheader("Tren Konsumsi Harian (30 hari terakhir) – Unit Porsi")
            n_hist = 30
            hist_tail = df_agg.set_index("Tanggal").tail(n_hist)[bahan_list]
            st.line_chart(hist_tail)

            st.subheader("Rata-rata Konsumsi per Minggu – Unit Porsi")
            df_weekly = df_agg.copy()
            df_weekly["Minggu"] = df_weekly["Tanggal"].dt.isocalendar().week
            weekly_mean = df_weekly.groupby("Minggu")[bahan_list].mean()
            st.bar_chart(weekly_mean)

            st.markdown("#### Ringkasan Angka")
            summary = df_agg[bahan_list].agg(["sum", "mean", "max"]).T
            summary.columns = ["Total 3 Bulan", "Rata-rata Harian", "Maksimum Harian"]
            st.dataframe(summary)

            st.subheader("Distribusi Penggunaan per Bahan")
            basis_dist = st.radio("Basis", ["Unit", "Real"], index=0, key="dist_basis")
            sumber_dist = st.radio("Sumber", ["Historis", "Prediksi"], index=0, key="dist_src")
            n_dist = st.slider("Jumlah hari untuk distribusi", 7, 120, 30, 1, key="dist_days")

            if sumber_dist == "Historis":
                tail_df = df_agg.set_index("Tanggal").tail(n_dist)
                cols = [b for b in bahan_list if b in tail_df.columns]
                if basis_dist == "Unit":
                    dist_series = tail_df[cols].sum().sort_values(ascending=False)
                else:
                    factors = conv["Factor_Real_per_Unit"]
                    present = [b for b in cols if b in factors.index]
                    real_df = tail_df[present].copy()
                    for b in present:
                        real_df[b] = real_df[b] * float(factors.get(b, np.nan))
                    dist_series = real_df.sum().sort_values(ascending=False)
            else:
                if basis_dist == "Unit":
                    cols = [b for b in bahan_list if b in forecast_units.columns]
                    dist_series = forecast_units[cols].sum().sort_values(ascending=False)
                else:
                    cols = [b for b in bahan_list if b in forecast_real.columns]
                    dist_series = forecast_real[cols].sum().sort_values(ascending=False)

            dist_df = dist_series.to_frame("Total")
            st.bar_chart(dist_df)
            total_sum = float(dist_series.sum()) if len(dist_series) else 0.0
            share = (dist_series / total_sum * 100.0).round(2) if total_sum != 0 else dist_series
            st.dataframe(pd.DataFrame({"Total": dist_series.round(3), "Share (%)": share}).rename_axis("Bahan"))
        else:
            st.info("Pilih minimal satu bahan untuk ditampilkan.")

    # ---------- TAB 2: Pola & MA7 ----------
    with tab2:
        st.header("📈 Analisis Pola Konsumsi dan MA7")

        bahan = st.selectbox("Pilih bahan:", TARGETS, index=0)
        n_hist2 = st.slider("Jumlah hari historis ditampilkan", 14, 120, 60, 1)

        if bahan in df_agg.columns:
            df_temp = df_ma[["Tanggal", bahan, f"{bahan}_MA7"]].set_index("Tanggal").tail(n_hist2)

            st.subheader(f"Konsumsi Harian vs MA7 – {bahan}")
            st.line_chart(df_temp.rename(columns={bahan: "Raw", f"{bahan}_MA7": "MA7"}))

            st.subheader(f"Distribusi Konsumsi Harian – {bahan}")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(df_agg[bahan].dropna(), bins=15)
            ax.set_xlabel("Unit per hari")
            ax.set_ylabel("Frekuensi")
            ax.set_title(f"Histogram Konsumsi Harian – {bahan}")
            st.pyplot(fig)

        else:
            st.info("Bahan tidak tersedia di data agregasi.")

    # ---------- TAB 3: Evaluasi Model ----------
    with tab3:
        st.header("📉 Evaluasi Model (Linear, Ridge, Lasso)")

        bahan_eval = st.selectbox("Pilih bahan untuk evaluasi:", TARGETS, index=0, key="eval_bahan")
        n_hist_eval = st.slider("Jumlah hari historis ditampilkan", 14, 120, 60, 1, key="eval_hist")

        if bahan_eval in in_sample_pred.columns and bahan_eval in df_model.columns:
            df_eval = pd.DataFrame({
                "Tanggal": df_model["Tanggal"],
                "Actual": df_model[bahan_eval].values,
                "Pred": in_sample_pred[bahan_eval].values
            }).set_index("Tanggal").tail(n_hist_eval)

            st.subheader(f"MA7 Aktual vs Prediksi – {bahan_eval}")
            st.line_chart(df_eval)

            mae_is = float(np.mean(np.abs(df_eval["Actual"] - df_eval["Pred"])))
            rmse_is = float(np.sqrt(np.mean((df_eval["Actual"] - df_eval["Pred"]) ** 2)))
            mape_is = float(mape(df_eval["Actual"], df_eval["Pred"]))
            smape_is = float(smape(df_eval["Actual"], df_eval["Pred"]))
            y_true_is = df_eval["Actual"].values
            y_pred_is = df_eval["Pred"].values
            ss_res_is = float(np.sum((y_true_is - y_pred_is) ** 2))
            ss_tot_is = float(np.sum((y_true_is - np.mean(y_true_is)) ** 2))
            r2_is = float(1.0 - ss_res_is / ss_tot_is) if ss_tot_is != 0 else np.nan

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MAE (In-sample)", f"{mae_is:.2f}")
            c2.metric("RMSE (In-sample)", f"{rmse_is:.2f}")
            c3.metric("MAPE (In-sample)", f"{mape_is:.2f}%")
            c4.metric("SMAPE (In-sample)", f"{smape_is:.2f}%")
            c5.metric("R² (In-sample)", f"{r2_is:.2f}")

            st.subheader("Perbandingan Algoritma (In-sample)")
            cmp_models = models_by_algo.get(bahan_eval, {})
            df_cmp = pd.DataFrame({"Tanggal": df_model["Tanggal"], "Actual": df_model[bahan_eval].values}).set_index("Tanggal").tail(n_hist_eval)
            if cmp_models.get("LinearRegression") is not None:
                df_cmp["Pred Linear"] = cmp_models["LinearRegression"].predict(X_all_scaled)[-n_hist_eval:]
            if cmp_models.get("Ridge") is not None:
                df_cmp["Pred Ridge"] = cmp_models["Ridge"].predict(X_all_scaled)[-n_hist_eval:]
            if cmp_models.get("Lasso") is not None:
                df_cmp["Pred Lasso"] = cmp_models["Lasso"].predict(X_all_scaled)[-n_hist_eval:]
            df_cmp = df_cmp.applymap(lambda v: max(v, 0) if isinstance(v, (int, float, np.floating)) else v)
            st.line_chart(df_cmp)

            m_all = metrics_all.get(bahan_eval, {})
            table_rows = []
            for algo in ["LinearRegression", "Ridge", "Lasso"]:
                info = m_all.get(algo)
                if info:
                    table_rows.append({
                        "Algoritma": algo,
                        "MAE (Test)": info["MAE_test"],
                        "RMSE (Test)": info["RMSE_test"],
                        "MAPE (Test)": info["MAPE_test"],
                        "SMAPE (Test)": info["SMAPE_test"],
                        "R2 (Test)": info["R2_test"],
                    })
            if table_rows:
                st.dataframe(pd.DataFrame(table_rows).set_index("Algoritma").round(3))

            # Parity plot
            st.subheader("Parity Plot (Actual vs Predicted)")
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.scatter(df_eval["Actual"], df_eval["Pred"], alpha=0.7)
            min_v = min(df_eval["Actual"].min(), df_eval["Pred"].min())
            max_v = max(df_eval["Actual"].max(), df_eval["Pred"].max())
            ax2.plot([min_v, max_v], [min_v, max_v], "r--")
            ax2.set_xlabel("Actual MA7")
            ax2.set_ylabel("Predicted MA7")
            ax2.set_title(f"Parity Plot – {bahan_eval}")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        st.subheader("Ringkasan Metrik (Test Set)")
        if metrics:
            rows = []
            for b, info in metrics.items():
                rows.append({
                    "Bahan": b,
                    "Algoritma Terpilih": info["algo"],
                    "MAE (Test)": info["MAE_test"],
                    "RMSE (Test)": info["RMSE_test"],
                    "MAPE (Test)": info["MAPE_test"],
                    "SMAPE (Test)": info["SMAPE_test"],
                    "R2 (Test)": info.get("R2_test", np.nan),
                    "SMAPE (Validasi)": info["SMAPE_val"],
                })
            df_metrics = pd.DataFrame(rows).set_index("Bahan")
            st.dataframe(df_metrics.round(3))
        else:
            st.info("Belum ada metrik yang dapat dihitung.")

    # ---------- TAB 4: Prediksi & Stok Real ----------
    with tab4:
        st.header("🔮 Prediksi Konsumsi & Stok Real")

        bahan_fore = st.selectbox("Pilih bahan untuk forecast:", TARGETS, index=0, key="fore_bahan")

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.subheader(f"Prediksi Konsumsi (Unit Porsi) – {bahan_fore}")
            if bahan_fore in forecast_units.columns:
                st.line_chart(forecast_units[[bahan_fore]])
            else:
                st.info("Prediksi belum tersedia untuk bahan ini.")

        with col_f2:
            st.subheader(f"Prediksi Pemakaian Stok Real – {bahan_fore}")
            if bahan_fore in forecast_real.columns:
                st.line_chart(forecast_real[[bahan_fore]])
            else:
                st.info("Konversi stok belum tersedia untuk bahan ini.")

        st.subheader(f"Tabel Prediksi Stok Real per Hari – Horizon {horizon} Hari")
        table_real = forecast_real.copy().reset_index()
        st.dataframe(table_real.style.format(precision=3))

    # ---------- TAB 5: Simulasi Stok & Restock ----------
    with tab5:
        st.header("📆 Simulasi Stok & Rekomendasi Restock")

        st.subheader("Ringkasan Simulasi per Bahan")
        table_stok = stok_sim.copy()
        table_stok["Tanggal Melewati Safety Stock"] = table_stok["Tanggal Melewati Safety Stock"].astype(str)
        table_stok["Tanggal Habis (<=0)"] = table_stok["Tanggal Habis (<=0)"].astype(str)
        st.dataframe(
            table_stok.style.format({
                "Stok Awal (Real)": "{:.3f}",
                "Total Prediksi Pemakaian (Real)": "{:.3f}",
                "Sisa di Akhir Horizon (Real)": "{:.3f}",
            })
        )

        st.subheader("Grafik Sisa Stok vs Hari")
        bahan_sim = st.selectbox("Pilih bahan:", TARGETS, index=0, key="sim_bahan")

        if bahan_sim in forecast_real.columns:
            stok0 = stok_awal.get(bahan_sim, 0.0)
            ss = safety_stock.get(bahan_sim, 0.0)
            df_sr = forecast_real[[bahan_sim]].copy()
            df_sr["Sisa_Stok"] = stok0 - df_sr[bahan_sim].cumsum()

            fig3, ax3 = plt.subplots(figsize=(7, 3))
            ax3.plot(df_sr.index, df_sr["Sisa_Stok"], label="Sisa stok (simulasi)")
            ax3.axhline(ss, color="orange", linestyle="--", label="Safety stock")
            ax3.axhline(0, color="red", linestyle="--", label="Habis")
            ax3.set_xlabel("Tanggal")
            ax3.set_ylabel("Sisa stok (real)")
            ax3.set_title(f"Simulasi Sisa Stok – {bahan_sim}")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            st.pyplot(fig3)
        else:
            st.info("Tidak ada data simulasi stok untuk bahan ini.")

    # ---------- TAB 6: Faktor & Data Teknis ----------
    with tab6:
        st.header("🧮 Faktor Konversi & Data Teknis")

        st.subheader("Tabel Faktor Konversi Unit ➜ Stok Real")
        st.dataframe(
            conv[["Avg_Units_per_Day", "Real_Usage_per_Day_from_Stock", "Factor_Real_per_Unit"]]
            .rename(columns={
                "Avg_Units_per_Day": "Rata-rata Unit/Hari",
                "Real_Usage_per_Day_from_Stock": "Pemakaian Real/Hari (dari stok)",
                "Factor_Real_per_Unit": "Konversi (Real per 1 Unit)"
            })
            .style.format(precision=5)
        )

        st.subheader("Contoh Data ModelReady (07b)")
        st.dataframe(df_model.head(20))

        st.subheader("Contoh Data Agregasi Harian (05_Agregasi_Harian)")
        st.dataframe(df_agg.head(20))

    with tab7:
        st.header("📄 Visualisasi Data Mentah – Cons_Real_Harian")
        rw = get_cons_real_wide(cons_real)
        if rw.empty:
            st.info("Data mentah tidak memiliki format yang dapat divisualisasikan.")
        else:
            bahan_raw = st.multiselect("Pilih bahan", options=list(rw.columns), default=list(rw.columns[:3]))
            n_days_raw = st.slider("Jumlah hari historis", 7, 180, 60, 1)
            tail_rw = rw.tail(n_days_raw)
            if bahan_raw:
                st.subheader("Garis Waktu Pemakaian Harian")
                st.line_chart(tail_rw[bahan_raw])
                st.subheader("Total Penggunaan – Per Bahan")
                dist = tail_rw[bahan_raw].sum().sort_values(ascending=False).to_frame("Total")
                st.bar_chart(dist)
            st.subheader("Cuplikan Data Mentah")
            st.dataframe(cons_real.head(50))


if __name__ == "__main__":
    main()
