import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import math

# ---- 1. Load Data (sheet 06 pivot Anda)
df = pd.read_excel("Data_FullProcess_v6_Preprocessed_v2.xlsx",
                   sheet_name="06_Pivot_Prediksi")

df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)

bahan = "Kopi"   # <--- ubah sesuai bahan


# ---- 2. Buat fitur waktu
df["Day_Index"] = np.arange(len(df))
df["DayOfWeek"] = df["Tanggal"].dt.weekday
df["IsWeekend"] = df["DayOfWeek"].isin([5,6]).astype(int)
df["Month"] = df["Tanggal"].dt.month
df["IsMonthStart"] = (df["Tanggal"].dt.day == 1).astype(int)
df["IsMonthEnd"] = (df["Tanggal"].dt.is_month_end).astype(int)

# ---- 3. Buat MA7 NON-LEAK (shift)
df["MA7"] = df[bahan].rolling(7).mean().shift(1)

# ---- 4. Bersihkan NaN
df_model = df.dropna(subset=["MA7"]).reset_index(drop=True)

# ---- 5. Siapkan fitur & target
feature_cols = ["Day_Index", "DayOfWeek", "IsWeekend",
                "Month", "IsMonthStart", "IsMonthEnd"]

X = df_model[feature_cols].values
y = df_model["MA7"].values   # <--- target adalah MA7

# ---- 6. Train-test split 80/20
n = len(df_model)
split = int(n * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---- 7. Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# ---- 8. Prediksi test
y_pred_test = model.predict(X_test)
y_pred_test = np.maximum(0, y_pred_test)

mae = mean_absolute_error(y_test, y_pred_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))

print("MAE:", mae)
print("RMSE:", rmse)


# ---- 9. Prediksi untuk seluruh histori
df_model["MA7_Pred"] = np.maximum(0, model.predict(X))

# ---- 10. Forecast 14 hari ke depan
def forecast_future_ma7(df_model, model, bahan, horizon=14):
    last_date = df_model["Tanggal"].iloc[-1]
    last_day_index = df_model["Day_Index"].iloc[-1]

    future_rows = []

    for i in range(1, horizon+1):
        tgl = last_date + timedelta(days=i)
        di = last_day_index + i

        dow = tgl.weekday()
        is_wknd = 1 if dow in [5,6] else 0
        month = tgl.month
        is_ms = 1 if tgl.day == 1 else 0
        is_me = 1 if (tgl + timedelta(days=1)).month != month else 0

        x_input = np.array([di, dow, is_wknd, month, is_ms, is_me]).reshape(1,-1)

        y_pred = max(0, model.predict(x_input)[0])

        future_rows.append({
            "Tanggal": tgl,
            "MA7_Pred": y_pred
        })

    return pd.DataFrame(future_rows)

df_future = forecast_future_ma7(df_model, model, bahan)

# ---- 11. Konsumsi harian dari MA7 prediksi
df_future["Pred_Konsumsi_Harian"] = df_future["MA7_Pred"]

print(df_future.head(10))
import sys
try:
    import streamlit as st
except Exception:
    st = None

if st is not None:
    st.set_page_config(page_title="Prediksi Konsumsi Bahan (MA7)", layout="wide")
    st.title("☕ Prediksi Konsumsi Bahan — MA7 + Linear Regression")
    st.write("Upload Excel atau gunakan default untuk melihat metrik, grafik, dan forecast.")

    col_top1, col_top2 = st.columns([3, 2])
    with col_top2:
        st.markdown("### Pengaturan")
        sheet_name = st.text_input("Nama Sheet", value="06_Pivot_Prediksi")
        horizon_days = st.number_input("Horizon Prediksi (hari)", min_value=7, max_value=60, value=14, step=1)

    uploaded = st.file_uploader("Upload file Excel", type=["xlsx"]) 
    if uploaded is None:
        default_path = "Data_FullProcess_v6_Preprocessed_v2.xlsx"
        df2 = pd.read_excel(default_path, sheet_name=sheet_name)
        st.info(f"Memakai file default: {default_path}")
    else:
        df2 = pd.read_excel(uploaded, sheet_name=sheet_name)

    df2["Tanggal"] = pd.to_datetime(df2["Tanggal"])
    df2 = df2.sort_values("Tanggal").reset_index(drop=True)

    numeric_cols = [c for c in df2.columns if c not in ["Tanggal"] and pd.api.types.is_numeric_dtype(df2[c])]
    bahan_sel = st.selectbox("Pilih bahan", options=numeric_cols, index=(numeric_cols.index("Kopi") if "Kopi" in numeric_cols else 0))

    df2["Day_Index"] = np.arange(len(df2))
    df2["DayOfWeek"] = df2["Tanggal"].dt.weekday
    df2["IsWeekend"] = df2["DayOfWeek"].isin([5,6]).astype(int)
    df2["Month"] = df2["Tanggal"].dt.month
    df2["IsMonthStart"] = (df2["Tanggal"].dt.day == 1).astype(int)
    df2["IsMonthEnd"] = (df2["Tanggal"].dt.is_month_end).astype(int)

    df2["MA7"] = df2[bahan_sel].rolling(7).mean().shift(1)
    df2_model = df2.dropna(subset=["MA7"]).reset_index(drop=True)

    feature_cols2 = ["Day_Index", "DayOfWeek", "IsWeekend", "Month", "IsMonthStart", "IsMonthEnd"]
    X2 = df2_model[feature_cols2].values
    y2 = df2_model["MA7"].values

    n2 = len(df2_model)
    split2 = int(n2 * 0.8)
    X2_train, X2_test = X2[:split2], X2[split2:]
    y2_train, y2_test = y2[:split2], y2[split2:]

    model2 = LinearRegression()
    model2.fit(X2_train, y2_train)

    y2_pred_test = np.maximum(0, model2.predict(X2_test))
    mae2 = mean_absolute_error(y2_test, y2_pred_test)
    rmse2 = math.sqrt(mean_squared_error(y2_test, y2_pred_test))

    df2_model["MA7_Pred"] = np.maximum(0, model2.predict(X2))

    df2_future = forecast_future_ma7(df2_model, model2, bahan_sel, horizon=int(horizon_days))
    df2_future["Pred_Konsumsi_Harian"] = df2_future["MA7_Pred"]

    with st.container():
        st.subheader(f"Metrik Akurasi untuk: {bahan_sel}")
        c1, c2 = st.columns(2)
        c1.metric("MAE (Test)", f"{mae2:.4f}")
        c2.metric("RMSE (Test)", f"{rmse2:.4f}")

    with st.tabs(["Historis", "Forecast", "Tabel", "Export"])[0]:
        hist_plot = df2_model[["Tanggal", "MA7", "MA7_Pred"]].set_index("Tanggal")
        st.line_chart(hist_plot)

    with st.tabs(["Historis", "Forecast", "Tabel", "Export"])[1]:
        future_plot = df2_future[["Tanggal", "MA7_Pred"]].set_index("Tanggal")
        st.line_chart(future_plot)

    with st.tabs(["Historis", "Forecast", "Tabel", "Export"])[2]:
        st.dataframe(df2_future.rename(columns={"MA7_Pred": f"{bahan_sel}_Pred"}))

    with st.tabs(["Historis", "Forecast", "Tabel", "Export"])[3]:
        csv_hist = df2_model[["Tanggal", "MA7", "MA7_Pred"]].to_csv(index=False).encode("utf-8")
        csv_future = df2_future.to_csv(index=False).encode("utf-8")
        colA, colB = st.columns(2)
        colA.download_button("Download Historis+Pred (CSV)", data=csv_hist, file_name=f"Hist_{bahan_sel}.csv", mime="text/csv")
        colB.download_button("Download Forecast (CSV)", data=csv_future, file_name=f"Forecast_{bahan_sel}.csv", mime="text/csv")
