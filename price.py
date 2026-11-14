import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Computer Price Predictor", page_icon="💻", layout="wide")
st.title("💻 Computer Price Predictor")
st.caption("Predict price from speed, hd, ram and screen")

FEATURES = ["speed", "hd", "ram", "screen"]
TARGET = "price"
DEFAULT_PATH = "my computer data.csv"  # change if your file has a different name


# ---------- Data loading ----------
@st.cache_data
def load_csv(source):
    df = pd.read_csv(source)
    df.columns = [c.strip().lower() for c in df.columns]
    # drop an unnamed index column if present (e.g. "unnamed: 0")
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]
    return df


st.sidebar.header("1. Load data")
uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
path_input = st.sidebar.text_input("...or file path", DEFAULT_PATH)

df = None
if uploaded is not None:
    df = load_csv(uploaded)
elif os.path.exists(path_input):
    df = load_csv(path_input)

if df is None:
    st.info("Upload your **my computer data** CSV in the sidebar (or enter a valid file path) to begin.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
    st.stop()

df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()

# ---------- Data overview ----------
tab_data, tab_model, tab_predict = st.tabs(["📊 Data", "🤖 Model", "🎯 Predict"])

with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(df.head(50), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Average price", f"{df[TARGET].mean():,.0f}")
    c3.metric("Price range", f"{df[TARGET].min():,.0f} – {df[TARGET].max():,.0f}")
    st.subheader("Summary statistics")
    st.dataframe(df.describe().T, use_container_width=True)
    st.subheader("Correlation with price")
    st.bar_chart(df.corr()[TARGET].drop(TARGET))

# ---------- Model training ----------
st.sidebar.header("2. Model")
model_name = st.sidebar.selectbox("Algorithm", ["Random Forest", "Linear Regression"])
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)


@st.cache_resource
def train(df, model_name, test_size):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": float(np.sqrt(np.mean((y_test - preds) ** 2))),
    }
    return model, metrics, y_test, preds


model, metrics, y_test, preds = train(df, model_name, test_size)

with tab_model:
    st.subheader(f"{model_name} performance (on held-out test data)")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{metrics['r2']:.3f}")
    c2.metric("MAE", f"{metrics['mae']:,.1f}")
    c3.metric("RMSE", f"{metrics['rmse']:,.1f}")

    st.subheader("Actual vs predicted price")
    st.scatter_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": preds}),
                     x="Actual", y="Predicted")

    if model_name == "Random Forest":
        st.subheader("Feature importance")
        st.bar_chart(pd.Series(model.feature_importances_, index=FEATURES))
    else:
        st.subheader("Coefficients")
        st.bar_chart(pd.Series(model.coef_, index=FEATURES))

# ---------- Prediction ----------
with tab_predict:
    st.subheader("Enter computer specs")
    cols = st.columns(4)
    inputs = {}
    for col, feat in zip(cols, FEATURES):
        lo, hi = float(df[feat].min()), float(df[feat].max())
        inputs[feat] = col.number_input(
            feat.upper(), min_value=lo, max_value=hi,
            value=float(df[feat].median()),
        )

    if st.button("Predict price", type="primary"):
        pred = model.predict(pd.DataFrame([inputs]))[0]
        st.success(f"Estimated price: **{pred:,.2f}**")
