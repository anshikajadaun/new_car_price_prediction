import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AutoBloom | Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# ---------------- DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F3EEFF, #E8F6FF, #FFF1E8);
    color: #344477;
}
.block-container {
    max-width: 850px;
    padding-top: 1rem;
    padding-bottom: 1rem;
}
h1, h2, h3, p, label {
    color: #344477 !important;
}
div.stButton > button {
    background: linear-gradient(90deg, #8274D9, #65A9E8);
    color: white;
    border: none;
    border-radius: 10px;
    height: 44px;
    font-weight: bold;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DCD6FF;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>🚗 AutoBloom</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>"
    "Smart Used Car Price Prediction ✨"
    "</p>",
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("new_car_price_prediction.csv")

try:
    df = load_data()
except Exception:
    st.error(
        "Dataset nahi mili. GitHub par "
        "new_car_price_prediction.csv check karo."
    )
    st.stop()

# ---------------- REQUIRED COLUMNS ----------------
features = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats"
]

target = "selling_price"

missing_columns = [
    col for col in features + [target]
    if col not in df.columns
]

if missing_columns:
    st.error(f"Dataset mein columns missing hain: {missing_columns}")
    st.stop()

# ---------------- CLEAN DATA ----------------
data = df[features + [target]].copy()

# Numeric columns
numeric_columns = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
    "selling_price"
]

# Extract numeric values from text such as
# "1248 CC", "74 bhp", "22.0 km/kg", "190 Nm"
for col in numeric_columns:
    data[col] = pd.to_numeric(
        data[col].astype(str).str.extract(
            r"([-+]?\d*\.?\d+)",
            expand=False
        ),
        errors="coerce"
    )

# Clean categorical columns
categorical_columns = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

for col in categorical_columns:
    data[col] = (
        data[col]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

# Remove rows without target price
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=[target])
data = data[data[target] >= 0]
data = data.drop_duplicates()

if len(data) < 10:
    st.error("Model train karne ke liye sufficient data nahi hai.")
    st.stop()

# ---------------- TRAIN MODEL ----------------
X = data[features]
y = data[target]

numeric_features = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats"
]

categorical_features = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model.fit(x_train, y_train)

# Evaluate model on held-out test data
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ---------------- USER INPUT ----------------
st.markdown("### ✨ Enter Your Car Details")

col1, col2 = st.columns(2)

car_names = sorted(
    data["name"].dropna().unique().tolist()
)

fuel_options = sorted(
    data["fuel"].dropna().unique().tolist()
)

seller_options = sorted(
    data["seller_type"].dropna().unique().tolist()
)

transmission_options = sorted(
    data["transmission"].dropna().unique().tolist()
)

owner_options = sorted(
    data["owner"].dropna().unique().tolist()
)

if not all([
    car_names,
    fuel_options,
    seller_options,
    transmission_options,
    owner_options
]):
    st.error("Dataset mein categorical options available nahi hain.")
    st.stop()

with col1:
    car_name = st.selectbox("🚘 Car Model", car_names)

    year = st.number_input(
        "📅 Manufacturing Year",
        min_value=1900,
        max_value=2100,
        value=int(data["year"].median()),
        step=1
    )

    km_driven = st.number_input(
        "🛣️ Kilometres Driven",
        min_value=0,
        max_value=2000000,
        value=30000,
        step=1000
    )

    fuel = st.selectbox("⛽ Fuel Type", fuel_options)

    transmission = st.selectbox(
        "⚙️ Transmission",
        transmission_options
    )

    owner = st.selectbox("👤 Previous Owner", owner_options)

with col2:
    seller_type = st.selectbox("🏷️ Seller Type", seller_options)

    mileage = st.number_input(
        "⛽ Mileage (km/litre)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=0.5
    )

    engine = st.number_input(
        "🔧 Engine Capacity (CC)",
        min_value=0.0,
        max_value=10000.0,
        value=1200.0,
        step=100.0
    )

    max_power = st.number_input(
        "⚡ Maximum Power (bhp)",
        min_value=0.0,
        max_value=2000.0,
        value=75.0,
        step=5.0
    )

    torque = st.number_input(
        "🔩 Torque (Nm)",
        min_value=0.0,
        max_value=5000.0,
        value=100.0,
        step=10.0
    )

    seats = st.selectbox(
        "💺 Number of Seats",
        sorted(
            int(s) for s in data["seats"].dropna().unique()
            if s > 0 and float(s).is_integer()
        )
    )

# ---------------- PREDICT ----------------
if st.button("✨ Predict Car Price", use_container_width=True):

    new_car = pd.DataFrame([{
        "name": car_name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "torque": torque,
        "seats": seats
    }], columns=features)

    predicted_price = model.predict(new_car)[0]

    st.markdown("### 💜 Prediction Result")

    st.metric(
        label="Estimated Selling Price",
        value=f"₹ {predicted_price:,.0f}"
    )

    st.caption(
        "Prediction is an estimate. Actual market prices may differ."
    )

    if predicted_price < 0:
        st.warning(
            "Model ne negative price predict ki hai. "
            "Dataset aur model performance check karo."
        )

    with st.expander("📊 Model Performance"):
        st.write(f"**Mean Absolute Error (MAE):** ₹{mae:,.2f}")
        st.write(f"**R² Score:** {r2:.3f}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:12px;'>"
    "🌸 AutoBloom | Python • Streamlit • Machine Learning"
    "</p>",
    unsafe_allow_html=True
)
