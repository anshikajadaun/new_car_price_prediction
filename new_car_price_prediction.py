import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AutoBloom | Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(
            135deg,
            #F7F3FF 0%,
            #EDF7FF 55%,
            #FFF2E8 100%
        );
        color: #26345C;
    }

    h1, h2, h3 {
        color: #344477 !important;
    }

    p, label, span {
        color: #344477;
    }

    .hero {
        background: linear-gradient(
            110deg,
            #DCD6FF,
            #D8EEFF,
            #FFE4D6
        );
        padding: 35px;
        border-radius: 25px;
        text-align: center;
        border: 1px solid #FFFFFF;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        color: #344477;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 17px;
        color: #4C5980;
    }

    .section-card {
        background: rgba(255,255,255,0.85);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #DDE4FA;
        margin-bottom: 18px;
    }

    div.stButton > button {
        background: linear-gradient(
            90deg,
            #8274D9,
            #65A9E8
        );
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px;
        font-size: 17px;
        font-weight: bold;
        width: 100%;
    }

    div.stButton > button:hover {
        color: white;
        border: 1px solid #FFFFFF;
        background: linear-gradient(
            90deg,
            #6E60C8,
            #4A95D8
        );
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #DFE5FA;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px;
        background-color: #FFFFFF;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🚘 AutoBloom</h1>
    <p>Your smart companion for estimating car prices.</p>
    <p>✨ Machine Learning • Multiple Features • Smart Prediction</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("new_car_price_prediction.csv")


try:
    df = load_data()
except Exception:
    st.error(
        "Dataset load nahi hui. GitHub par "
        "new_car_price_prediction.csv filename check karo."
    )
    st.stop()


# ---------------- VALIDATE DATA ----------------
required_columns = [
    "name",
    "year",
    "seats",
    "selling_price"
]

if not all(col in df.columns for col in required_columns):
    st.error(
        "CSV mein name, year, seats aur selling_price "
        "columns hone chahiye."
    )
    st.stop()


# ---------------- PREPARE DATA ----------------
data = df[required_columns].copy()

data["name"] = data["name"].astype("string").str.strip()

for col in ["year", "seats", "selling_price"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.replace([np.inf, -np.inf], np.nan)

data = data.dropna(
    subset=["name", "year", "seats", "selling_price"]
)

data = data[
    (data["name"] != "") &
    (data["selling_price"] >= 0) &
    (data["seats"] > 0)
]

data = data.drop_duplicates()

if len(data) < 5:
    st.error("Model train karne ke liye sufficient clean data nahi hai.")
    st.stop()


# ---------------- TRAIN MODEL ----------------
X = data[["name", "year", "seats"]].copy()
y = data["selling_price"].copy()

# Convert car names into numerical features
X = pd.get_dummies(
    X,
    columns=["name"],
    dtype=int
)

# Keep the same feature columns for prediction
feature_columns = X.columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

# Model evaluation
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.15, 0.85], gap="large")


# ---------------- INPUT SECTION ----------------
with left:
    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("🌷 Tell Us About Your Car")
    st.write("Fill in the details below to estimate its price.")

    car_names = sorted(
        data["name"].dropna().unique().tolist()
    )

    selected_car = st.selectbox(
        "🚗 Select Car Model",
        car_names
    )

    min_year = int(data["year"].min())
    max_year = int(data["year"].max())

    selected_year = st.number_input(
        "📅 Manufacturing Year",
        min_value=min_year,
        max_value=max_year,
        value=min(max_year, max(min_year, 2017)),
        step=1
    )

    min_seats = max(1, int(data["seats"].min()))
    max_seats = int(data["seats"].max())

    selected_seats = st.number_input(
        "💺 Number of Seats",
        min_value=min_seats,
        max_value=max_seats,
        value=min(max_seats, max(min_seats, 5)),
        step=1
    )

    predict_button = st.button(
        "✨ Predict My Car Price"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- INFORMATION SECTION ----------------
with right:
    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("💜 About AutoBloom")

    st.write(
        "AutoBloom uses Multiple Linear Regression "
        "to estimate a car's selling price."
    )

    st.markdown("##### 🌟 Features used")
    st.write("🚘 Car Model")
    st.write("📅 Manufacturing Year")
    st.write("💺 Number of Seats")

    st.markdown("##### 🧠 How it works")
    st.write(
        "1. Select your car details.\n\n"
        "2. The trained model processes the inputs.\n\n"
        "3. Get the estimated selling price."
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- PREDICTION ----------------
if predict_button:

    # Start with zero values for all encoded features
    new_car = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Set numerical features
    new_car["year"] = selected_year
    new_car["seats"] = selected_seats

    # Set the selected car's encoded feature
    car_column = f"name_{selected_car}"

    if car_column in new_car.columns:
        new_car.loc[0, car_column] = 1

    predicted_price = model.predict(new_car)[0]

    st.markdown("---")
    st.subheader("🎉 Your Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Estimated Selling Price",
            f"₹ {predicted_price:,.0f}"
        )

    with result_col2:
        st.metric(
            "Mean Absolute Error",
            f"₹ {mae:,.0f}"
        )

    if predicted_price < 0:
        st.warning(
            "The model produced a negative price. "
            "This may indicate that the model needs improvement."
        )
    else:
        st.success(
            "Your estimated car price is ready! "
            "Actual market prices may differ."
        )

    st.caption(
        "Note: This is a machine learning estimate, "
        "not a guaranteed market price."
    )


# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:12px;">
        <h4 style="color:#8274D9;">🌸 AutoBloom</h4>
        <p>Your journey to smarter car price estimation.</p>
        <p>Built with Python, Streamlit & Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)
