import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="Crop Profit Predictor", layout="centered")

st.title("🌾 Crop Profit Prediction App")
st.write("Predict **Net Profit per Hectare** based on **Cost of Cultivation**")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cost_profit.csv")

df = load_data()

# Features & target
X = df[["Cost"]]
y = df["Profit"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ("Linear Regression", "Random Forest Regressor")
)

cost_input = st.sidebar.number_input(
    "Cost of Cultivation (₹ per hectare)",
    min_value=0,
    value=100000,
    step=5000
)


# Prediction
if st.sidebar.button("Predict Profit"):
    if model_choice == "Linear Regression":
        prediction = lr_model.predict([[cost_input]])
        model_used = "Linear Regression"
    else:
        prediction = rf_model.predict([[cost_input]])
        model_used = "Random Forest Regressor"

    st.success(f"✅ Model Used: {model_used}")
    st.metric(
        label="Predicted Net Profit (₹ per hectare)",
        value=f"₹ {int(prediction[0]):,}"
    )

# Show dataset
with st.expander("📊 View Training Data"):
    st.dataframe(df)

# Footer
st.markdown("---")
st.caption("⚠️ Predictions are indicative. Accuracy improves with more features & data.")
st.success("Done")