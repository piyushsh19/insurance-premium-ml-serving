import os
import requests
import streamlit as st

API_URL = os.getenv(
    "INSURANCE_API_URL",
    "http://127.0.0.1:8000/predict"
)

OCCUPATIONS = [
    "retired",
    "freelancer",
    "student",
    "government_job",
    "business_owner",
    "unemployed",
    "private_job",
]

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🛡️"
)

st.title("Insurance Premium Category Predictor")
st.write("Enter your details below:")

with st.form("prediction_form"):
    age = st.number_input("Age", 0, 120, 30)
    weight = st.number_input("Weight (kg)", min_value=0.1, value=65.0)
    height = st.number_input("Height (m)", 0.1, 2.49, 1.70)
    income_lpa = st.number_input("Annual Income (LPA)", min_value=0.0, value=10.0)

    smoker = st.selectbox("Are you a smoker?", [False, True])
    city = st.text_input("City", "Mumbai")
    occupation = st.selectbox("Occupation", OCCUPATIONS)

    submitted = st.form_submit_button("Predict Premium Category")

if submitted:
    if not city.strip():
        st.error("Please enter a city.")
    else:
        input_data = {
            "age": age,
            "weight": weight,
            "height": height,
            "income_lpa": income_lpa,
            "smoker": smoker,
            "city": city.strip(),
            "occupation": occupation,
        }

        try:
            response = requests.post(
                API_URL,
                json=input_data,
                timeout=15
            )
            response.raise_for_status()

            result = response.json()

            st.success(
                f"Predicted Insurance Premium Category: "
                f"**{result['predicted_category']}**"
            )

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI. Start your backend.")

        except requests.exceptions.HTTPError:
            st.error(f"API Error: {response.text}")

        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")