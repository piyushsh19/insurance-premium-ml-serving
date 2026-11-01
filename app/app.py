from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, computed_field


# Resolve the model path from the project root, not the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


# Keep these lists identical to the city lists used during model training.
TIER_1_CITIES = {
    "mumbai",
    "delhi",
    "bangalore",
    "chennai",
    "kolkata",
    "hyderabad",
    "pune",
}

TIER_2_CITIES = {
    "jaipur",
    "chandigarh",
    "indore",
    "lucknow",
    "patna",
    "ranchi",
    "visakhapatnam",
    "coimbatore",
    "bhopal",
    "nagpur",
    "vadodara",
    "surat",
    "rajkot",
    "jodhpur",
    "raipur",
    "amritsar",
    "varanasi",
    "agra",
    "dehradun",
    "mysore",
    "jabalpur",
    "guwahati",
    "thiruvananthapuram",
    "ludhiana",
    "nashik",
    "allahabad",
    "udaipur",
    "aurangabad",
    "hubli",
    "belgaum",
    "salem",
    "vijayawada",
    "tiruchirappalli",
    "bhavnagar",
    "gwalior",
    "dhanbad",
    "bareilly",
    "aligarh",
    "gaya",
    "kozhikode",
    "warangal",
    "kolhapur",
    "bilaspur",
    "jalandhar",
    "noida",
    "guntur",
    "asansol",
    "siliguri",
}


Occupation = Literal[
    "retired",
    "freelancer",
    "student",
    "government_job",
    "business_owner",
    "unemployed",
    "private_job",
]


class UserInput(BaseModel):
    """Original customer fields accepted by the prediction API."""

    model_config = ConfigDict(str_strip_whitespace=True)

    age: Annotated[int, Field(ge=0, le=120, description="Age of the customer")]
    weight: Annotated[
        float, Field(gt=0, description="Weight in kilograms")
    ]
    height: Annotated[
        float, Field(gt=0, lt=2.5, description="Height in metres")
    ]
    income_lpa: Annotated[
        float, Field(ge=0, description="Annual income in lakhs")
    ]
    smoker: bool
    city: Annotated[str, Field(min_length=1, description="Customer's city")]
    occupation: Occupation

    @computed_field
    @property
    def bmi(self) -> float:
        """Calculate BMI from weight (kg) and height (m)."""
        return self.weight / (self.height**2)

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        if self.age < 45:
            return "adult"
        if self.age < 60:
            return "middle_aged"
        return "senior"

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        if self.smoker or self.bmi > 27:
            return "medium"
        return "low"

    @computed_field
    @property
    def city_tier(self) -> int:
        normalized_city = self.city.casefold()

        if normalized_city in TIER_1_CITIES:
            return 1
        if normalized_city in TIER_2_CITIES:
            return 2
        return 3


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained pipeline once when the API starts."""
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Trained model not found at {MODEL_PATH}. "
            "Complete Step 1 and save models/model.joblib first."
        )

    app.state.model = joblib.load(MODEL_PATH)
    yield
    app.state.model = None


app = FastAPI(
    title="Insurance Premium Prediction API",
    description=(
        "Predict an insurance premium category from original customer data "
        "using a trained Scikit-learn pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "message": "Insurance Premium Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok" if model_loaded else "model_not_loaded"}


@app.post("/predict")
def predict_premium(data: UserInput, request: Request):
    """Accept original customer data, engineer features, and predict a category."""

    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The ML model is not loaded. Check the API startup logs.",
        )

    # These names and values must match the features used to train the pipeline.
    input_df = pd.DataFrame(
        [
            {
                "bmi": data.bmi,
                "age_group": data.age_group,
                "lifestyle_risk": data.lifestyle_risk,
                "city_tier": data.city_tier,
                "income_lpa": data.income_lpa,
                "occupation": data.occupation,
            }
        ],
        columns=[
            "bmi",
            "age_group",
            "lifestyle_risk",
            "city_tier",
            "income_lpa",
            "occupation",
        ],
    )

    try:
        prediction = model.predict(input_df)[0]
    except Exception as exc:
        # Keep internal model details in server logs, not in the API response.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API server logs.",
        ) from exc

    return {"predicted_category": str(prediction)}
