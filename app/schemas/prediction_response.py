from typing import Dict
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):

    predicted_category: str = Field(
        ...,
        description="Predicted insurance premium category",
        examples=["High"]
    )

    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability assigned to the predicted class",
        examples=[0.8432]
    )

    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all classes",
        examples=[{
            "Low": 0.01,
            "Medium": 0.1468,
            "High": 0.8432
        }]
    )

    model_version: str = Field(
        default="1.0.0",
        description="Version of the trained ML model"
    )