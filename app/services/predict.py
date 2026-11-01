from typing import Any

import pandas as pd

from app.schemas.prediction_response import PredictionResponse


MODEL_FEATURES = [
    "bmi",
    "age_group",
    "lifestyle_risk",
    "city_tier",
    "income_lpa",
    "occupation",
]


def predict_insurance_premium(
    model: Any,
    data: Any,
    model_version: str,
) -> PredictionResponse:
    """
    Build the model input from validated customer data and run inference.

    The feature names and feature-engineering rules must remain consistent
    with the training pipeline.
    """
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
        columns=MODEL_FEATURES,
    )

    prediction = str(model.predict(input_df)[0])

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The loaded model does not support predict_proba(). "
            "Train or export a classifier that supports probability estimates."
        )

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    class_probabilities = {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    confidence = class_probabilities.get(prediction)
    if confidence is None:
        raise ValueError(
            "Predicted class was not found in the model's class labels."
        )

    return PredictionResponse(
        predicted_category=prediction,
        confidence=confidence,
        class_probabilities=class_probabilities,
        model_version=model_version,
    )
