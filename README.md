Insurance Premium Prediction — ML Model Training
A hands-on machine learning project that trains a classification model to predict an insurance premium category (Low, Medium, or High) from customer information. This repository is being built in stages: model training first, FastAPI model serving next, and a Streamlit frontend after that.
Repository name
Recommended: insurance-premium-ml-serving

Short description: Train and serve an insurance premium classification model using Scikit-learn, FastAPI, and Streamlit.
Project status
- [x] Step 1 — Train and save the machine learning model
- [ ] Step 2 — Expose predictions through a FastAPI POST /predict endpoint
- [ ] Step 3 — Build a Streamlit frontend that consumes the API
This README currently documents Step 1 only. The API and frontend will be documented when those stages are implemented.
Step 1 — Model training
Goal
Train a Scikit-learn classification pipeline that predicts the insurance premium category from customer data. The trained pipeline includes preprocessing and a RandomForestClassifier, and is saved as a single model artifact for reuse.
Dataset
Place the dataset at:
data/insurance.csv
The notebook expects these source columns:
Column	Description	Example
age	Customer age	32
weight	Weight in kilograms	75.5
height	Height in metres	1.75
income_lpa	Annual income in lakhs	12.5
smoker	Whether the customer smokes	False
city	Customer's city	Mumbai
occupation	Customer's occupation	private_job
insurance_premium_category	Target label	Medium


The target column is insurance_premium_category, with the expected classes Low, Medium, and High.
The dataset is not included in this repository by default. Add your own insurance.csv file before running the notebook. Do not commit private or sensitive customer data.

Folder structure
insurance-premium-ml-serving/
├── data/
│   └── insurance.csv
├── notebooks/
│   └── model_training.ipynb
├── models/
│   └── model.joblib
├── training/
│   ├── __init__.py
│   ├── features.py
│   └── train.py
├── app/                       # Step 2: FastAPI (planned)
│   ├── __init__.py
│   ├── app.py
│   ├── schemas/
│   └── services/
├── frontend/                  # Step 3: Streamlit (planned)
│   └── streamlit_app.py
├── tests/                     # Tests (planned)
├── .gitignore
├── pyproject.toml
└── README.md
For the current training stage, the main working file is notebooks/model_training.ipynb. The training/, app/, frontend/, and tests/ directories provide a structure for the later stages; they do not need implementation yet.
Environment setup
This project uses uv to manage Python dependencies.
1. Create and enter the project
mkdir insurance-premium-ml-serving
cd insurance-premium-ml-serving
uv init
If you already created the project, simply open its directory and continue.
2. Install the Step 1 dependencies
uv add pandas numpy scikit-learn joblib
uv add --dev jupyter ipykernel
3. Create the working directories
mkdir -p data notebooks models training
touch training/__init__.py
4. Add the dataset and notebook
- Copy insurance.csv into data/.
- Create notebooks/model_training.ipynb.
- Run Jupyter from the repository root:
uv run jupyter notebook
Training workflow
The notebook follows this sequence.
1. Load and inspect the data
Read the CSV with pandas and inspect the shape, data types, summary statistics, missing values, and target distribution.
import pandas as pd

df = pd.read_csv("../data/insurance.csv")

print(df.shape)
df.info()
print(df.isnull().sum())
print(df["insurance_premium_category"].value_counts())
2. Feature engineering
Create additional features from the original customer fields:
- BMI: weight / height ** 2
- Age group: young, adult, middle_aged, or senior
- Lifestyle risk: high, medium, or low, based on the example's smoking and BMI rules
- City tier: 1, 2, or 3, based on the city lists defined in the notebook
These are example business rules for this learning project. They are not medically or actuarially validated rules.
3. Select model inputs and target
The engineered model inputs are:
feature_columns = [
    "bmi",
    "age_group",
    "lifestyle_risk",
    "city_tier",
    "income_lpa",
    "occupation",
]

X = df_model[feature_columns]
y = df_model["insurance_premium_category"]
The original columns replaced by engineered features (age, weight, height, smoker, and city) are not passed directly to the classifier.
4. Preprocess the features
Categorical columns are one-hot encoded. Numerical columns are passed through unchanged.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = [
    "age_group",
    "lifestyle_risk",
    "occupation",
    "city_tier",
]

numeric_features = ["bmi", "income_lpa"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("num", "passthrough", numeric_features),
    ]
)
5. Build and train the pipeline
The preprocessing transformer and Random Forest classifier are combined into one Scikit-learn pipeline.
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
)

pipeline.fit(X_train, y_train)
6. Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
Accuracy is only one measure of model quality. Review precision, recall, and F1-score for each class as well. A small dataset or a single train/test split can produce an unstable score; use more data and cross-validation before drawing conclusions about real-world performance.
7. Save and reload the trained pipeline
Save the complete pipeline so that the same preprocessing is applied when making predictions later.
import joblib

joblib.dump(pipeline, "../models/model.joblib")

# Verify that the saved artifact can be loaded.
loaded_model = joblib.load("../models/model.joblib")
print(loaded_model.predict(X_test.head()))
The expected output artifact is:
models/model.joblib
Important: Only load joblib/pickle model files from trusted sources. Such files can execute code during deserialization.
Important note for Step 2 — FastAPI
The saved pipeline expects the six engineered features:
- bmi
- age_group
- lifestyle_risk
- city_tier
- income_lpa
- occupation
The planned API will accept the original customer fields (age, weight, height, income_lpa, smoker, city, and occupation), validate them, derive the engineered features using the same rules as training, and then call the saved pipeline.

Training and API feature-engineering logic must stay consistent. In the next stage, these transformations should be moved into a shared Python module so that the notebook and API do not maintain separate, drifting implementations.
Tech stack
- Python
- pandas and NumPy
- Scikit-learn
- joblib
- uv for dependency and environment management
- Jupyter Notebook
FastAPI and Streamlit are planned for later project stages.
Disclaimer
This is an educational machine learning project. Predictions are based on the supplied dataset and illustrative feature-engineering rules. They must not be used to determine real insurance premiums or make actual underwriting decisions.