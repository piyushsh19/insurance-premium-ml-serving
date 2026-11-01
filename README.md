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

## ## Step 2 — FastAPI Model Serving

### Goal

Expose the trained Scikit-learn model through a REST API using FastAPI. The API accepts original customer information, performs feature engineering, and returns the predicted insurance premium category.

### Architecture

```text
Client Request (Original Customer Data)
              |
              v
       FastAPI Endpoint
          POST /predict
              |
              v
       Pydantic Validation
           UserInput
              |
              v
       Feature Engineering
       - BMI
       - Age Group
       - Lifestyle Risk
       - City Tier
              |
              v
       Pandas DataFrame
              |
              v
       Trained ML Pipeline
         model.joblib
              |
              v
       Random Forest Model
              |
              v
       Prediction Response
       Low / Medium / High
```

### Project Structure

```text
app/
├── __init__.py
└── app.py
```

### Dependencies

Install the required dependencies using `uv`:

```bash
uv add fastapi "uvicorn[standard]" pandas scikit-learn joblib
```

### Implementation Details

**1. FastAPI Application**

Created the FastAPI application with application metadata and lifecycle management.

**2. Pydantic Input Validation**

Implemented a `UserInput` model using `BaseModel`, `Annotated`, `Field`, and `Literal` to validate incoming customer information.

Original input fields:

- `age`
- `weight`
- `height`
- `income_lpa`
- `smoker`
- `city`
- `occupation`

**3. Computed Features**

Used Pydantic's `@computed_field` and `@property` decorators to calculate features automatically:

| Feature | Calculation |
|---|---|
| BMI | Weight / Height² |
| Age Group | Based on customer age |
| Lifestyle Risk | Based on smoking status and BMI |
| City Tier | Based on predefined city classifications |

**4. Model Loading**

Used `joblib.load()` to load the trained pipeline from `models/model.joblib`.

The model is loaded once during FastAPI application startup using the lifespan context manager.

**5. Prediction Endpoint**

Implemented:

```http
POST /predict
```

The endpoint:
- Accepts original customer data as JSON.
- Validates the request using Pydantic.
- Calculates engineered features.
- Creates a pandas DataFrame.
- Passes the DataFrame to the trained ML pipeline.
- Returns the predicted insurance premium category.

### Sample API Request

```json
{
  "age": 32,
  "weight": 75.5,
  "height": 1.75,
  "income_lpa": 12.5,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "private_job"
}
```

### Sample API Response

```json
{
  "predicted_category": "Medium"
}
```

*The prediction above is illustrative. The actual result depends on the trained model.*

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Check API and model status |
| POST | `/predict` | Predict insurance premium category |

### Run the Application

Start the FastAPI development server from the project root:

```bash
uv run uvicorn app.app:app --reload
```

API base URL:

```text
http://127.0.0.1:8000
```

### API Documentation

FastAPI automatically generates interactive Swagger documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

Use Swagger UI to test the `/predict` endpoint by submitting customer information.

### Concepts Covered

- FastAPI application and REST endpoints
- Pydantic BaseModel and request validation
- Annotated, Field and Literal
- Pydantic computed fields
- Python properties and decorators
- FastAPI lifespan events
- Loading serialized ML models using joblib
- Pandas DataFrame creation
- Integrating Scikit-learn pipelines with REST APIs

### Step 2 Status

- [x] FastAPI application created
- [x] Pydantic request validation implemented
- [x] Feature engineering implemented
- [x] Trained model loading implemented
- [x] POST `/predict` endpoint implemented
- [ ] API execution and prediction testing

## Step 3 — Streamlit Frontend
Goal
Build a simple interactive frontend using Streamlit. The frontend collects the customer's original information, sends it to the FastAPI POST /predict endpoint, and displays the predicted insurance premium category.
Architecture
Customer
   |
   v
Streamlit UI
(frontend/streamlit_app.py)
   |
   | HTTP POST with original customer JSON
   v
FastAPI
POST /predict
   |
   v
Pydantic validation + feature engineering
   |
   v
Trained Scikit-learn pipeline
(models/model.joblib)
   |
   v
Prediction JSON
   |
   v
Streamlit displays Low / Medium / High
Project Structure
frontend/
└── streamlit_app.py
Dependencies
Install Streamlit and Requests:
uv add streamlit requests
Run the application
Start the FastAPI backend in one terminal from the project root:
uv run uvicorn app.app:app --reload
Start Streamlit in a second terminal:
uv run streamlit run frontend/streamlit_app.py
Streamlit will print a local URL, usually:
http://localhost:8501
The frontend sends requests to:
http://127.0.0.1:8000/predict
To use a different API URL, set the INSURANCE_API_URL environment variable before starting Streamlit.

macOS/Linux:
export INSURANCE_API_URL="http://127.0.0.1:8000/predict"
uv run streamlit run frontend/streamlit_app.py
PowerShell:
$env:INSURANCE_API_URL = "http://127.0.0.1:8000/predict"
uv run streamlit run frontend/streamlit_app.py
Frontend inputs
The form collects the original customer fields:
- Age
- Weight (kg)
- Height (m)
- Annual income (LPA)
- Smoking status
- City
- Occupation
The frontend does not calculate BMI, age group, lifestyle risk, or city tier. It sends the original values to FastAPI, which validates the request and calculates those features before calling the model.
API request example
{
  "age": 32,
  "weight": 75.5,
  "height": 1.75,
  "income_lpa": 12.5,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "private_job"
}
API response example
{
  "predicted_category": "Medium"
}
The displayed category depends on the trained model and supplied dataset; this example is illustrative.
Error handling
The frontend handles common API failures:
- API connection errors (backend not running or unreachable)
- HTTP errors returned by FastAPI
- Request timeouts
- Invalid or unexpected JSON responses
Concepts covered
- Building interactive forms with Streamlit
- st.number_input, st.selectbox, and st.text_input
- Handling form submission with st.form
- Sending JSON requests using Python Requests
- Integrating a frontend with a REST API
- Displaying prediction results and handling API errors
- Configuring the API URL through an environment variable
Step 3 Status
- [x] Streamlit frontend implemented
- [x] Original customer data submitted to FastAPI
- [x] Prediction response displayed in the UI
- [x] Basic API error handling implemented
- [ ] End-to-end frontend and API testing