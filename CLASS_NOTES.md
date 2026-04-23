# EDGE Class Notes - ML Deployment End-to-End

Author note:
This document is designed for live teaching. You can follow it line by line.

## 1) Class Objective

Today students will learn:
1. How to train a simple ML model
2. How to export the model as PKL
3. How to serve predictions from a Flask API
4. How to run the same app in Docker
5. How cloud deployment works (Render flow + fallback)
6. Why Streamlit is useful for rapid ML demos

## 2) Teaching Storyline (Recommended)

Use this order in class:
1. Problem setup and architecture
2. Notebook training demo
3. Script-based retraining
4. Local Flask deployment
5. API testing (Postman)
6. Docker deployment
7. Render deployment concept
8. Streamlit rapid UI variant

## 3) Project Architecture (Explain First)

Data -> Train -> Export PKL -> Load model in Flask -> Predict API -> Deploy

Key files:
- [train_and_export.py](train_and_export.py)
- [app.py](app.py)
- [templates/index.html](templates/index.html)
- [notebooks/ml_deploy_demo.ipynb](notebooks/ml_deploy_demo.ipynb)
- [Dockerfile](Dockerfile)
- [render.yaml](render.yaml)
- [streamlit_app.py](streamlit_app.py)
- [.github/workflows/ci.yml](.github/workflows/ci.yml)

### Core Code Snapshot (Show Early)

Training script entry:

```python
# train_and_export.py (simplified)
MODEL_PATH = Path("models/student_pass_model.pkl")
MODEL_INFO_PATH = Path("models/model_info.json")

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

with open(MODEL_PATH, "wb") as f:
  pickle.dump(model, f)

model_info = {
  "active_version": model_version,
  "accuracy": round(float(accuracy), 3),
}
with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
  json.dump(model_info, f, indent=2)
```

## 4) Pre-Class Setup (Windows)

Run in PowerShell from project root:

python -m venv .venv                 # create virtual environment
.\.venv\Scripts\activate            # activate virtual environment
python -m pip install --upgrade pip setuptools wheel   # update tooling
python -m pip install -r requirements.txt --upgrade --no-cache-dir   # install dependencies

Quick health check:

.\.venv\Scripts\python.exe -m pytest -q   # should pass tests

## 5) Part A - Notebook Training Demo

Open notebook:
- [notebooks/ml_deploy_demo.ipynb](notebooks/ml_deploy_demo.ipynb)

Run cells in order:
1. Imports
2. Dummy data generation
3. Train/test split + model training
4. Accuracy report
5. Export PKL
6. Sample prediction

Talking points:
- Notebook is great for explanation and exploration
- Model output is saved for app usage
- We use fixed random seed for reproducibility

### 5A) Detailed Jupyter Instructor Walkthrough (Use This Live)

Goal of this part:
1. Show students how raw data becomes a deployable model.
2. Explain why notebook is for experimentation and script is for automation.

Before running notebook:

```powershell
# Ensure dependencies are installed in the same environment used by notebook kernel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --upgrade --no-cache-dir
```

Notebook run plan:
1. Run Cell 1 (markdown): set context and expected output of session.
2. Run Cell 2 (imports): confirm no import error.
3. Run Cell 3 (data generation): verify row count and preview.
4. Run Cell 4 (training): explain split, scaling, classifier, accuracy.
5. Run Cell 5 (artifact export): confirm `.pkl` save message.
6. Run Cell 6 (sample inference): confirm prediction and probability.

Cell-by-cell explanation script:

Cell 1:
1. Explain the objective: create data, train model, export artifact.
2. Tell students this notebook prepares deployment assets.

Cell 2:
1. Explain each import group:
  - data ops: `numpy`, `pandas`
  - ML: `train_test_split`, `Pipeline`, `StandardScaler`, `LogisticRegression`
  - evaluation: `accuracy_score`, `classification_report`
  - packaging: `pickle`
2. If import fails, stop and fix environment first.

Cell 3:
1. Show synthetic feature creation.
2. Explain score formula and thresholding into target label.
3. Explain why random seed (`42`) is used.
4. Show saved dataset path and preview table.

Cell 4:
1. Explain train/test split and why stratification matters.
2. Explain pipeline concept (scaler + classifier together).
3. Fit model and print accuracy/classification report.
4. Tell students accuracy may vary slightly on different data generation settings.

Cell 5:
1. Explain artifact export to `models/student_pass_model.pkl`.
2. Explain this file will be loaded by Flask and Streamlit.

Cell 6:
1. Show one manual sample input.
2. Display class (`Pass/Fail`) and probability.
3. Conclude: model is ready for API serving.

Expected notebook checkpoints:
1. Dataset file exists: `data/dummy_students.csv`.
2. Model file exists: `models/student_pass_model.pkl`.
3. Sample inference prints prediction and probability.

Jupyter troubleshooting (quick):
1. Kernel mismatch:
  - Symptom: import error despite pip install.
  - Fix: select `.venv` kernel and rerun all cells.
2. Stale state:
  - Symptom: variable not defined or inconsistent output.
  - Fix: restart kernel, run cells from top.
3. Path issue:
  - Symptom: file save path not found.
  - Fix: keep notebook in `notebooks/` folder and use provided relative paths.

## 6) Part B - Script-Based Training (Production Style)

Run:

.\.venv\Scripts\python.exe train_and_export.py   # retrain model and export files

Expected outputs:
- Model accuracy printed
- Latest model: models/student_pass_model.pkl
- Versioned model: models/student_pass_model_vTIMESTAMP.pkl
- Metadata: models/model_info.json

Explain to students:
- Notebook is for learning
- Script is for repeatable automation
- Versioning helps rollback and traceability

## 7) Part C - Local Flask Deployment

Run app:

.\.venv\Scripts\python.exe app.py   # start Flask server

Open browser:
- http://127.0.0.1:5000

What students should notice:
1. Input form fields
2. Prediction output
3. Model version displayed in UI

### Flask API Code (Explain Line by Line)

```python
@app.post("/predict")
def predict():
  payload = request.get_json(silent=True)
  is_valid, cleaned_payload, error_message = validate_payload(payload)
  if not is_valid:
    return jsonify({"error": error_message}), 400

  if MODEL is None:
    return jsonify({"error": "Model file not found."}), 503

  df = parse_features(cleaned_payload)
  prediction = int(MODEL.predict(df)[0])
  probability = float(MODEL.predict_proba(df)[0][1])

  return jsonify(
    {
      "prediction": prediction,
      "prediction_label": "Pass" if prediction == 1 else "Fail",
      "pass_probability": probability,
      "model_version": MODEL_VERSION,
      "features": cleaned_payload,
    }
  )
```

## 8) Part D - API Testing (Postman)

Method: POST
URL: http://127.0.0.1:5000/predict
Header: Content-Type: application/json

Sample body:
{
  "study_hours": 4.5,
  "attendance": 85,
  "assignments_completed": 7
}

Expected response fields:
1. prediction
2. prediction_label
3. pass_probability
4. model_version
5. features

## 9) Input Validation Rules in API

The API validates:
1. Required fields must exist
2. All values must be numeric
3. Allowed ranges:
- study_hours: 0 to 24
- attendance: 0 to 100
- assignments_completed: 0 to 20

If invalid input:
- API returns status 400 with error message

Demo invalid case:
{
  "study_hours": 5,
  "attendance": 150,
  "assignments_completed": 3
}

Validation helper code:

```python
VALID_RANGES = {
  "study_hours": (0.0, 24.0),
  "attendance": (0.0, 100.0),
  "assignments_completed": (0.0, 20.0),
}

def validate_payload(payload):
  if not isinstance(payload, dict):
    return False, None, "Request body must be a JSON object."

  missing = [f for f in FEATURES if f not in payload]
  if missing:
    return False, None, f"Missing required fields: {', '.join(missing)}"

  cleaned = {}
  for field in FEATURES:
    value = float(payload[field])
    min_v, max_v = VALID_RANGES[field]
    if not (min_v <= value <= max_v):
      return False, None, f"Field '{field}' must be between {min_v} and {max_v}."
    cleaned[field] = value
  return True, cleaned, None
```

## 10) Part E - Docker Deployment

Build image:

docker build -t edge-ml-demo:1.0 .   # important: keep space before dot

Run container:

docker run --name edge-ml-container -p 5000:5000 edge-ml-demo:1.0

Open:
- http://127.0.0.1:5000

Verify container:

docker ps                             # container should be running
docker logs --tail 20 edge-ml-container   # recent logs

### Local to Docker Safe Switch (Live Class)

Use this sequence to avoid confusion:

# stop local Flask manually with Ctrl + C first
.\.venv\Scripts\python.exe train_and_export.py     # refresh model
docker rm -f edge-ml-container                     # remove old container
docker build --no-cache -t edge-ml-demo:2.0 .     # rebuild clean image
docker run --name edge-ml-container -p 5001:5000 edge-ml-demo:2.0   # run on 5001

Open:
- http://127.0.0.1:5001

Dockerfile used in project:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 11) Part F - Render Deployment (Cloud)

Current project uses Python runtime in Render:
- env: python
- build command: pip install -r requirements.txt
- start command: gunicorn app:app

Important clarification:
- Render internal logs may look container-like
- That does not mean your Dockerfile is being used
- Dockerfile is used only when Docker runtime is selected

Blueprint flow:
1. Push code to GitHub
2. Render -> New + -> Blueprint
3. Select repo
4. Deploy

If payment info is required:
1. Use New + -> Web Service (manual)
2. Runtime: Python
3. Build: pip install -r requirements.txt
4. Start: gunicorn app:app
5. Choose Free instance if available

If no free option:
1. Do local + Docker live
2. Explain Render as walkthrough

## 12) Part G - Streamlit Variant (Rapid UI)

Run:

.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py

Why Streamlit:
1. Fast UI without writing frontend JS
2. Great for model demos and student interaction
3. Good for PoC and internal tools

Difference from Flask:
- Flask: stronger API and production web architecture
- Streamlit: faster demo/prototype workflow

Streamlit code sample:

```python
st.title("Student Pass Predictor - Streamlit")

study_hours = st.slider("Study hours per day", 0.0, 24.0, 4.5, 0.1)
attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0, 1.0)
assignments_completed = st.slider("Assignments completed", 0, 20, 7, 1)

if st.button("Predict"):
  sample = pd.DataFrame([{
    "study_hours": float(study_hours),
    "attendance": float(attendance),
    "assignments_completed": float(assignments_completed),
  }])
  pred = int(model.predict(sample)[0])
  prob = float(model.predict_proba(sample)[0][1])
  st.success(f"Prediction: {'Pass' if pred == 1 else 'Fail'}")
  st.write(f"Pass probability: {prob:.3f}")
```

## 13) CI/CD Overview

File:
- [.github/workflows/ci.yml](.github/workflows/ci.yml)

Pipeline runs on push and pull request:
1. Install dependencies
2. Train model artifact
3. Run tests

Why this matters:
- Prevents breaking changes
- Adds confidence before deployment

GitHub Actions workflow code:

```yaml
name: CI

on:
  push:
    branches: ["main", "master"]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install --upgrade pip
      - run: pip install -r requirements.txt
      - run: python train_and_export.py
      - run: pytest -q
```

## 14) Common Errors and Quick Fixes

1. ModuleNotFoundError
- Cause: wrong Python environment
- Fix: activate .venv and reinstall requirements

2. Docker buildx requires 1 argument
- Cause: missing build context
- Fix: docker build -t edge-ml-demo:1.0 .

3. Port already in use
- Fix: run Docker with a different host port, for example 5001

4. Model file not found
- Fix: run train_and_export.py before app start or Docker build

5. WinError 32 during pip install
- Cause: locked files by running Python/Jupyter processes
- Fix: stop background Python processes, then reinstall

## 15) Suggested 70-Minute Class Timing

1. Intro and architecture: 10 min
2. Notebook training: 15 min
3. Script training and versioning: 10 min
4. Local Flask + Postman: 10 min
5. Docker deployment: 15 min
6. Render + Streamlit + Q&A: 10 min

## 16) 60-Second Speaking Script (Ready to Read)

Today we are taking one ML model from notebook to real deployment.
First, we train and export a PKL model.
Then, we serve predictions from a Flask app and test the API.
Next, we package the same app with Docker so it runs consistently anywhere.
After that, we discuss cloud deployment with Render.
Finally, we show a Streamlit variant for rapid demo UI.
The key lesson is this: notebook helps us learn, script helps us automate, API helps us integrate, and deployment helps us deliver real value.

## 17) Command Cheat Sheet (Fast Reference)

Environment:
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt --upgrade --no-cache-dir

Train:
.\.venv\Scripts\python.exe train_and_export.py

Local run:
.\.venv\Scripts\python.exe app.py

Tests:
.\.venv\Scripts\python.exe -m pytest -q

Docker:
docker build -t edge-ml-demo:1.0 .
docker run --name edge-ml-container -p 5000:5000 edge-ml-demo:1.0

Streamlit:
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py

## 18) End-to-End Classroom Runbook (No-Gap Step-by-Step)

Use this exact sequence in live class:

Step 1 - Environment setup:

```powershell
python -m venv .venv                                  # create env once
.\.venv\Scripts\activate                             # activate env
python -m pip install --upgrade pip setuptools wheel  # tooling update
python -m pip install -r requirements.txt --upgrade --no-cache-dir  # install deps
```

Step 2 - Notebook demonstration:
1. Open [notebooks/ml_deploy_demo.ipynb](notebooks/ml_deploy_demo.ipynb).
2. Run Cell 1 to Cell 6 in order.
3. Confirm dataset and PKL generation in outputs.

Step 3 - Script demonstration (same logic, automated):

```powershell
.\.venv\Scripts\python.exe train_and_export.py   # one-command retraining and export
```

Step 4 - Flask local app:

```powershell
.\.venv\Scripts\python.exe app.py                # start API + HTML UI
```

Step 5 - Browser check:
1. Open `http://127.0.0.1:5000`.
2. Input values and click Predict.
3. Verify prediction + probability + model version.

Step 6 - Postman API check:
1. Method: `POST`
2. URL: `http://127.0.0.1:5000/predict`
3. Header: `Content-Type: application/json`
4. Send valid JSON body and verify 200 response.
5. Send invalid JSON body and verify 400 response.

Step 7 - Docker build and run:

```powershell
docker build -t edge-ml-demo:1.0 .                  # image build
docker run --name edge-ml-container -p 5000:5000 edge-ml-demo:1.0  # container run
```

Step 8 - Docker conflict-safe mode (recommended during class):

```powershell
# Stop local Flask terminal manually with Ctrl + C first
docker rm -f edge-ml-container                      # clear previous container
docker build --no-cache -t edge-ml-demo:2.0 .      # rebuild latest code
docker run --name edge-ml-container -p 5001:5000 edge-ml-demo:2.0  # run on alternate host port
```

Step 9 - Cloud deployment explanation (Render):
1. Explain runtime from [render.yaml](render.yaml): Python, not Docker.
2. Show build command and start command.
3. Explain payment fallback (manual web service + free instance if available).

Step 10 - Streamlit variant:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py  # rapid UI variant
```

Step 11 - CI/CD explanation:
1. Open [.github/workflows/ci.yml](.github/workflows/ci.yml).
2. Explain push/PR triggers.
3. Explain automated training check + tests.

Step 12 - Wrap-up key lesson:
1. Notebook = learning and experimentation.
2. Script = repeatability and automation.
3. API = integration point.
4. Docker = environment consistency.
5. CI/CD = stability and team confidence.

## 20) Instructor Speaking Prompts (Detailed)

Use these prompts during each phase:

Notebook phase prompt:
"Now we are generating controlled dummy data, training a simple classifier, and exporting a real deployment artifact."

Flask/API phase prompt:
"This endpoint validates input first, then predicts using the model, then returns structured JSON for integration."

Docker phase prompt:
"Docker freezes app + dependencies so the same build runs consistently across machines."

Render phase prompt:
"Our Render setup uses Python runtime commands, so Dockerfile is not mandatory for this path."

Streamlit phase prompt:
"Streamlit is the fastest way to demonstrate model interaction when you do not need full frontend engineering."

## 19) Ready-to-Copy API Examples

Valid request body:

```json
{
  "study_hours": 4.5,
  "attendance": 85,
  "assignments_completed": 7
}
```

Invalid request body (attendance বেশি):

```json
{
  "study_hours": 4.5,
  "attendance": 120,
  "assignments_completed": 7
}
```

Expected error response:

```json
{
  "error": "Field 'attendance' must be between 0.0 and 100.0."
}
```

## 18) Homework Ideas for Students

1. Add one new feature column and retrain model
2. Add API authentication token
3. Add confusion matrix in notebook
4. Deploy Streamlit version to a free cloud platform
