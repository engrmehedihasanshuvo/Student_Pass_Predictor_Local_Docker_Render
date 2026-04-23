# ML Deployment Demo

A beginner-friendly end-to-end ML deployment project for class/demo use.

This project shows one trained ML model deployed in 3 ways:
1. Local (Flask)
2. Docker
3. Render (Python runtime)

It now includes:
1. Input validation for `/predict`
2. Model versioning with metadata
3. CI/CD pipeline via GitHub Actions
4. Streamlit frontend variant

## What This Project Does

Pipeline:
Data -> Train Model -> Export PKL -> Flask API -> Deploy

You will:
1. Generate a dummy student dataset
2. Train a Logistic Regression model
3. Export model to `.pkl`
4. Serve predictions via Flask UI + API
5. Deploy locally, in Docker, and on cloud

## Project Structure

```text
EDGE/
|-- app.py
|-- train_and_export.py
|-- requirements.txt
|-- Dockerfile
|-- render.yaml
|-- data/
|   `-- dummy_students.csv
|-- models/
|   `-- student_pass_model.pkl
|-- notebooks/
|   `-- ml_deploy_demo.ipynb
|-- templates/
|   `-- index.html
`-- README.md
```

## Prerequisites

1. Python 3.11+
2. Git
3. Docker Desktop (for Docker demo)
4. A GitHub account (for cloud deploy)

## Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt --upgrade --no-cache-dir
```

## Step 1: Train Model and Export PKL

```powershell
.\.venv\Scripts\python.exe train_and_export.py
```

Expected output:
1. Model accuracy shown in terminal
2. PKL saved at `models/student_pass_model.pkl`

## Step 2: Run Local Flask App

```powershell
.\.venv\Scripts\python.exe app.py
```

Open:
http://127.0.0.1:5000

## Step 3: Test API (Postman or curl)

Endpoint:
- `POST /predict`

Sample JSON body:

```json
{
  "study_hours": 4.5,
  "attendance": 85,
  "assignments_completed": 7
}
```

curl example:

```powershell
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":4.5,\"attendance\":85,\"assignments_completed\":7}"
```

### Input Validation Rules (`/predict`)

Required fields:
1. `study_hours` (numeric, range: 0 to 24)
2. `attendance` (numeric, range: 0 to 100)
3. `assignments_completed` (numeric, range: 0 to 20)

If validation fails, API returns `400` with an error message.

## Model Versioning

After each training run:
1. Latest model is saved to `models/student_pass_model.pkl`
2. Versioned model is saved to `models/student_pass_model_vYYYYMMDD-HHMMSS.pkl`
3. Metadata is saved to `models/model_info.json`

API response includes `model_version` so you can verify which model served the prediction.

## Step 4: Docker Deployment

Build image:

```powershell
docker build -t edge-ml-demo:1.0 .
```

Run container:

```powershell
docker run --name edge-ml-container -p 5000:5000 edge-ml-demo:1.0
```

Open:
http://127.0.0.1:5000

### Local -> Docker Safe Switch (Recommended for Live Class)

```powershell
# Stop local Flask first (Ctrl + C in local terminal)
.\.venv\Scripts\python.exe train_and_export.py
docker rm -f edge-ml-container
docker build --no-cache -t edge-ml-demo:2.0 .
docker run --name edge-ml-container -p 5001:5000 edge-ml-demo:2.0
```

Verify Docker is serving:

```powershell
docker ps
docker logs --tail 20 edge-ml-container
```

Open:
http://127.0.0.1:5001

## Step 5: Render Deployment

This repo contains `render.yaml` configured as Python runtime.

Important:
1. This uses `env: python` (not `env: docker`)
2. Build command: `pip install -r requirements.txt`
3. Start command: `gunicorn app:app`

Deploy flow:
1. Push code to GitHub
2. Render dashboard -> New + -> Blueprint
3. Connect repo
4. Deploy

### If Blueprint Asks for Payment

Use fallback:
1. Render dashboard -> New + -> Web Service (manual)
2. Runtime: Python
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
5. Select Free instance (if available)

If free instance is unavailable in your account:
1. Use Local + Docker for live demo
2. Show Render as walkthrough only

## Streamlit Frontend Variant

Run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local Streamlit URL shown in terminal.

## CI/CD Pipeline (GitHub Actions)

Workflow file:
- `.github/workflows/ci.yml`

On push and pull request, pipeline runs:
1. Install dependencies
2. Train model artifact
3. Run test suite (`pytest`)

## Notebook Flow (Teaching Friendly)

Open:
- `notebooks/ml_deploy_demo.ipynb`

Run cells in order:
1. Imports
2. Dummy data generation
3. Model training
4. Evaluation
5. PKL export
6. Sample prediction

## Common Errors and Fixes

1. `ModuleNotFoundError: numpy`
- Cause: wrong Python env
- Fix: use `.venv` Python and install requirements again

2. Docker error: `buildx build requires 1 argument`
- Cause: missing build context path
- Fix: use `docker build -t edge-ml-demo:1.0 .`

3. Port already in use
- Fix: run Docker on a different host port, e.g. `-p 5001:5000`

4. PKL not found
- Fix: run `train_and_export.py` before starting app or building Docker image

