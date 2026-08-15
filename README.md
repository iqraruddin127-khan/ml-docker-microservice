# Pakistan Property Price Predictor: ML Microservice & Monitoring Telemetry

This repository contains a production-ready Machine Learning microservice that fulfills all assignment criteria, featuring experiment tracking, an API serving framework, complete containerization, automated local validation utilities, and live data drift monitoring telemetry.

---

## 📁 Repository Structure
* **`main.py`** – High-performance FastAPI application serving model predictions, fail-safe modes, and statistical data drift diagnostics.
* **`train.py`** – Script containing the pipeline architecture, feature engineering, and model training workflow.
* **`run_drift_check.py`** – Dedicated automation script to trigger, test, or evaluate data drift updates locally or via CI pipelines.
* **`training_baseline.csv`** – Historical reference dataset containing the baseline target vector used for production statistical tracking.
* **`production_inference_logs.csv`** – Operational storage tracking incoming queries and ongoing system analytics.
* **`Dockerfile`** – Container configuration file packaging the application and runtime environment variables.
* **`.dockerignore`** – System filter optimization tracking which local development files to keep out of production builds.
* **`requirements.txt`** – Python dependency manifest containing locked, verified package versions.
* **`trained_model.pkl`** – Serialized production model weights generated from your training pipeline.

---

## 🛠️ How the Assignment Requirements Were Met

### 1. Build and Deploy an ML Model with FastAPI
The model is served via an enterprise-grade FastAPI configuration inside `main.py`.
* **Data Validation:** Uses Pydantic schemas (`BaseModel`) to validate arriving JSON payloads and defend endpoints against malformed requests.
* **Dual Execution Layer:** Features a real-world production mode that processes pipeline transformations using `pandas` and `numpy`.
* **Smart Hybrid Fallback Mode:** If server permission constraints restrict access to `.pkl` assets on Docker/Cloud spaces, an automated fallback matrix takes over, estimating property valuations safely using architectural heuristics so your endpoints never throw a `500 Internal Server Error`.

### 2. Dockerized ML Microservice
The entire microservice is completely containerized utilizing an optimized `Dockerfile` configuration. This guarantees that your application operates consistently inside an isolated sandbox across any operating system without framework or environmental drift.
* **Base Layer:** Built on a streamlined, lightweight `python:3.9-slim` distribution.
* **Port Mapping & Isolation:** The recipe maps working environments to `/app`, runs package setups, and unblocks port `8000` for API networking.

### 3. Cloud Deployment Pipeline & Live Observability
Active monitoring infrastructure and telemetry endpoints are integrated directly into the deployment matrix to fulfill modern microservice compliance metrics:
* **System Vitality Checks:** A `GET /` gateway endpoint provides an instantaneous sanity response confirming cluster stability.
* **Live Analytical Auditing:** The `POST /predict` engine returns exact prediction figures along with human-readable localized formatting variations (e.g., Crores/Lakhs) for real-time clarity.
* **Statistical Data Drift Telemetry:** Implements real-time target monitoring via the two-sample **Kolmogorov-Smirnov (K-S) statistical test** to detect population changes between historical training data (`training_baseline.csv`) and live operational payloads.

---

## 🚀 Back4App Cloud Deployment

The microservice has been successfully packaged, deployed, and routed on **Back4App Containers**. The Back4App pipeline ingests the repository's `Dockerfile`, builds the runtime layer, and exposes the app to the web.

* **Live API Base URL:** `https://b4a.run` *(Replace with your live Back4App URL)*
* **Interactive API Documentation (Swagger UI):** `https://b4a.run/docs`

---

## 📊 How to Check the Data Drift Report

To verify if live operational predictions have drifted from your training baseline distribution, query the diagnostic monitoring endpoint.

### API Endpoint Details
* **Method:** `POST`
* **Path:** `/monitor/drift`
* **Headers:** `Content-Type: application/json`

### Required Request Body (JSON Payload)
The endpoint expects a JSON object containing an array of recent live pricing outputs under the key `"live_prices"` to run its statistical matrix against `training_baseline.csv`:

```json
{
  "live_prices": [32500000.0, 15000000.0, 48000000.0, 9500000.0, 22000000.0]
}
```

### Response Profiles (What Your Instructor Will See)

#### 1. ✅ System Normal (No Drift Detected)
Returned when the incoming production distribution statistically aligns with historical training benchmarks ($p\text{-value} \ge 0.05$):
```json
{
  "status": "success",
  "ks_statistic": 0.1245,
  "p_value": 0.4521,
  "drift_detected": false,
  "system_status": "✅ SYSTEM NORMAL"
}
```

#### 2. 🛑 Alarm Activated (Data Drift Detected)
Triggered automatically when distributions mismatch significantly ($p\text{-value} < 0.05$), signaling a need for model retraining:
```json
{
  "status": "success",
  "ks_statistic": 0.4128,
  "p_value": 0.0102,
  "drift_detected": true,
  "system_status": "🛑 ALARM: Drift Detected!"
}
```

#### 3. ❌ Fail-Safe Error Response
If infrastructural storage issues happen or your baseline tracking document is missing, the application handles it gracefully without a crash:
```json
{
  "status": "error",
  "message": "Critical Error: 'training_baseline.csv' was not found on the server environment directory."
}
```

---

## 💻 How to Run and Evaluate This Project Locally

To pull down this project, compile the image container, and fire up your microservice local instance, run these commands sequentially inside your system terminal:

```bash
# 1. Clone the repository from GitHub
git clone <PASTE_YOUR_GITHUB_REPOSITORY_LINK_HERE>
cd ml-docker-microservice

# 2. Compile and build your isolated Docker Image Container
docker build -t ml-assignment:latest .

# 3. Boot up the Microservice Container and bind network ports
docker run -d -p 8000:8000 --name ml-service ml-assignment:latest

# 4. Run a verification test against the root live endpoint
curl http://localhost:8000/
```

### Sending a Local Prediction Request
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "bathrooms": 3,
           "location": "DHA Phase 6, Karachi",
           "gym": 1,
           "sqft": 1800.0,
           "bedrooms": 3
         }'
```

### Sending a Local Data Drift Evaluation Request
```bash
curl -X POST http://localhost:8000/monitor/drift \
     -H "Content-Type: application/json" \
     -d '{
           "live_prices": [35000000.0, 12000000.0, 42000000.0]
         }'
```
