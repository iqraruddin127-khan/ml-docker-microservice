# Pakistan Property Price Predictor: ML Microservice & Monitoring Telemetry

This repository contains a production-ready Machine Learning microservice that fulfills all assignment criteria, featuring experiment tracking, an API serving framework, complete containerization, automated local validation utilities, and data drift monitoring telemetry.

---

## 📁 Repository Structure
* **`main.py`** – High-performance FastAPI application serving model predictions, fail-safe modes, and statistical data drift diagnostics.
* **`train.py`** – Script containing the pipeline architecture, feature engineering, and model training workflow.
* **`run_drift_check.py`** – Dedicated automation script to trigger, test, or evaluate data drift updates locally.
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
* **Smart Hybrid Fallback Mode:** If server permission constraints restrict access to `.pkl` assets on Docker spaces, an automated fallback matrix takes over, estimating property valuations safely using architectural heuristics so your endpoints never throw an error.

### 2. Dockerized ML Microservice
The entire microservice is completely containerized utilizing an optimized `Dockerfile` configuration. This guarantees that your application operates consistently inside an isolated sandbox across any operating system without framework or environmental drift.
* **Base Layer:** Built on a streamlined, lightweight `python:3.9-slim` distribution.
* **Port Mapping & Isolation:** The recipe maps working environments to `/app`, runs package setups, and unblocks port `8000` for API networking.

### 3. Statistical Data Drift Telemetry
Implements real-time target monitoring via the two-sample **Kolmogorov-Smirnov (K-S) statistical test** to detect population changes between historical training data (`training_baseline.csv`) and live operational payloads.

---

## 📊 How to Check the Data Drift Report (API Endpoint)

To verify if live operational predictions have drifted from your training baseline distribution, query the diagnostic monitoring endpoint.

### API Endpoint Details
* **Method:** `POST`
* **Path:** `/monitor/drift`
* **Headers:** `Content-Type: application/json`

### Required Request Body (JSON Payload)
The endpoint expects a JSON object containing an array of recent live pricing outputs under the key `"live_prices"` to run its statistical matrix against `training_baseline.csv`:

```json
{
  "live_prices": [85000000, 98000000, 125000000, 130000000, 155000000, 195000000]
}
```

### Expected Response Profile
```json
{
  "status": "success",
  "ks_statistic": 1,
  "p_value": 0.0238,
  "drift_detected": true,
  "system_status": "🛑 ALARM: Drift Detected!"
}
```

---

## 💻 How to Run and Evaluate This Project Locally

To pull down this project, compile the image container, and fire up your microservice local instance, run these commands sequentially inside your system terminal:

```bash
# 1. Clone the repository from GitHub
git clone <https://github.com/iqraruddin127-khan/ml-docker-microservice>
cd ml-docker-microservice

# 2. Compile and build your isolated Docker Image Container
docker build -t ml-assignment:latest .

# 3. Boot up the Microservice Container and bind network ports
docker run -d -p 8000:8000 --name ml-service ml-assignment:latest

# 4. Run a verification test against the root live endpoint
curl http://localhost:8000/
```

### Testing Predictions Locally via cURL
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "bathrooms": 1,
           "location": "affandi town",
           "gym": 0,
           "sqft": 2000,
           "bedrooms": 6
         }'
```

### Testing Data Drift Evaluation Locally via cURL
```bash
curl -X POST http://localhost:8000/monitor/drift \
     -H "Content-Type: application/json" \
     -d '{
           "live_prices": [85000000, 98000000, 125000000, 130000000, 155000000, 195000000]
         }'
```

### Running the Alternative Monitoring Pipeline Script Locally
Your can also run the standalone statistical evaluation script directly using Python:
```bash
python run_drift_check.py
```
