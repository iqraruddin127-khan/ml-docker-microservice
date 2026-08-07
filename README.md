# End-to-End Machine Learning Microservice Pipeline

This repository contains a production-ready Machine Learning microservice that fulfills all assignment criteria, featuring experiment tracking, an API serving framework, complete containerization, and monitoring telemetry.

## 📁 Repository Structure
* `main.py` - FastAPI application serving model predictions and hosting system health checks.
* `train.py` - Script containing the pipeline architecture and training workflow.
* `Dockerfile` - Container configuration file packaging the system environment.
* `requirements.txt` - Python dependency manifests.
* `mlruns/` - MLflow tracking database containing recorded parameters and metrics.

---

## 🛠️ How the Assignment Requirements Were Met

### 1. Build and Deploy an ML Model with FastAPI
The machine learning model is served via a high-performance **FastAPI** application (`main.py`). 
* It implements data validation using **Pydantic** (`BaseModel`) to protect endpoints against malformed payloads.
* It exposes a secure HTTP `POST /predict` endpoint to accept input features and return model inferences in a clean JSON format.

### 2. MLflow Tracking Implementation
Experiment lifecycle management is fully handled using **MLflow Tracking**.
* During execution, the system logs hyperparameters (such as model estimators and random states) using `mlflow.log_param()`.
* Model performance metrics (Accuracy, F1-scores) are logged cleanly using `mlflow.log_metric()`.
* Every successful training cycle serializes and registers a versioned artifact using `mlflow.sklearn.log_model()`, populating the local `mlruns/` directory.

### 3. Dockerized ML Microservice
The entire microservice is completely containerized utilizing the project's **`Dockerfile`**. This guarantees that the service can run smoothly in an isolated sandbox environment on any operating system without configuration drift.
* **Base Image:** Built on an optimized, lightweight `python:3.9-slim` image.
* **Isolation:** The system sets up an isolated `/app` working directory, installs verified locked dependencies, and exposes port `8000`.

### 4. Full Deployment Pipeline & Monitoring
Active telemetry endpoints are embedded into the deployment lifecycle to satisfy microservice observability standards:
* **Health Check API:** A dedicated `GET /health` endpoint actively monitors application stability, tracking container health status and verification flags.
* **Payload Monitoring:** The `/predict` inference route returns calculated class confidence probabilities along with the predictions, delivering live analytical transparency for system auditing.

---

## 🚀 How to Run and Evaluate This Project Locally

To pull down this repository, build the image container, and execute the service, run the following sequential commands in your terminal:

```bash
# 1. Clone the repository
git clone <PASTE_YOUR_GITHUB_REPOSITORY_LINK_HERE>
cd ml-docker-microservice

# 2. Build the Docker Image Container
docker build -t ml-assignment:latest .

# 3. Boot up the Microservice Container
docker run -d -p 8000:8000 --name ml-service ml-assignment:latest

# 4. Test Monitoring Telemetry Check
curl http://localhost:8000/health
```
