FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

# FORCE MLFLOW TO USE AN IN-MEMORY TEMPORARY TRACKING URI BYPASSING DISK PERMISSIONS
ENV MLFLOW_TRACKING_URI=sqlite:///:memory:

# Force train the model during build to create the .pkl assets locally
RUN python train.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
