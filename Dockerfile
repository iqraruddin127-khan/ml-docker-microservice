# Use a lightweight Python image
FROM python:3.10-slim

# Set workspace directory
WORKDIR /app

# Copy requirement listings first
COPY requirements.txt .

# Install dependencies smoothly
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy all code files into the container
COPY . .

# RUN THE TRAINING SCRIPT CODE FIRST TO LOCALLY GENERATE THE PICKLE FILES
RUN python train.py

# Expose communications port
EXPOSE 8000

# Start up your FastAPI microservice production server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
