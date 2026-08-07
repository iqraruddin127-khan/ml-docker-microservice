FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Use --prefer-binary to pull pre-compiled packages instead of building them in RAM
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
