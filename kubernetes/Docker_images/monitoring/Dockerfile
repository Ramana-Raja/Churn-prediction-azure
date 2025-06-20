FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY ../codes/monitoring/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /mlops

COPY ../codes/monitoring/monitoring.py .

CMD ["python", "monitoring.py"]