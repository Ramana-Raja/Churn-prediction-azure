FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r mlopsuser && useradd -r -g mlopsuser mlopsuser

COPY ../requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /mlops

COPY ../retrain.py .
COPY ../train_monitor.py .

RUN chown -R mlopsuser:mlopsuser /mlops

USER mlopsuser

CMD ["python", "train_monitor.py"]
