FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    azure-storage-blob \
    pyarrow

WORKDIR /mlops

COPY codes/train_model.py .

CMD ["python", "train_model.py"]