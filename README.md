# Churn Prediction Monitoring System

A production-grade MLOps monitoring pipeline for detecting data drift, logging events, exporting metrics, and tracing operations in a customer churn prediction system. Built with Kubernetes and integrated with Azure services, the pipeline supports full observability through Prometheus, Grafana, Loki, Tempo, and OpenTelemetry.

---

## 📌 Features

* **Model Monitoring with Evidently**

  * Tracks data drift, missing values, and feature distribution shifts.
* **Metrics Export via Prometheus**

  * Exposes custom metrics like `data_drift_value`, `drifted_columns_count`, and `missing_value_count` on a `/metrics` endpoint.
* **Logging via Loki & Promtail**

  * Streams application logs from mounted volumes to Grafana Loki.
* **Tracing with OpenTelemetry + Tempo**

  * Captures fine-grained traces for model data load, inference, and monitoring logic.
* **Kubernetes-native Deployment**

  * Deploys all services (monitor, Prometheus, Grafana, Loki, Tempo) on AKS.
* **Azure Blob Storage**

  * Securely loads training, test datasets, and serialized models from Azure Blob.

---

## 📁 Project Structure

```
Azure
├── codes
│   ├── monitoring
│   │     ├── DokerFile            # Image for running monitoring.py
│   │     ├── monitoring.py        # Monitoring of the entire kubernetes
│   │     └── requirements.txt     # Requirements to run monitoring.py
│   └── train
│         ├── Dockerfile           # Image for running retrain.py and train_monitor.py
│         ├── retrain.py           # Retrains model
│         ├── train_monitor.py     # Checks if data_drift_value > 0.05
│         └── requirements.txt     # Requirements to run retrain.py and train_monitor.py
│          
└── kubernets/
    ├── config-maps/                # contains config for prometheus, tempo, loki, promtail, monitoring.py     
    ├── Cronjob/                    # contains Cronjob.yaml which checks whether model needs to be retrained or not
    ├── daemonset/                  # contains node-exporter.yaml and promtail.ymal
    ├── deployments/                # contains yaml files of grafana, loki, prometheus, monitoring.py, tempo
    ├── PersistentVolumeClaims/     # contians PersistentVolumeClaims for storing grafana, loki, promethesus and promtail data
    └── service/                    # contains service fo grafana, loki, monitor, node-exporter, prometheus, tempo
                                      so that they can communicate with each other.

```

---

## ⚙️ Tools & Tech

* **Python** – Prediction & monitoring logic
* **Evidently** – Drift detection
* **Prometheus** – Metric scraping
* **Grafana** – Dashboards for metrics and logs
* **Loki + Promtail** – Centralized logging
* **Tempo** – Distributed tracing backend
* **OpenTelemetry** – Trace generation
* **Azure Blob Storage** – Artifact management
* **Kubernetes (AKS)** – Orchestrated deployments

---

## 🚀 Usage
* Make sure credentials are set in config maps
### 1. Apply PersistentVolumeClaims

```bash
kubectl apply -f Azure/kubernetes/PersistentVolumeClaims/
```

### 2. Start applying Config Maps

```bash
kubectl apply -f Azure/kubernetes/config-maps/
```

### 3. Apply deployments

```bash
kubectl apply -f Azure/kubernetes/deployments/
```

### 4. Apply daemonset

```bash
kubectl apply -f Azure/kubernetes/daemonset/
```

### 5. Apply CronJob

```bash
kubectl apply -f Azure/kubernetes/CronJob/
```

### 6. Apply services
```bash
kubectl apply -f Azure/kubernetes/service/
```

### Get Access To Grafana Dashboard
```bash
kubectl port-forward <grafana-deployment-id> -n monitoring 3000:3000
```

* Go to localhost:3000
* Go to Add Data Sources and add Tempo, Loki, Prometheus

---

## 🔍 Metrics Exposed to Prometheus

| Metric Name             | Description                          |
| ----------------------- | ------------------------------------ |
| `data_drift_value`      | Evidently-calculated drift value     |
| `drifted_columns_count` | Number of drifted features           |
| `missing_value_count`   | Count of missing values in inference |

---
## ⚙️ Model Retraining
Model retraining is checked by CronJob "cronjob-retrain.yaml" which retrain the model whenever the data_drift_value exceeds 0.05.

## 🧠 Credits

Built by **Ramana Raja** as part of a comprehensive MLOps monitoring project.
