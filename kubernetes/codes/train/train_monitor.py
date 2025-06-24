from prometheus_api_client import PrometheusConnect
from retrain import retrain

def get_latest_drift_value():
    PROMETHEUS_URL = "http://prometheus.monitoring.svc.cluster.local:9090"

    query = 'data_drift_value{instance="monitor.monitoring.svc.cluster.local:8000"}'
    prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)
    result = prom.custom_query(query=query)

    drift_value = float(result[0]["value"][1])
    return drift_value


if __name__ == "__main__":
    drift_value = get_latest_drift_value()
    if drift_value > 0.05:
        retrain()

